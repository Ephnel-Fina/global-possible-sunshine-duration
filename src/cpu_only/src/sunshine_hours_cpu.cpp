#include <cmath>
#include <limits>
#include <algorithm>
#include "sunshine_hours_cpu.h"
#include <iostream>

namespace
{
    constexpr double PI = 3.14159265358979323846;

    inline double deg2rad(double d) { return d * PI / 180.0; }

    // Solar declination formula consistent with the CUDA version
    double calculateSolarDeclination(int day_of_year)
    {
        double tau = 2.0 * PI * (day_of_year - 1) / 365.2422;
        double delta = 0.006894
                     - 0.399512 * std::cos(tau)
                     + 0.072075 * std::sin(tau)
                     - 0.006799 * std::cos(2 * tau)
                     + 0.000896 * std::sin(2 * tau)
                     - 0.002689 * std::cos(3 * tau)
                     + 0.001516 * std::sin(3 * tau);
        return delta;
    }

    // Compute sunrise/sunset hour-angle range from latitude phi and declination delta
    // Returns omega_r and omega_s in radians; returns false for polar night
    bool computeDaytimeRange(double phi, double delta,
                             double &omega_r, double &omega_s)
    {
        // cos(H0) = -tan(phi) * tan(delta)
        double cosH0 = -std::tan(phi) * std::tan(delta);

        if (cosH0 >= 1.0)
        {
            // Polar night: the sun remains below the horizon all day
            return false;
        }

        if (cosH0 <= -1.0)
        {
            // Polar day: sunshine is possible throughout the day
            omega_s = PI;
            omega_r = -PI;
            return true;
        }

        double H0 = std::acos(std::max(-1.0, std::min(1.0, cosH0)));
        omega_s = H0;
        omega_r = -H0;
        return true;
    }

    // Compute solar altitude h and azimuth A for a given hour angle; azimuth is clockwise from north in [0, 2*pi)
    void computeSunPosition(double phi, double delta, double omega,
                            double &h, double &A)
    {
        double sin_phi = std::sin(phi);
        double cos_phi = std::cos(phi);
        double sin_delta = std::sin(delta);
        double cos_delta = std::cos(delta);

        double sin_h = sin_phi * sin_delta + cos_phi * cos_delta * std::cos(omega);
        sin_h = std::max(-1.0, std::min(1.0, sin_h));
        h = std::asin(sin_h);

        // If the sun is below the horizon, altitude <= 0 and azimuth is not meaningful
        if (h <= 0.0)
        {
            A = 0.0;
            return;
        }

        double cos_h = std::cos(h);
        double sinA = cos_delta * std::sin(omega) / cos_h;
        double cosA = (sin_h * sin_phi - sin_delta) / (cos_h * cos_phi);

        // Clamp for numerical safety
        sinA = std::max(-1.0, std::min(1.0, sinA));
        cosA = std::max(-1.0, std::min(1.0, cosA));

        A = std::atan2(sinA, cosA); // return range (-pi, pi]
        if (A < 0.0)
            A += 2.0 * PI;          // convert to [0, 2pi)
    }

    // Approximate meters per degree of latitude/longitude at a given latitude
    void metersPerDegree(double phi, double &m_per_deg_lat, double &m_per_deg_lon)
    {
        // Common approximation
        double cosphi = std::cos(phi);
        double cos2phi = std::cos(2.0 * phi);
        double cos4phi = std::cos(4.0 * phi);

        // Latitude direction, roughly 111 km per degree
        m_per_deg_lat = 111132.954 - 559.822 * cos2phi + 1.175 * cos4phi;
        // Longitude direction
        m_per_deg_lon = 111132.954 * cosphi;
    }

    // Single-DEM ray tracing: determine whether the pixel is shadowed at solar altitude h_sun and azimuth A_sun
    bool isVisibleSingleDem(int row0, int col0,
                            double phi,            // pixel latitude in radians
                            double h_sun,          // solar altitude in radians, > 0
                            double A_sun,          // solar azimuth in radians, clockwise from north
                            const RasterWithLatitude &dem)
    {
        const int rows = dem.rows;
        const int cols = dem.cols;
        const float *elev = dem.data.get();
        const double *gt = dem.geo_transform;

        int idx0 = row0 * cols + col0;
        float base_h = elev[idx0];
        if (std::isnan(base_h))
            return false; // Invalid source pixel; treat it as not visible

        // Convert one DEM pixel step to meters
        double m_per_deg_lat, m_per_deg_lon;
        metersPerDegree(phi, m_per_deg_lat, m_per_deg_lon);

        // Pixel spacing in longitude/latitude degrees
        double dlon_deg = gt[1];        // usually positive
        double dlat_deg = gt[5];        // usually negative for north-up rasters

        // Index-space step along the ray, in pixel units
        // A_sun = 0 points north,pi/2 points east
        double step_col = std::sin(A_sun);   // column direction; positive is eastward
        double step_row = -std::cos(A_sun);  // row direction; positive is southward because row indices increase downward

        // Ground distance per step in meters
        // First compute meters per pixel in longitude and latitude directions
        double dx_m = std::abs(dlon_deg) * m_per_deg_lon;
        double dy_m = std::abs(dlat_deg) * m_per_deg_lat;

        // Actual horizontal distance per step in this direction, weighted by step_row/step_col
        double step_ground_m = std::sqrt(
            (step_col * dx_m) * (step_col * dx_m) +
            (step_row * dy_m) * (step_row * dy_m));

        if (step_ground_m <= 0.0)
            return true; // Should not happen; defensively return visible

        // Maximum tracing distance in meters; adjust as needed
        // For small 0.5 x 0.5 degree tiles, 50 km is usually sufficient
        const double MAX_TRACE_DISTANCE_M = 50000.0;
        int max_steps = static_cast<int>(MAX_TRACE_DISTANCE_M / step_ground_m);
        if (max_steps < 1)
            max_steps = 1;

        // March along the ray and check whether terrain elevation angle exceeds h_sun
        double x = static_cast<double>(col0) + 0.5; // pixel center
        double y = static_cast<double>(row0) + 0.5;
        double tan_h_sun = std::tan(h_sun);

        for (int s = 1; s <= max_steps; ++s)
        {
            x += step_col;
            y += step_row;

            int cx = static_cast<int>(x);
            int cy = static_cast<int>(y);

            if (cx < 0 || cx >= cols || cy < 0 || cy >= rows)
                break;  // Out of bounds; no more obstacles along the ray

            int idx = cy * cols + cx;
            float obs_h = elev[idx];
            if (std::isnan(obs_h))
                continue; // Invalid pixel; treat it as no obstacle

            double distance = s * step_ground_m; // horizontal distance in meters
            if (distance <= 0.0)
                continue;

            double angle_terrain = std::atan2(static_cast<double>(obs_h - base_h), distance);

            // If terrain elevation angle exceeds solar altitude, the pixel is shadowed
            if (angle_terrain > h_sun)
                return false;
        }

        // No terrain exceeded the solar altitude; treat the pixel as visible
        return true;
    }
} // namespace

// =================== Main calculation function: single DEM plus terrain shadowing ===================

void calculateSunshineHoursCpu(
    Raster &sunshine_hours,
    const RasterWithLatitude &dem,
    int day_of_year,
    int time_step_minutes)
{
    const int rows = dem.rows;
    const int cols = dem.cols;

    float *out       = sunshine_hours.data.get();
    const float *elev = dem.data.get();
    const double *gt  = dem.geo_transform;

    const float nanf = std::numeric_limits<float>::quiet_NaN();

    if (time_step_minutes <= 0)
        std::cout<<"time_step_minutes must be a positive integer"<<endl;

    // Hour-angle rate during a day: 15 deg/hour = pi/12 rad/hour = pi/720 rad/minute
    const double d_omega_per_minute = PI / 720.0;

    double delta = calculateSolarDeclination(day_of_year);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            int idx = i * cols + j;
            float h0 = elev[idx];
            if (std::isnan(h0))
            {
                out[idx] = nanf;
                continue;
            }

            // Use GeoTransform to compute pixel-center longitude/latitude
            double x = gt[0] + j * gt[1] + i * gt[2]
                             + 0.5 * gt[1] + 0.5 * gt[2];
            double y = gt[3] + j * gt[4] + i * gt[5]
                             + 0.5 * gt[4] + 0.5 * gt[5];

            double phi = deg2rad(y);  // latitude, degrees to radians

            double omega_r, omega_s;
            if (!computeDaytimeRange(phi, delta, omega_r, omega_s))
            {
                // Polar night: no sunshine during the day
                out[idx] = 0.0f;
                continue;
            }

            // Theoretical daylight hour-angle span at this latitude
            double omega_span = omega_s - omega_r;
            if (omega_span <= 0.0)
            {
                out[idx] = 0.0f;
                continue;
            }

            double d_omega = time_step_minutes * d_omega_per_minute;
            if (d_omega <= 0.0)
            {
                out[idx] = 0.0f;
                continue;
            }

            int n_steps = static_cast<int>(std::floor(omega_span / d_omega)) + 1;
            if (n_steps < 1)
            {
                out[idx] = 0.0f;
                continue;
            }

            double sunshine_minutes = 0.0;

            // Scan from sunrise to sunset using the configured time step
            for (int k = 0; k < n_steps; ++k)
            {
                // Use the midpoint hour angle of each interval for better stability
                double omega = omega_r + (k + 0.5) * d_omega;
                if (omega < omega_r || omega > omega_s)
                    continue;

                double h_sun, A_sun;
                computeSunPosition(phi, delta, omega, h_sun, A_sun);

                if (h_sun <= 0.0)
                    continue;   // Sun is below the horizon

                // Terrain-shadow visibility check
                bool visible = isVisibleSingleDem(i, j, phi, h_sun, A_sun, dem);
                if (visible)
                    sunshine_minutes += static_cast<double>(time_step_minutes);
            }

            out[idx] = static_cast<float>(sunshine_minutes);
        }
    }
}
