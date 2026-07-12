#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "sunshine_hours_cuda.cuh"

#include "timer.h"

#define CHECK_CUDA(call)                                \
    do                                                  \
    {                                                   \
        const cudaError_t error_code = call;            \
        if (error_code != cudaSuccess)                  \
        {                                               \
            printf("CUDA Error:\n");                    \
            printf("    File:       %s\n", __FILE__);   \
            printf("    Line:       %d\n", __LINE__);   \
            printf("    Error code: %d\n", error_code); \
            printf("    Error text: %s\n",              \
                   cudaGetErrorString(error_code));     \
            exit(1);                                    \
        }                                               \
    } while (0)

__host__
double calculateSolarDeclination(const int day_of_year)
{
    double tau = 2 * M_PI * (day_of_year - 1) / 365.2422;
    double delta = 0.006894
                    - 0.399512 * cos(tau)
                    + 0.072075 * sin(tau)
                    - 0.006799 * cos(2 * tau)
                    + 0.000896 * sin(2 * tau)
                    - 0.002689 * cos(3 * tau)
                    + 0.001516 * sin(3 * tau);
    return delta;
}

__device__
void calculateSolarHourAngle(const double phi,
                             const double delta,
                             const int    time_step,
                             double &delta_omega,
                             double &omega_r,
                             double &omega_s,
                             int    &n_steps)
{
    // cos(H0) = -tan(phi) * tan(delta)
    double cosH0 = -tan(phi) * tan(delta);

    // Polar night: no sunshine during the day
    if (cosH0 >= 1.0)
    {
        omega_r = omega_s = 0.0;
        n_steps = 0;
        delta_omega = 0.0;
        return;
    }

    // Polar day: sunshine is possible throughout the day
    if (cosH0 <= -1.0)
    {
        omega_s = M_PI;
        omega_r = -M_PI;
    }
    else
    {
        double H0 = acos(fmin(1.0, fmax(-1.0, cosH0)));
        omega_s = H0;
        omega_r = -H0;
    }

    if (time_step <= 0)
    {
        n_steps = 0;
        delta_omega = 0.0;
        return;
    }

    // Hour-angle rate during a day: 15 deg/h = pi/12 rad/h = pi/720 rad/min
    const double d_omega_per_minute = M_PI / 720.0;

    double omega_span = omega_s - omega_r;
    if (omega_span <= 0.0)
    {
        n_steps = 0;
        delta_omega = 0.0;
        return;
    }

    double d_omega = time_step * d_omega_per_minute;

    int n = (int)floor(omega_span / d_omega) + 1;  // at least one step
    if (n < 1)
    {
        n_steps = 0;
        delta_omega = 0.0;
        return;
    }

    n_steps     = n;
    delta_omega = d_omega;  // hour-angle increment per time step
}


// Longitude/latitude to meters
__device__ inline void metersPerDegree(double phi,
                                       double &m_per_deg_lat,
                                       double &m_per_deg_lon)
{
    double cosphi  = cos(phi);
    double cos2phi = cos(2.0 * phi);
    double cos4phi = cos(4.0 * phi);

    m_per_deg_lat = 111132.954 - 559.822 * cos2phi + 1.175 * cos4phi;
    m_per_deg_lon = 111132.954 * cosphi;
}

// Ray-march in longitude/latitude using real ground step length to test whether the obstacle DEM blocks sunlight
__device__
bool isVisibleBresenhamObstacle(const int    /*base_row*/,  // Kept in the signature but unused
                                const int    /*base_col*/,
                                const double phi,           // latitude in radians
                                const double h_sun,         // solar altitude angle
                                const double A_sun,         // solar azimuth, clockwise from north
                                const float *obstacle_dem,
                                const int    o_rows,
                                const int    o_cols,
                                const double *o_geotransform,
                                const float  base_h,
                                const double base_x,        // target point longitude in degrees
                                const double base_y)        // target point latitude in degrees
{
    if (isnan(base_h)) return false;

    double m_per_deg_lat, m_per_deg_lon;
    metersPerDegree(phi, m_per_deg_lat, m_per_deg_lon);

    // Pixel size in longitude/latitude directions in meters
    double dlon_deg = o_geotransform[1];
    double dlat_deg = o_geotransform[5];
    double cell_dx_m = fabs(dlon_deg) * m_per_deg_lon;
    double cell_dy_m = fabs(dlat_deg) * m_per_deg_lat;

    // Choose a suitable ground step length in meters, for example half a pixel
    double ds_m = 0.5 * fmin(cell_dx_m, cell_dy_m);
    if (ds_m <= 0.0) return true;

    // Unit ray direction on the ground in a metric coordinate system; north = 0, east = 90 deg
    double vx = sin(A_sun);   // eastward
    double vy = -cos(A_sun);  // southward

    double norm = sqrt(vx * vx + vy * vy);
    if (norm <= 0.0) return true;
    vx /= norm;
    vy /= norm;

    // Displacement for one step in metric coordinates
    double step_x_m = vx * ds_m;
    double step_y_m = vy * ds_m;

    // Convert to longitude/latitude displacement in degrees
    double step_lon_deg = step_x_m / m_per_deg_lon;
    double step_lat_deg = step_y_m / m_per_deg_lat;

    const double MAX_TRACE_DISTANCE_M = 50000.0;
    int max_steps = (int)(MAX_TRACE_DISTANCE_M / ds_m);
    if (max_steps <= 0) return true;

    double cur_lon = base_x;
    double cur_lat = base_y;

    for (int s = 1; s <= max_steps; ++s)
    {
        cur_lon += step_lon_deg;
        cur_lat += step_lat_deg;

        // longitude/latitude to row/column
        int col = (int)((cur_lon - o_geotransform[0]) / o_geotransform[1]);
        int row = (int)((cur_lat - o_geotransform[3]) / o_geotransform[5]);

        if (col < 0 || col >= o_cols || row < 0 || row >= o_rows)
            break;

        int idx = row * o_cols + col;
        float obs_h = obstacle_dem[idx];
        if (isnan(obs_h))
            continue;

        double distance = s * ds_m;  // horizontal distance in meters

        double angle_terrain = atan2((double)obs_h - (double)base_h, distance);
        if (angle_terrain > h_sun)
            return false;
    }

    return true;
}

__device__
bool isVisibleSingleDemRay(const double phi,           // latitude in radians
                           const double h_sun,         // solar altitude angle
                           const double A_sun,         // solar azimuth, clockwise from north
                           const float  *dem,
                           const int     rows,
                           const int     cols,
                           const double *gt,           // geotransform of this DEM
                           const float   base_h,
                           const double  base_x,       // target pixel-center longitude in degrees
                           const double  base_y)       // target pixel-center latitude in degrees
{
    if (isnan(base_h)) return false;

    double m_per_deg_lat, m_per_deg_lon;
    metersPerDegree(phi, m_per_deg_lat, m_per_deg_lon);

    // Pixel size in meters
    double dlon_deg   = gt[1];
    double dlat_deg   = gt[5];
    double cell_dx_m  = fabs(dlon_deg) * m_per_deg_lon;
    double cell_dy_m  = fabs(dlat_deg) * m_per_deg_lat;

    // Choose a ground step length: half a pixel
    double ds_m = 0.5 * fmin(cell_dx_m, cell_dy_m);
    if (ds_m <= 0.0) return true;

    // Unit ray direction on the ground in a metric coordinate system; north = 0, east = 90 deg
    double vx = sin(A_sun);   // eastward
    double vy = -cos(A_sun);  // southward because row indices increase downward

    double norm = sqrt(vx * vx + vy * vy);
    if (norm <= 0.0) return true;
    vx /= norm;
    vy /= norm;

    // Metric displacement per step
    double step_x_m = vx * ds_m;
    double step_y_m = vy * ds_m;

    // meters to degrees
    double step_lon_deg = step_x_m / m_per_deg_lon;
    double step_lat_deg = step_y_m / m_per_deg_lat;

    const double MAX_TRACE_DISTANCE_M = 50000.0;
    int max_steps = (int)(MAX_TRACE_DISTANCE_M / ds_m);
    if (max_steps <= 0) return true;

    double cur_lon = base_x;
    double cur_lat = base_y;

    for (int s = 1; s <= max_steps; ++s)
    {
        cur_lon += step_lon_deg;
        cur_lat += step_lat_deg;

        // longitude/latitude to row/column
        int col = (int)((cur_lon - gt[0]) / gt[1]);
        int row = (int)((cur_lat - gt[3]) / gt[5]);

        if (col < 0 || col >= cols || row < 0 || row >= rows)
            break;

        int idx = row * cols + col;
        float obs_h = dem[idx];
        if (isnan(obs_h))
            continue;

        double distance = s * ds_m;  // horizontal distance in meters
        double angle_terrain = atan2((double)obs_h - (double)base_h, distance);

        if (angle_terrain > h_sun)
            return false;
    }

    return true;
}

__device__
void calculateSolarAltitudeAndAzimuth(const double phi, const double delta, const double omega_r, const double delta_omega, const double step_i, double &h_i, double &A_i)
{
    double omega_i = omega_r + delta_omega * step_i;
    h_i = asin(sin(phi) * sin(delta) + cos(phi) * cos(delta) * cos(omega_i));
    A_i = acos((sin(phi) * sin(h_i) - sin(delta)) / (cos(phi) * cos(h_i)));

    if (omega_i < 0)
    {
        A_i = M_PI - A_i;
    }
    else
    {
        A_i = M_PI + A_i;
    }
}

__global__
void calculateSunshineHoursCudaKernel(float *sunshine_hours, const float *target_dem, const double *target_latitude, const int t_rows, const int t_cols, const double *t_geotransform, const float *obstacle_dem, const int o_rows, const int o_cols, const double *o_geotransform, const double solar_declination, const int time_step)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    

    for (int i = idx_y; i < t_rows; i += stride_y)
    {
        for (int j = idx_x; j < t_cols; j += stride_x)
        {
            
            // double x = t_geotransform[0] + j * t_geotransform[1] + i * t_geotransform[2]
            //              + 0.5 * t_geotransform[1] + 0.5 * t_geotransform[2];
            double y = t_geotransform[3] + j * t_geotransform[4] + i * t_geotransform[5]
                                    + 0.5 * t_geotransform[4] + 0.5 * t_geotransform[5];

            // Actual latitude equals y
            double phi = y * M_PI / 180.0;
            double delta_omega, omega_r, omega_s;
            int n_steps;

            calculateSolarHourAngle(phi, solar_declination, time_step, delta_omega, omega_r, omega_s, n_steps);

            if (i == 0 && j == 0 &&
                blockIdx.x == 0 && blockIdx.y == 0 &&
                threadIdx.x == 0 && threadIdx.y == 0)
            {
                printf("[debug] phi=%.3f deg, n_steps=%d, dt=%d\n",
                    phi * 180.0 / M_PI, n_steps, time_step);
            }

            double base_x = t_geotransform[0] + t_geotransform[1] * j;
            double base_y = t_geotransform[3] + t_geotransform[5] * i;
            float base_h = target_dem[i * t_cols + j] + 0.0001;
            if (isnan(base_h))
            {
                sunshine_hours[i * t_cols + j] = NAN;
                continue;
            }

            int base_obs_col = int((base_x - o_geotransform[0]) / o_geotransform[1]);
            int base_obs_row = int((base_y - o_geotransform[3]) / o_geotransform[5]);
            // float base_h = obstacle_dem[base_obs_row * o_cols + base_obs_col] + 0.0001;

            int visible_steps = 0;

            for (int step_i = 0; step_i < n_steps; step_i++)
            {
                double h_i, A_i;
                calculateSolarAltitudeAndAzimuth(phi, solar_declination,
                                                omega_r, delta_omega,
                                                step_i,
                                                h_i, A_i);

                if (h_i <= 0.0)
                    continue;   // Sun is below the horizon; skip

                bool is_visible = isVisibleBresenhamObstacle(
                    base_obs_row, base_obs_col,  // No longer used, but kept in the signature
                    phi,
                    h_i, A_i,
                    obstacle_dem,
                    o_rows, o_cols,
                    o_geotransform,
                    base_h,
                    base_x, base_y               // added pixel-center longitude/latitude in degrees
                );

                if (is_visible)
                    visible_steps++;
            }

            sunshine_hours[i * t_cols + j] = visible_steps * time_step;

            // int visible_steps = 0;

            // for (int step_i = 0; step_i < n_steps; step_i++)
            // {
            //     double h_i, A_i;
            //     calculateSolarAltitudeAndAzimuth(phi, solar_declination, omega_r, delta_omega, step_i, h_i, A_i);

            //     double delta_x_i = sin(A_i);
            //     double delta_y_i = -cos(A_i);
            //     double delta_l_i = o_geotransform[1];
            //     float delta_h_i = tan(h_i) * delta_l_i;

            //     int search_steps = int((8849 - base_h) / delta_h_i);

            //     double current_x = base_obs_col;
            //     double current_y = base_obs_row;
            //     float current_h = base_h;
            //     bool is_visible = true;
            //     for (int search_i = 1; search_i < search_steps; search_i++)
            //     {
            //         current_x += delta_x_i;
            //         current_y += delta_y_i;
            //         current_h += delta_h_i;

            //         if (current_x >= 0 && current_x < o_cols && current_y >= 0 && current_y < o_rows)
            //         {
            //             float obs_h = obstacle_dem[int(current_y) * o_cols + int(current_x)];
                        
            //             if (current_h < obs_h)
            //             {
            //                 is_visible = false;
            //                 break;
            //             }
            //         }
            //         else
            //         {
            //             break;
            //         }
            //     }

            //     if (is_visible)
            //     {
            //         visible_steps++;
            //     }
            // }
            // sunshine_hours[i * t_cols + j] = visible_steps * time_step;
        }
    }
}

void calculateSunshineHoursCuda(Raster & sunshine_hours, const RasterWithLatitude &target_dem, const Raster &obstacle_dem, const int day_of_year, const int time_step)
{
    double solar_declination = calculateSolarDeclination(day_of_year);

    float *d_sunshine_hours;
    float *d_target_dem;
    double *d_target_latitude;
    double *d_t_geotransform;
    float *d_obstacle_dem;
    double *d_o_geotransform;

    // Choose cuda device
    cudaSetDevice(7);

    cudaMalloc((void **)&d_sunshine_hours, sizeof(float) * sunshine_hours.size());
    cudaMalloc((void **)&d_target_dem, sizeof(float) * target_dem.size());
    cudaMalloc((void **)&d_target_latitude, sizeof(double) * target_dem.size());
    cudaMalloc((void **)&d_t_geotransform, sizeof(double) * 6);
    cudaMalloc((void **)&d_obstacle_dem, sizeof(float) * obstacle_dem.size());
    cudaMalloc((void **)&d_o_geotransform, sizeof(double) * 6);

    cudaMemcpy(d_target_dem, target_dem.data.get(), sizeof(float) * target_dem.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_latitude, target_dem.latitude.get(), sizeof(double) * target_dem.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t_geotransform, target_dem.geo_transform, sizeof(double) * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_obstacle_dem, obstacle_dem.data.get(), sizeof(float) * obstacle_dem.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_o_geotransform, obstacle_dem.geo_transform, sizeof(double) * 6, cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size((sunshine_hours.rows + block_size.y - 1) / block_size.y, (sunshine_hours.cols + block_size.x - 1) / block_size.x);

    calculateSunshineHoursCudaKernel<<<grid_size, block_size>>>(d_sunshine_hours, d_target_dem, d_target_latitude, sunshine_hours.rows, sunshine_hours.cols, d_t_geotransform, d_obstacle_dem, obstacle_dem.rows, obstacle_dem.cols, d_o_geotransform, solar_declination, time_step);

    cudaDeviceSynchronize();

    cudaMemcpy(sunshine_hours.data.get(), d_sunshine_hours, sizeof(float) * sunshine_hours.size(), cudaMemcpyDeviceToHost);


    cudaFree(d_sunshine_hours);
    cudaFree(d_target_dem);
    cudaFree(d_target_latitude);
    cudaFree(d_t_geotransform);
    cudaFree(d_obstacle_dem);
    cudaFree(d_o_geotransform);
}

__global__
void calculateSunshineHoursCudaKernel(float *sunshine_hours, const float *dem, const double *latitude, const int rows, const int cols, const double *geotransform, const double solar_declination, const int time_step)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for (int i = idx_y; i < rows; i += stride_y)
    {
        for (int j = idx_x; j < cols; j += stride_x)
        {
            // pixel-center longitude/latitude
            double base_x = geotransform[0] + j * geotransform[1] + i * geotransform[2]
                                        + 0.5 * geotransform[1] + 0.5 * geotransform[2];
            double base_y = geotransform[3] + j * geotransform[4] + i * geotransform[5]
                                        + 0.5 * geotransform[4] + 0.5 * geotransform[5];
            double phi = base_y * M_PI / 180.0;
            double delta_omega, omega_r, omega_s;
            int n_steps;

            calculateSolarHourAngle(phi, solar_declination, time_step, delta_omega, omega_r, omega_s, n_steps);


            float base_h = dem[i * cols + j] + 0.0001;
            if (isnan(base_h))
            {
                sunshine_hours[i * cols + j] = NAN;
                continue;
            }

            int visible_steps = 0;

            for (int step_i = 0; step_i < n_steps; step_i++)
            {
                double h_i, A_i;
                calculateSolarAltitudeAndAzimuth(phi, solar_declination,
                                                omega_r, delta_omega,
                                                step_i, h_i, A_i);

                if (h_i <= 0.0)
                    continue;   // Sun is below the horizon

                bool is_visible = isVisibleSingleDemRay(
                    phi,
                    h_i, A_i,
                    dem,
                    rows, cols,
                    geotransform,
                    base_h,
                    base_x, base_y   // pixel-center longitude/latitude in degrees
                );

                if (is_visible)
                {
                    ++visible_steps;
                }
            }

            sunshine_hours[i * cols + j] = visible_steps * time_step;
        }
    }
}

void calculateSunshineHoursCuda(Raster &sunshine_hours, const RasterWithLatitude &dem, const int day_of_year, const int time_step)
{
    double solar_declination = calculateSolarDeclination(day_of_year);

    float *d_sunshine_hours;
    float *d_dem;
    double *d_latitude;
    double *d_geotransform;

    // Choose cuda device
    cudaSetDevice(7);

    CHECK_CUDA(cudaMalloc((void **)&d_sunshine_hours, sizeof(float) * sunshine_hours.size()));
    CHECK_CUDA(cudaMalloc((void **)&d_dem, sizeof(float) * dem.size()));
    CHECK_CUDA(cudaMalloc((void **)&d_latitude, sizeof(double) * dem.size()));
    CHECK_CUDA(cudaMalloc((void **)&d_geotransform, sizeof(double) * 6));

    timer.tick("cuda copy to device");
    CHECK_CUDA(cudaMemcpy(d_dem, dem.data.get(), sizeof(float) * dem.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_latitude, dem.latitude.get(), sizeof(double) * dem.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_geotransform, dem.geo_transform, sizeof(double) * 6, cudaMemcpyHostToDevice));
    timer.tock();

    timer.tick("cuda KERNEL");
    dim3 block_size(16, 16);
    dim3 grid_size((sunshine_hours.rows + block_size.y - 1) / block_size.y, (sunshine_hours.cols + block_size.x - 1) / block_size.x);

    calculateSunshineHoursCudaKernel<<<grid_size, block_size>>>(d_sunshine_hours, d_dem, d_latitude, sunshine_hours.rows, sunshine_hours.cols, d_geotransform, solar_declination, time_step);
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());

    timer.tock();

    timer.tick("cuda copy back");
    CHECK_CUDA(cudaMemcpy(sunshine_hours.data.get(), d_sunshine_hours, sizeof(float) * sunshine_hours.size(), cudaMemcpyDeviceToHost));
    timer.tock();

    CHECK_CUDA(cudaFree(d_sunshine_hours));
    CHECK_CUDA(cudaFree(d_dem));
    CHECK_CUDA(cudaFree(d_latitude));
    CHECK_CUDA(cudaFree(d_geotransform));
}
