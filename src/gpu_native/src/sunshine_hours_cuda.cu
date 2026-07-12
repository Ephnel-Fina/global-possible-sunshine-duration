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
    m_per_deg_lat = 111132.92 - 559.82 * cos(2.0 * phi) + 1.175 * cos(4.0 * phi);
    m_per_deg_lon = 111412.84 * cos(phi) - 93.5 * cos(3.0 * phi);
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

    // 1) Metric coefficients for latitude
    double m_per_deg_lat, m_per_deg_lon;
    metersPerDegree(phi, m_per_deg_lat, m_per_deg_lon);

    const double meters_per_pixel_x = m_per_deg_lon * o_geotransform[1];
    const double meters_per_pixel_y = -m_per_deg_lat * o_geotransform[5];

    const double pixel_x_rad = o_geotransform[1] * M_PI / 180.0;
    const double pixel_y_rad = o_geotransform[5] * M_PI / 180.0;

    // 2) Ray direction in row/column space
    const double delta_x = sin(A_sun);
    const double delta_y = -cos(A_sun);

    const double step_dist = hypot(meters_per_pixel_x * delta_x,
                                   meters_per_pixel_y * delta_y);
    if (step_dist <= 0.0) return true;

    const double MAX_TRACE_DISTANCE_M = 50000.0;
    int max_search_steps = (int)(MAX_TRACE_DISTANCE_M / step_dist);
    if (max_search_steps <= 0) return true;

    int dx_pix = (int)(delta_x * max_search_steps);
    int dy_pix = (int)(delta_y * max_search_steps);
    int dx = abs(dx_pix), dy = abs(dy_pix);
    int sx = (dx_pix >= 0) ? 1 : -1;
    int sy = (dy_pix >= 0) ? 1 : -1;
    int err = dx - dy;
    int x = 0, y = 0;

    const int base_col = (int)((base_x - o_geotransform[0]) / o_geotransform[1]);
    const int base_row = (int)((base_y - o_geotransform[3]) / o_geotransform[5]);

    const double tan_h   = tan(h_sun);
    const double R_earth = 6371000.0;

    for (int k = 0; k < max_search_steps; ++k)
    {
        const int e2 = err << 1;
        if (e2 > -dy) { err -= dy; x += sx; }
        if (e2 <  dx) { err += dx; y += sy; }

        const int col = base_col + x;
        const int row = base_row + y;

        if (col < 0 || col >= o_cols || row < 0 || row >= o_rows)
            break;

        const float obs_h = obstacle_dem[row * o_cols + col];
        if (isnan(obs_h))
            continue;

        const double delta_phi    = y * pixel_y_rad;
        const double delta_lambda = x * pixel_x_rad;
        const double sp2 = sin(0.5 * delta_phi);
        const double sl2 = sin(0.5 * delta_lambda);
        const double a   = sp2 * sp2
                         + cos(phi) * cos(phi + delta_phi) * sl2 * sl2;
        const double c   = 2.0 * atan2(sqrt(a),
                                       sqrt(fmax(1.0 - a, 0.0)));

        const double curv_term = R_earth * (1.0 - cos(c));

        const double ground_dist =
              (x * meters_per_pixel_x * delta_x
             + y * meters_per_pixel_y * delta_y);

        const double height_limit = ground_dist * tan_h + curv_term;
        const double height_diff  = (double)obs_h - (double)base_h;

        if (height_diff > height_limit)
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

    // 1) Metric scale at the row latitude.
    double m_per_deg_lat, m_per_deg_lon;
    metersPerDegree(phi, m_per_deg_lat, m_per_deg_lon);

    const double meters_per_pixel_x = m_per_deg_lon * gt[1];
    const double meters_per_pixel_y = -m_per_deg_lat * gt[5];   // gt[5] < 0 for north-up rasters

    const double pixel_x_rad = gt[1] * M_PI / 180.0;
    const double pixel_y_rad = gt[5] * M_PI / 180.0;

    // 2) Unit ray direction in row/column space; east is +x and south is +y.
    const double delta_x = sin(A_sun);
    const double delta_y = -cos(A_sun);

    // Projected ground distance per step in meters
    const double step_dist = hypot(meters_per_pixel_x * delta_x,
                                   meters_per_pixel_y * delta_y);
    if (step_dist <= 0.0) return true;

    // 3) Convert the 50 km maximum tracing distance to a Bresenham step count.
    const double MAX_TRACE_DISTANCE_M = 50000.0;
    int max_search_steps = (int)(MAX_TRACE_DISTANCE_M / step_dist);
    if (max_search_steps <= 0) return true;

    // 4) Bresenham endpoint in pixel offsets
    int dx_pix = (int)(delta_x * max_search_steps);
    int dy_pix = (int)(delta_y * max_search_steps);
    int dx = abs(dx_pix), dy = abs(dy_pix);
    int sx = (dx_pix >= 0) ? 1 : -1;
    int sy = (dy_pix >= 0) ? 1 : -1;
    int err = dx - dy;
    int x = 0, y = 0;

    // Base pixel row/column, kept consistent with the DDA longitude/latitude-to-row/column mapping
    const int base_col = (int)((base_x - gt[0]) / gt[1]);
    const int base_row = (int)((base_y - gt[3]) / gt[5]);

    const double tan_h   = tan(h_sun);
    const double R_earth = 6371000.0;

    for (int k = 0; k < max_search_steps; ++k)
    {
        const int e2 = err << 1;
        if (e2 > -dy) { err -= dy; x += sx; }
        if (e2 <  dx) { err += dx; y += sy; }

        const int col = base_col + x;
        const int row = base_row + y;

        if (col < 0 || col >= cols || row < 0 || row >= rows)
            break;

        const float obs_h = dem[row * cols + col];
        if (isnan(obs_h))
            continue;

        // 5) Along-ray distance plus Earth-curvature term, consistent with the optimized version, using haversine
        const double delta_phi    = y * pixel_y_rad;
        const double delta_lambda = x * pixel_x_rad;
        const double sp2 = sin(0.5 * delta_phi);
        const double sl2 = sin(0.5 * delta_lambda);
        const double a   = sp2 * sp2
                         + cos(phi) * cos(phi + delta_phi) * sl2 * sl2;
        const double c   = 2.0 * atan2(sqrt(a),
                                       sqrt(fmax(1.0 - a, 0.0)));

        const double curv_term = R_earth * (1.0 - cos(c));

        // Projected distance along the ray in meters: project (x,y) onto the ray direction
        const double ground_dist =
              (x * meters_per_pixel_x * delta_x
             + y * meters_per_pixel_y * delta_y);

        // Line-of-sight height model: theoretical solar-ray height
        const double height_limit = ground_dist * tan_h + curv_term;
        const double height_diff  = (double)obs_h - (double)base_h;

        // If terrain elevation exceeds the theoretical ray height, the target is shadowed
        if (height_diff > height_limit)
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
