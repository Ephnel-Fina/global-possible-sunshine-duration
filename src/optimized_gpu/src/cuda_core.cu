#ifdef __INTELLISENSE__
#define __CUDACC__
#endif
#include <cuda_runtime.h>

#include <iostream>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <sstream>
#include "cuda_core.cuh"
#include <iostream>
#include "timer.h"
#include <math_constants.h>
#include <algorithm>
#include "config.h"

//

//Compile-time optimization switches: 0 disables, 1 enables
#ifndef OPT_HALF_DAY_SYMM //Optimization: half-day symmetry switch 
#define OPT_HALF_DAY_SYMM 1
#endif

#ifndef OPT_DEGREE_RECURRENCE//Optimization: trigonometric recurrence
#define OPT_DEGREE_RECURRENCE 1
#endif

#ifndef OPT_TWO_STAGE_VIS  // Optimization (8): two-stage visibility early-exit strategy
#define OPT_TWO_STAGE_VIS 0
#endif

#ifndef OPT_CURV_RADIUS_BISECT  // Adaptive search radius: curvature-consistent bisection upper bound; defaults to optimization (8)
#define OPT_CURV_RADIUS_BISECT OPT_TWO_STAGE_VIS
#endif

#ifndef OPT_DYNAMIC_RADIUS//Optimization (6): dynamic search radius
#define OPT_DYNAMIC_RADIUS 1
#endif

#ifndef OPT_EARTH_CURVATURE // Earth-curvature correction switch: 1 enables, 0 disables
#define OPT_EARTH_CURVATURE 1
#endif

//Accuracy optimization
#ifndef OPT_ROBUST_SUNRISE//Optimization: robust sunrise/sunset solving
#define OPT_ROBUST_SUNRISE 1
#endif


#define CHECK_CUDA(call)                                                      \
    do {                                                                     \
        const cudaError_t error_code = (call);                                \
        if (error_code != cudaSuccess) {                                      \
            std::ostringstream _oss;                                          \
            _oss << "CUDA Error:\n"                                          \
                 << "    File:       " << __FILE__ << "\n"               \
                 << "    Line:       " << __LINE__ << "\n"               \
                 << "    Error code: " << error_code << "\n"            \
                 << "    Error text: " << cudaGetErrorString(error_code);    \
            throw std::runtime_error(_oss.str());                             \
        }                                                                     \
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

__host__ inline
void calculateSolarHourAngle(double phi, double delta, int time_step,
                             double &domega, double &omega_r, double &omega_s, int &n_steps)
{
    double cosw0 = -std::tan(phi)*std::tan(delta);
    cosw0 = std::max(-1.0, std::min(1.0, cosw0));
    if (cosw0 <= -1.0) { omega_r = -M_PI; omega_s =  M_PI; }
    else if (cosw0 >=  1.0) { omega_r = 0.0; omega_s = 0.0; n_steps = 0; domega=0; return; }
    else { double w0 = std::acos(cosw0); omega_r = -w0; omega_s = w0; }

    const int ts = std::max(1, time_step);
    const double hours = (omega_s - omega_r) * (12.0/M_PI);
    int req = (int)std::ceil(hours*60.0/ts);
    if (req < 1) req = 1;
    int cap = (24*60)/ts;
    n_steps = std::min(req, cap);
    domega = (n_steps>1)? (omega_s-omega_r)/(n_steps-1) : 0.0;
}


__host__ __device__ inline
void calcSolarHourAngle_HD(double phi, double delta, int time_step_min,
                           int max_n_steps_cap,
                           double& domega, double& omega_r, double& omega_s, int& n_steps)
{
    const int ts = (time_step_min > 0) ? time_step_min : 1;
    const int hard_cap = (24 * 60) / ts; 
    const int cap =
        (max_n_steps_cap > 0) ? min(max_n_steps_cap, hard_cap) : hard_cap;

#if OPT_ROBUST_SUNRISE
    double cosw0 = -tan(phi) * tan(delta);
    if (cosw0 < -1.0) cosw0 = -1.0;
    if (cosw0 >  1.0) cosw0 =  1.0;

    if (cosw0 <= -1.0) {
    #ifdef CUDART_PI
        omega_r = -CUDART_PI;
        omega_s =  CUDART_PI;
    #else
        omega_r = -M_PI;
        omega_s =  M_PI;
    #endif
    } else if (cosw0 >= 1.0) {
        omega_r = 0.0;
        omega_s = 0.0;
        n_steps = 0;
        domega  = 0.0;
        return;
    } else {
        const double w0 = acos(cosw0);
        omega_r = -w0;
        omega_s =  w0;
    }

    const double span_hours = (omega_s - omega_r) * (12.0 / M_PI);
    int req = (int)ceil((span_hours * 60.0) / (double)ts);
    if (req < 1) req = 1;

    n_steps = min(req, cap);

#else

#ifdef CUDART_PI
    omega_r = -CUDART_PI;
    omega_s =  CUDART_PI;
#else
    omega_r = -M_PI;
    omega_s =  M_PI;
#endif

    n_steps = cap;
#endif  

    domega = (n_steps > 1)
           ? (omega_s - omega_r) / (double)(n_steps - 1)
           : 0.0;
}


__global__
void calculateSolarAltitudeAndAzimuthBatchKernel(
    const int    batch_size,
    const int    max_n_steps,        
    const double delta,               
    const int    time_step,           
    const size_t row_base,            
    const double* __restrict__ geo_transform,  
    double* __restrict__ h_array,     
    double* __restrict__ A_array,     
    int*    __restrict__ n_steps_arr)  
{
    const int row  = blockIdx.y * blockDim.y + threadIdx.y; 
    const int step = blockIdx.x * blockDim.x + threadIdx.x; 
    if (row >= batch_size || step >= max_n_steps) return;

    const double lat_deg =
        geo_transform[3] + (double)(row_base + (size_t)row) * geo_transform[5];
    const double phi = lat_deg * M_PI / 180.0;

    double omega_r, omega_s, delta_omega;
    int n_steps = 0;
    calcSolarHourAngle_HD(phi, delta, time_step, max_n_steps,
                          delta_omega, omega_r, omega_s, n_steps);

    double sphi, cphi;  sincos(phi,   &sphi, &cphi);
    double sdel, cdel;  sincos(delta, &sdel, &cdel);

    if (n_steps < 1) n_steps = 1;
    if (step == 0) n_steps_arr[row] = n_steps;

#if OPT_HALF_DAY_SYMM
    const int effective_n = (n_steps + 1) / 2;   
#else
    const int effective_n = n_steps;             
#endif

    const size_t stride_x = gridDim.x * blockDim.x;

#if OPT_DEGREE_RECURRENCE
    const double omega0 = omega_r + delta_omega * double(step);
    double so, co;
    sincos(omega0, &so, &co);

    const double delta_omega_stride = delta_omega * double(stride_x);
    double sdb, cdb;
    sincos(delta_omega_stride, &sdb, &cdb);
#endif

    for (size_t j = step, t = 0; j < (size_t)effective_n; j += stride_x, ++t) {

        const double omega_j = omega_r + delta_omega * double(j);

        double co_local;

#if OPT_DEGREE_RECURRENCE
        co_local = co;
#else
        co_local = cos(omega_j);
#endif

        const double sh  = sphi * sdel + cphi * cdel * co_local;
        const double shc = fmin(1.0, fmax(-1.0, sh));
        const double h_i = asin(shc);
        const double ch  = sqrt(fmax(0.0, 1.0 - shc * shc));

        double numer = sphi * shc - sdel;
        double denom = cphi * ch;
        denom = (fabs(denom) < 1e-15) ? copysign(1e-15, denom) : denom;
        double u = numer / denom;
        u = fmin(1.0, fmax(-1.0, u));

        double A_i = acos(u);
        A_i = (omega_j < 0.0) ? (CUDART_PI - A_i) : (CUDART_PI + A_i);

#if OPT_HALF_DAY_SYMM
        const size_t idx_am = (size_t)row * (size_t)max_n_steps + j;
        const size_t idx_pm =
            (size_t)row * (size_t)max_n_steps + (size_t)(n_steps - 1 - (int)j);

        h_array[idx_am] = h_i;
        h_array[idx_pm] = h_i;

        const double A_sym = 2.0 * CUDART_PI - A_i;
        A_array[idx_am] = A_i;
        A_array[idx_pm] = A_sym;
#else
        const size_t idx = (size_t)row * (size_t)max_n_steps + j;
        h_array[idx] = h_i;
        A_array[idx] = A_i;
#endif

#if OPT_DEGREE_RECURRENCE
        double co_next = fma(co, cdb, -so * sdb); 
        double so_next = fma(so, cdb,  co * sdb);
        double s       = fmaf(co_next, co_next, so_next * so_next);
        double inv     = rsqrtf(s);
        co = co_next * inv;
        so = so_next * inv;
#endif
    }
}




__global__
void calculateRayAndStepHeightChangesBatchKernel(
    const int    batch_size,
    const int    max_n_steps,
    const int    max_search_steps,
    const size_t row_base,
    const double* __restrict__ geo_transform,
    const float  max_relative_height,
    const double* __restrict__ h_array,
    const double* __restrict__ A_array,
    const int*   __restrict__ n_steps_arr,

    int*   __restrict__ search_steps_flat, 
    int*   __restrict__ dxs_flat,          
    int*   __restrict__ dys_flat,
    float* __restrict__ height_changes_flat)
{
    const int row  = blockIdx.y * blockDim.y + threadIdx.y;
    const int step = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch_size) return;

    const int n_steps = n_steps_arr[row];
    if (n_steps <= 0) return;               
#if OPT_HALF_DAY_SYMM
    const int effective_n = (n_steps + 1) / 2;
#else
    const int effective_n = n_steps;
#endif
    if (step >= effective_n) return;          

    const double lat_deg = geo_transform[3] + (double)(row_base + row) * geo_transform[5];
    const double phi     = lat_deg * M_PI / 180.0;

    const double meters_per_deg_lat = 111132.92 - 559.82 * cos(2 * phi) + 1.175 * cos(4 * phi);
    const double meters_per_deg_lon = 111412.84 * cos(phi) - 93.5  * cos(3 * phi);
    const double meters_per_pixel_x = meters_per_deg_lon * geo_transform[1];
    const double meters_per_pixel_y = -meters_per_deg_lat * geo_transform[5];   
    const double pixel_x_rad = geo_transform[1] * M_PI / 180.0;
    const double pixel_y_rad = geo_transform[5] * M_PI / 180.0;

    const size_t base_idx = (size_t)row * (size_t)max_n_steps + (size_t)step;
    const double h_i = h_array[base_idx];
    const double A_i = A_array[base_idx];


    const double delta_x_i = sin(A_i);
    const double delta_y_i = -cos(A_i);
    const double step_dist = hypot(meters_per_pixel_x * delta_x_i,
                                   meters_per_pixel_y * delta_y_i);

    const double tan_h_i   = tan(h_i);
    const double delta_h_i = tan_h_i * step_dist;

    int search_step;

#if OPT_DYNAMIC_RADIUS

  #if OPT_CURV_RADIUS_BISECT
    // ===== Curvature-consistent adaptive upper bound: solve L* by bisection, then map to search_step =====
    const double R = 6371000.0;
    const double H = (max_relative_height > 0.0f) ? (double)max_relative_height : 0.0;

    if (!(step_dist > 0.0) || !isfinite(step_dist) || !isfinite(tan_h_i) || H <= 0.0) {
        search_step = (H <= 0.0) ? 0 : max_search_steps;
    } else {
        const double s_hi = (double)max_search_steps * step_dist;

        // f(s) = tan(h)*s + curvature(s) - H
        #if OPT_EARTH_CURVATURE
        const double curv_hi = R * (1.0 - cos(s_hi / R));
        #else
        const double curv_hi = 0.0;
        #endif
        double f_hi = tan_h_i * s_hi + curv_hi - H;

        if (f_hi < 0.0) {
            // Even the upper bound does not reach H; keep the upper bound
            search_step = max_search_steps;
        } else {
            double lo = 0.0, hi = s_hi;

            #pragma unroll
            for (int it = 0; it < 9; ++it) {
                double mid = 0.5 * (lo + hi);
                #if OPT_EARTH_CURVATURE
                const double curv_mid = R * (1.0 - cos(mid / R));
                #else
                const double curv_mid = 0.0;
                #endif
                double f_mid = tan_h_i * mid + curv_mid - H;
                if (f_mid >= 0.0) hi = mid;
                else              lo = mid;
            }

            int cand = (int)ceil(hi / step_dist) + 1; // +1 margin to avoid a too-short radius from numerical error
            if (cand < 0) cand = 0;
            if (cand > max_search_steps) cand = max_search_steps;
            search_step = cand;
        }
    }

  #else
    // Dynamic radius using the linear uplift term.
    const double eps = 1e-6;
    if (fabs(delta_h_i) < eps) {
        search_step = max_search_steps;
    } else {
        const double q = max_relative_height / delta_h_i;
        int cand = (int)floor(q);
        if (cand < 0) cand = 0;
        if (cand > max_search_steps) cand = max_search_steps;
        search_step = cand;
    }
  #endif

#else
  search_step = max_search_steps;
#endif


    search_steps_flat[base_idx] = search_step;


    int dx_pix = int(delta_x_i * search_step);
    int dy_pix = int(delta_y_i * search_step);
    int dx = abs(dx_pix), dy = abs(dy_pix);
    int sx = (dx_pix > 0) ? 1 : -1;
    int sy = (dy_pix > 0) ? 1 : -1;
    int err = dx - dy;
    int x = 0, y = 0;

    int*   dxs            = dxs_flat + base_idx * (size_t)max_search_steps;
    int*   dys            = dys_flat + base_idx * (size_t)max_search_steps;
    float* height_changes = height_changes_flat + base_idx * (size_t)max_search_steps;

    for (int k = 0; k < search_step; ++k) {
        const int e2 = err << 1;
        if (e2 > -dy) { err -= dy; x += sx; }
        if (e2 <  dx) { err += dx; y += sy; }

        dxs[k] = x;
        dys[k] = y;

        const double delta_phi    = y * pixel_y_rad;
        const double delta_lambda = x * pixel_x_rad;
        const double sp2 = sin(0.5 * delta_phi);
        const double sl2 = sin(0.5 * delta_lambda);
        const double a   = sp2*sp2 + cos(phi) * cos(phi + delta_phi) * sl2*sl2;
        const double c   = 2.0 * atan2(sqrt(a), sqrt(max(1.0 - a, 0.0)));

        #if OPT_EARTH_CURVATURE
        const double curv_h = 6371000.0 * (1.0 - cos(c));
        #else
        const double curv_h = 0.0;
        #endif
        height_changes[k] = float(
              (x * meters_per_pixel_x * delta_x_i
             + y * meters_per_pixel_y * delta_y_i) * tan_h_i
            + curv_h
        );
    }

#if OPT_HALF_DAY_SYMM
    const size_t sym_base_idx =
        (size_t)row * (size_t)max_n_steps + (size_t)(n_steps - 1 - step);
    search_steps_flat[sym_base_idx] = search_step;

    int*   dxs_sym  = dxs_flat + sym_base_idx * (size_t)max_search_steps;
    int*   dys_sym  = dys_flat + sym_base_idx * (size_t)max_search_steps;
    float* hchg_sym = height_changes_flat + sym_base_idx * (size_t)max_search_steps;

    for (int k = 0; k < search_step; ++k) {
        dxs_sym[k]  = -dxs[k];
        dys_sym[k]  =  dys[k];
        hchg_sym[k] =  height_changes[k];
    }
#endif
}

__global__ void calculateSunshineHoursBatchKernel_Edge(
    const int    batch_size,
    const size_t row_base,
    const size_t col0_in_dem,
    const size_t cols_per_dem_row,  
    const size_t cols_per_result_row,
    const int    time_step,
    const int    max_n_steps,
    const int*   __restrict__ n_steps_arr,
    const int*   __restrict__ dxs,
    const int*   __restrict__ dys,
    const float* __restrict__ height_changes,
    const int*   __restrict__ search_steps,
    const int    max_search_steps,
    const int    dem_rows,           
    cudaTextureObject_t tex_dem_obj,
    float*       d_result_batch)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= (size_t)batch_size || col >= cols_per_result_row) return;

    size_t row_in_dem = row_base + row;

    float local_h = tex2D<float>(
        tex_dem_obj,
        (float)(col0_in_dem + col) + 0.5f,
        (float)row_in_dem + 0.5f
    );
    if (__isnanf(local_h)) {
    d_result_batch[row * cols_per_result_row + col] = CUDART_NAN_F;
    return;
    }

    int n_steps = n_steps_arr[row];
    if (n_steps <= 0) { 
        d_result_batch[row * cols_per_result_row + col] = 0.0f;
        return;
    }

    float result_time = 0.0f;

    for (int j = 0; j < n_steps; ++j) {
        int s0 = row * max_n_steps + j;
        int search_step = search_steps[s0];
        int base_x0 = (int)(col0_in_dem + col);
        int base_y0 = (int)row_in_dem;
        int off = s0 * max_search_steps;

        bool is_visible = true;

    #if OPT_TWO_STAGE_VIS
        is_visible = false;
        float max_obs_h = -1e20f;
        for (int k = 0; k < search_step; ++k) {
            int dx = dxs[off + k];
            int dy = dys[off + k];
            int xi = base_x0 + dx;
            int yi = base_y0 + dy;
            if ((unsigned)xi >= (unsigned)cols_per_dem_row || (unsigned)yi >= (unsigned)dem_rows) break;
            float oh = tex2D<float>(tex_dem_obj, (float)xi + 0.5f, (float)yi + 0.5f);
            if (oh > max_obs_h) max_obs_h = oh;
        }
        for (int k = 0; k < search_step; ++k) {
            float rh = local_h + height_changes[off + k];
            if (rh > max_obs_h) { is_visible = true; break; }
            int xi = base_x0 + dxs[off + k];
            int yi = base_y0 + dys[off + k];
            if ((unsigned)xi >= (unsigned)cols_per_dem_row || (unsigned)yi >= (unsigned)dem_rows) break;
            float oh = tex2D<float>(tex_dem_obj, (float)xi + 0.5f, (float)yi + 0.5f);
            if (__isnanf(oh)) break;
            if (oh > rh) { is_visible = false; break; }
        }
    #else
        // One-way point-by-point shadow comparison
        is_visible = true;
        for (int k = 0; k < search_step; ++k) {
            float rh = local_h + height_changes[off + k];
            int xi = base_x0 + dxs[off + k];
            int yi = base_y0 + dys[off + k];
            if ((unsigned)xi >= (unsigned)cols_per_dem_row || (unsigned)yi >= (unsigned)dem_rows) break;
            float oh = tex2D<float>(tex_dem_obj, (float)xi + 0.5f, (float)yi + 0.5f);
            if (__isnanf(oh)) break;
            if (oh > rh) { is_visible = false; break; }
        }
    #endif

    if (is_visible) result_time += (float)time_step;
    }


    d_result_batch[row * cols_per_result_row + col] = result_time;
}

CalculateSunshineHoursCuda::CalculateSunshineHoursCuda(Raster &result, const Raster &dem, const IndexRange &target_index_range, const int day_of_year, const int time_step, const int cuda_device_id)
    : result(result), dem(dem), target_index_range(target_index_range), day_of_year(day_of_year), time_step(time_step), cuda_device_id(cuda_device_id)
{
    // set cuda device
    CHECK_CUDA(cudaSetDevice(cuda_device_id));

    // debug output
    std::cout << "cuda_device_id: " << cuda_device_id << std::endl;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, cuda_device_id));
    std::cout << "asyncEngineCount: " << prop.asyncEngineCount << std::endl;

    // create texture object for the dem
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>(); // create channel description for the texture
    CHECK_CUDA(cudaMallocArray(&d_dem_array, &desc, dem.cols, dem.rows)); // allocate device memory for the texture
    CHECK_CUDA(cudaMemcpy2DToArray(d_dem_array, 0, 0, dem.data.get(), dem.cols * sizeof(float), dem.cols * sizeof(float), dem.rows, cudaMemcpyHostToDevice)); // copy data to the texture

    cudaResourceDesc resDesc = {}; // create resource description for the texture
    resDesc.resType = cudaResourceTypeArray; // set resource type to array
    resDesc.res.array.array = d_dem_array; // set array to the texture

    cudaTextureDesc texDesc = {}; // create texture description for the texture
    texDesc.addressMode[0] = cudaAddressModeClamp; // when the texture is out of U bounds, clamp the value
    texDesc.addressMode[1] = cudaAddressModeClamp; // when the texture is out of V bounds, clamp the value
    texDesc.filterMode     = cudaFilterModePoint; // use point sampling
    texDesc.readMode       = cudaReadModeElementType; // read the texture as element type
    texDesc.normalizedCoords = 0; // use actual coordinates

    CHECK_CUDA(cudaCreateTextureObject(&tex_dem_obj, &resDesc, &texDesc, nullptr)); // create texture object

    // // debug output
    // std::cout << "tex_dem_obj created" << std::endl;

    // allocate device memory for the geo_transform
    CHECK_CUDA(cudaMalloc(&d_geo_transform, 6 * sizeof(double)));

    // copy geo_transform to device
    CHECK_CUDA(cudaMemcpy(d_geo_transform, dem.geo_transform, 6 * sizeof(double), cudaMemcpyHostToDevice));

    // // debug output
    // std::cout << "d_geo_transform created" << std::endl;

    // calculate grid size
    grid_size = dim3((result.cols + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x,
                     (result.rows + BLOCK_SIZE.y - 1) / BLOCK_SIZE.y);
}

CalculateSunshineHoursCuda::~CalculateSunshineHoursCuda() noexcept(false)
{
    CHECK_CUDA(cudaDestroyTextureObject(tex_dem_obj)); // unbind the texture
    CHECK_CUDA(cudaFreeArray(d_dem_array));
    CHECK_CUDA(cudaFree(d_geo_transform));
}

void CalculateSunshineHoursCuda::calculate()
{
    const int batch = BATCH;              // Rows processed in one batch
    // debug output
    std::cout << "calculate started" << std::endl;

    std::vector<cudaStream_t> streams(NUM_STREAMS);
    std::vector<float*> d_result_row_buffers(NUM_STREAMS);
    std::vector<float*> h_result_row_buffers(NUM_STREAMS);

    // struct for async copy callback
    struct CopyCtx {
        float* dst;
        float* src;
        size_t bytes;
    };

    for (int s = 0; s < NUM_STREAMS; ++s)
    {
        CHECK_CUDA(cudaStreamCreate(&streams[s]));
        CHECK_CUDA(cudaMalloc(&d_result_row_buffers[s], batch * result.cols * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&h_result_row_buffers[s], batch * result.cols * sizeof(float)));
    }

    // calculate solar declination
    double delta = calculateSolarDeclination(day_of_year);

    // debug output
    std::cout << "delta: " << delta << std::endl;

    int computed_steps = int(
    std::sqrt(
        double(target_index_range.row_from) * target_index_range.row_from
      + double(target_index_range.col_from) * target_index_range.col_from
    ) + 1
    );

    int max_search_steps;
    if (PADDING_DEGREE == 0.0f) {
        max_search_steps = 3600;
    } else {
        max_search_steps = std::min(computed_steps, 3600);
    }
    int max_n_steps = 24 * 60 / time_step;


    // debug output
    std::cout << "dem.cols: " << dem.cols << std::endl;
    std::cout << "dem.rows: " << dem.rows << std::endl;
    std::cout << "max_search_steps: " << max_search_steps << std::endl;

    std::vector<int*> h_dxs(NUM_STREAMS);
    std::vector<int*> h_dys(NUM_STREAMS);
    std::vector<float*> h_height_changes(NUM_STREAMS);
    std::vector<int*> h_search_steps(NUM_STREAMS);
    std::vector<int*> d_dxs(NUM_STREAMS);
    std::vector<int*> d_dys(NUM_STREAMS);
    std::vector<float*> d_height_changes(NUM_STREAMS);
    std::vector<int*> d_search_steps(NUM_STREAMS);
    std::vector<int*> d_n_steps(NUM_STREAMS);     
    std::vector<double*> d_h_array(NUM_STREAMS);
    std::vector<double*> d_A_array(NUM_STREAMS);

    for (int s = 0; s < NUM_STREAMS; ++s)
    {
        CHECK_CUDA(cudaMallocHost(&h_dxs[s],     batch * max_search_steps * max_n_steps * sizeof(int)));
        CHECK_CUDA(cudaMallocHost(&h_dys[s],     batch * max_search_steps * max_n_steps * sizeof(int)));
        CHECK_CUDA(cudaMallocHost(&h_height_changes[s], batch * max_search_steps * max_n_steps * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&h_search_steps[s],   batch * max_n_steps * sizeof(int)));

        CHECK_CUDA(cudaMalloc(&d_dxs[s],         batch * max_search_steps * max_n_steps * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_dys[s],         batch * max_search_steps * max_n_steps * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_height_changes[s], batch * max_search_steps * max_n_steps * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_search_steps[s],   batch * max_n_steps * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_n_steps[s],        batch * sizeof(int)));

        CHECK_CUDA(cudaMalloc(&d_h_array[s], batch * max_n_steps * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_A_array[s], batch * max_n_steps * sizeof(double)));
    }

    // =================================================================================
    // CALCULATE SUNSHINE HOURS BY BATCH
    size_t total_rows = result.rows;
    for (size_t row_base = 0; row_base < total_rows; row_base += batch)
    {
        size_t cur_batch = std::min<size_t>(batch, total_rows - row_base);
        size_t s = (row_base / batch) % NUM_STREAMS;

        dim3 grid_size_batch((result.cols + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x,
                             (cur_batch + BLOCK_SIZE.y - 1) / BLOCK_SIZE.y);

        calculateSolarAltitudeAndAzimuthBatchKernel<<<grid_size_batch, BLOCK_SIZE, 0, streams[s]>>>(
            (int)cur_batch, max_n_steps, delta, time_step,
            row_base + target_index_range.row_from, d_geo_transform,
            d_h_array[s], d_A_array[s], d_n_steps[s]);

        calculateRayAndStepHeightChangesBatchKernel<<<grid_size_batch, BLOCK_SIZE, 0, streams[s]>>>(
            (int)cur_batch, max_n_steps, max_search_steps,
            row_base + target_index_range.row_from, d_geo_transform,
            dem.max_value - dem.min_value,
            d_h_array[s], d_A_array[s], d_n_steps[s],
            d_search_steps[s], d_dxs[s], d_dys[s], d_height_changes[s]);


        calculateSunshineHoursBatchKernel_Edge<<<grid_size_batch, BLOCK_SIZE, 0, streams[s]>>>(
            (int)cur_batch,
            row_base + target_index_range.row_from,
            target_index_range.col_from,
            dem.cols,              
            result.cols,
            time_step,
            max_n_steps,
            d_n_steps[s],
            d_dxs[s],
            d_dys[s],
            d_height_changes[s],
            d_search_steps[s],
            max_search_steps,
            (int)dem.rows,           
            tex_dem_obj,
            d_result_row_buffers[s]);

        CHECK_CUDA(cudaMemcpyAsync(h_result_row_buffers[s], d_result_row_buffers[s],
                                   cur_batch * result.cols * sizeof(float), cudaMemcpyDeviceToHost, streams[s]));

        CopyCtx* ctx = new CopyCtx{
            result.data.get() + row_base * result.cols,
            h_result_row_buffers[s],
            cur_batch * result.cols * sizeof(float)
        };
        CHECK_CUDA(cudaLaunchHostFunc(streams[s], [](void* userData) {
            CopyCtx* c = static_cast<CopyCtx*>(userData);
            std::memcpy(c->dst, c->src, c->bytes);
            delete c;
        }, ctx));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    // =================================================================================

    // free memory
    for (int s = 0; s < NUM_STREAMS; ++s)
    {
        CHECK_CUDA(cudaFreeHost(h_dxs[s]));
        CHECK_CUDA(cudaFreeHost(h_dys[s]));
        CHECK_CUDA(cudaFreeHost(h_height_changes[s]));
        CHECK_CUDA(cudaFreeHost(h_search_steps[s]));
        CHECK_CUDA(cudaFree(d_dxs[s]));
        CHECK_CUDA(cudaFree(d_dys[s]));
        CHECK_CUDA(cudaFree(d_height_changes[s]));
        CHECK_CUDA(cudaFree(d_search_steps[s]));
        CHECK_CUDA(cudaFree(d_n_steps[s]));
        CHECK_CUDA(cudaFree(d_h_array[s]));
        CHECK_CUDA(cudaFree(d_A_array[s]));
    }

    for (int s = 0; s < NUM_STREAMS; ++s)
    {
        CHECK_CUDA(cudaStreamDestroy(streams[s]));
        CHECK_CUDA(cudaFree(d_result_row_buffers[s]));
        CHECK_CUDA(cudaFreeHost(h_result_row_buffers[s]));
    }
}
