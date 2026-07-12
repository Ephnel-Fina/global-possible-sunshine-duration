#pragma once

#include "raster.h"
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include "config.h"

class CalculateSunshineHoursCuda
{
public:
    CalculateSunshineHoursCuda(Raster &result, const Raster &dem, const IndexRange &target_index_range, const int day_of_year, const int time_step, const int cuda_device_id);
    void calculate();
    ~CalculateSunshineHoursCuda() noexcept(false);

private:
    Raster &result;
    const Raster &dem;
    const IndexRange &target_index_range;
    const int day_of_year;
    const int time_step;
    const int cuda_device_id;

    dim3 grid_size;

    cudaArray_t d_dem_array;
    cudaTextureObject_t tex_dem_obj;
    double * d_geo_transform;
};