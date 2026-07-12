#include <iostream>
#include <cmath>

#include "cuda_core.cuh"
#include "sunshine_hours.h"
#include "config.h"

using namespace std;

SunshineHours::SunshineHours(const Raster &dem, const float padding_degree, const int day_of_year, const int time_step)
{
    this->dem = dem;
    this->padding_degree = padding_degree;
    this->day_of_year = day_of_year;
    this->time_step = time_step;

    index_range = calculateIndexRange();

    result = Raster(index_range.row_to - index_range.row_from, index_range.col_to - index_range.col_from);
    result.projection = dem.projection;
    
    // set the geo transform of the result raster
    result.geo_transform[0] = dem.geo_transform[0] + index_range.col_from * dem.geo_transform[1];  // longtitude of top left corner
    result.geo_transform[1] = dem.geo_transform[1];  // pixel width
    result.geo_transform[2] = dem.geo_transform[2];  // rotation
    result.geo_transform[3] = dem.geo_transform[3] + index_range.row_from * dem.geo_transform[5];  // latitude of top left corner
    result.geo_transform[4] = dem.geo_transform[4];  // rotation
    result.geo_transform[5] = dem.geo_transform[5];  // pixel height
    
    result.no_data_value = NAN;
}

IndexRange SunshineHours::calculateIndexRange()
{
    IndexRange range{};

    const double* G = dem.geo_transform;             
    const int W = static_cast<int>(dem.cols);
    const int H = static_cast<int>(dem.rows);

    const double dem_min_x = G[0];
    const double dem_max_x = G[0] + W*G[1] + H*G[2];
    const double dem_max_y = G[3];
    const double dem_min_y = G[3] + W*G[4] + H*G[5];    

    const double pad = padding_degree;                

    const double res_min_x = dem_min_x + pad;
    const double res_max_x = dem_max_x - pad;
    const double res_max_y = dem_max_y - pad;
    const double res_min_y = dem_min_y + pad;

    int col_from = static_cast<int>(std::ceil((res_min_x - dem_min_x) / G[1]));
    int col_to   = static_cast<int>(std::floor((res_max_x - dem_min_x) / G[1])) + 1;
    int row_from = static_cast<int>(std::ceil((res_max_y - dem_max_y) / G[5])); 
    int row_to   = static_cast<int>(std::floor((res_min_y - dem_max_y) / G[5])) + 1;

    col_from = std::min(std::max(col_from, 0), W);
    col_to   = std::min(std::max(col_to,   0), W);
    row_from = std::min(std::max(row_from, 0), H);
    row_to   = std::min(std::max(row_to,   0), H);

    if (col_to < col_from) std::swap(col_to, col_from);
    if (row_to < row_from) std::swap(row_to, row_from);

    range.col_from = col_from;
    range.col_to   = col_to;
    range.row_from = row_from;
    range.row_to   = row_to;
    return range;
}

void SunshineHours::calculate()
{
    CalculateSunshineHoursCuda calculateSunshineHoursCuda(result, dem, index_range, day_of_year, time_step, CUDA_DEVICE_ID);
    calculateSunshineHoursCuda.calculate();
}

Raster& SunshineHours::getResult()
{
    return result;
}

void SunshineHours::save(const string &file_path)
{
    result.save(file_path);
}

void SunshineHours::printCertainResult(int i, int j)
{
    double x, y;
    x = result.geo_transform[0] + j * result.geo_transform[1] + i * result.geo_transform[2] + 0.5 * result.geo_transform[1] + 0.5 * result.geo_transform[2];
    y = result.geo_transform[3] + j * result.geo_transform[4] + i * result.geo_transform[5] + 0.5 * result.geo_transform[4] + 0.5 * result.geo_transform[5];
    cout << "result.data[" << i << "][" << j << "] = " << result.data.get()[i * result.cols + j] << " at (" << x << ", " << y << ")" << endl;
}

void SunshineHours::printFirstNResult(int n)
{
    int count = 0;
    for (int i = 0; i < result.rows; i++)
    {
        for (int j = 0; j < result.cols; j++)
        {
            if (!isnan(result.data.get()[i * result.cols + j]) && abs(result.data.get()[i * result.cols + j]) > 0.0001)
            {
                printCertainResult(i, j);
                count++;
            }
            if (count == n)
            {
                break;
            }
        }
        if (count == n)
        {
            break;
        }
    }
}
