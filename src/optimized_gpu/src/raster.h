#pragma once

#include <string>
#include <memory>
#include <limits>

using namespace std;

class Raster
{
public:
    Raster() = default;
    Raster(const string &file_path);
    Raster(size_t rows, size_t cols);

    void copyGeoTransformFrom(const Raster &raster);

    size_t size() const;
    void printInfo();
    void save(const string &file_path);

    string file_path;
    shared_ptr<float> data;
    size_t rows = 0;
    size_t cols = 0;
    string projection;
    double geo_transform[6]{0.0, 1.0, 0.0, 0.0, 0.0, -1.0};
    float no_data_value = 65535.0f;

    float max_value = 0.0f;
    float min_value = 0.0f;

    float nodata_write_value = std::numeric_limits<float>::quiet_NaN();
    bool  has_nodata_write_value = false;   
};

struct IndexRange
{
    int row_from = 0;
    int row_to   = 0;  
    int col_from = 0;
    int col_to   = 0;  
};