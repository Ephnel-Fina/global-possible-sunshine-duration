#pragma once

#include <string>
#include <memory>

using namespace std;

class Raster
{
public:
    Raster() = default;
    Raster(const string &file_path);
    Raster(int rows, int cols);

    void copyGeoTransformFrom(const Raster &raster);

    size_t size() const;
    void printInfo();
    void save(const string &file_path);

    string file_path;
    shared_ptr<float> data;
    int rows = 0;
    int cols = 0;
    string projection;
    double geo_transform[6];
    float no_data_value = 65535;
};

class RasterWithLatitude : public Raster
{
public:
    RasterWithLatitude() = default;
    RasterWithLatitude(const string &file_path);
    RasterWithLatitude(int rows, int cols);
    void calculateLatitude();
    shared_ptr<double> latitude;
};

// TODO: Implement this class if needed
class RasterWithCompressedLatitude : public Raster
{
public:
    RasterWithCompressedLatitude() = default;
    RasterWithCompressedLatitude(const string &file_path);
    RasterWithCompressedLatitude(int rows, int cols);
    void calculateLatitude();
    shared_ptr<float> compressed_latitude;
    shared_ptr<size_t> compressed_latitude_index;

};
