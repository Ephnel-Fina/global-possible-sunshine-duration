#include "raster.h"
#include <iostream>
#include <filesystem>
#include <gdal_priv.h>
#include <ogr_spatialref.h>

#include "timer.h"

using namespace std;

Raster::Raster(const string &file_path)
{
    timer.tick("Read raster file");

    this->file_path = file_path;

    GDALDataset *p_dataset = (GDALDataset *)GDALOpen(file_path.c_str(), GA_ReadOnly);
    if (p_dataset == NULL)
    {
        cerr << "ERROR: Cannot open file: \"" << file_path << "\"!" << endl;
        return;
    }

    rows = p_dataset->GetRasterYSize();
    cols = p_dataset->GetRasterXSize();
    int nBands = p_dataset->GetRasterCount();
    // cout << "Size: " << cols << " x " << rows << " x " << nBands << endl;

    projection = p_dataset->GetProjectionRef();
    p_dataset->GetGeoTransform(geo_transform);
    no_data_value = p_dataset->GetRasterBand(1)->GetNoDataValue();

    if (geo_transform[2] != 0 || geo_transform[4] != 0)
    {
        cerr << "ERROR: Not support rotation!" << endl;
        return;
    }

    if (abs(geo_transform[1] + geo_transform[5]) > 1e-6)
    {
        cerr << "ERROR: Not support different resolution in x and y direction!" << endl;
        return;
    }

    data = shared_ptr<float>(new float[rows * cols], default_delete<float[]>());
    CPLErr err = p_dataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, cols, rows, data.get(), cols, rows, GDT_Float32, 0, 0);

    for (int i = 0; i < rows * cols; i++)
    {
        if (data.get()[i] == no_data_value)
        {
            data.get()[i] = -NAN;
        }
    }

    GDALClose(p_dataset);

    timer.tock();
}

Raster::Raster(int rows, int cols)
{
    this->rows = rows;
    this->cols = cols;
    data = shared_ptr<float>(new float[rows * cols], default_delete<float[]>());
}

void Raster::copyGeoTransformFrom(const Raster &raster)
{
    for (int i = 0; i < 6; i++)
    {
        geo_transform[i] = raster.geo_transform[i];
    }
}

size_t Raster::size() const
{
    return rows * cols;
}

void Raster::printInfo()
{
    cout << "file_path: " << file_path << endl;
    cout << "rows: " << rows << " cols: " << cols << endl;
    cout << "projection: " << projection << endl;
    cout << "geo_transform: ";
    for (int i = 0; i < 6; i++)
    {
        cout << geo_transform[i] << " ";
    }
    cout << endl;
    cout << "no_data_value: " << no_data_value << endl;
}

void Raster::save(const string &file_path)
{
    filesystem::path save_file_path(file_path);
    filesystem::path save_dir = save_file_path.parent_path();
    if (!filesystem::exists(save_dir))
    {
        filesystem::create_directories(save_dir);
    }

    GDALDriver *p_driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!p_driver)
    {
        cerr << "ERROR: Cannot get driver!" << endl;
        return;
    }

    GDALDataset *p_dataset = p_driver->Create(file_path.c_str(), cols, rows, 1, GDT_Float32, NULL);
    if (!p_dataset)
    {
        cerr << "ERROR: Cannot create file: \"" << file_path << "\"!" << endl;
        return;
    }

    GDALSetProjection(p_dataset, projection.c_str());
    p_dataset->SetGeoTransform(geo_transform);

    GDALRasterBand *p_band = p_dataset->GetRasterBand(1);
    p_band->SetNoDataValue(no_data_value);
    
    CPLErr err = p_band->RasterIO(GF_Write, 0, 0, cols, rows, data.get(), cols, rows, GDT_Float32, 0, 0);

    GDALClose(p_dataset);
    return;
}

RasterWithLatitude::RasterWithLatitude(const string &file_path) : Raster(file_path)
{
    calculateLatitude();
}

RasterWithLatitude::RasterWithLatitude(int rows, int cols) : Raster(rows, cols)
{
}

void RasterWithLatitude::calculateLatitude()
{
    timer.tick("Calculate latitude");

    OGRSpatialReference srs;
    srs.importFromWkt(projection.c_str());

    OGRSpatialReference wgs84;
    wgs84.importFromEPSG(4326);
    wgs84.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

    OGRCoordinateTransformation *transform = OGRCreateCoordinateTransformation(&srs, &wgs84);
    if (!transform)
    {
        cerr << "ERROR: Cannot create coordinate transformation!" << endl;
        return;
    }

    shared_ptr<double> x(new double[rows * cols], default_delete<double[]>());
    shared_ptr<double> y(new double[rows * cols], default_delete<double[]>());

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            x.get()[i * cols + j] = geo_transform[0] + j * geo_transform[1] + i * geo_transform[2];
            y.get()[i * cols + j] = geo_transform[3] + j * geo_transform[4] + i * geo_transform[5];
        }
    }

    if (!transform->Transform(rows * cols, x.get(), y.get()))
    {
        cerr << "ERROR: Cannot transform coordinates!" << endl;
        return;
    }

    latitude = y;

    timer.tock();
}
