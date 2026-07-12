#include "raster.h"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstring>
#include <iomanip>

#include <gdal_priv.h>
#include <gdal.h>      // C API
#include <ogr_spatialref.h>
#include <cpl_conv.h>
#include <cpl_error.h>

#include "timer.h"

using namespace std;

static inline bool sane_deg(double v) {
    return std::isfinite(v) && std::abs(v) < 1000.0;
}

Raster::Raster(const string &file_path)
{
    GDALAllRegister();

    timer.tick("Read raster file");

    this->file_path = file_path;
    rows = 0;
    cols = 0;
    for (int i = 0; i < 6; ++i) {
        geo_transform[i] = 0.0;
    }
    projection.clear();
    no_data_value = std::numeric_limits<float>::quiet_NaN();
    max_value = -std::numeric_limits<float>::infinity();
    min_value =  std::numeric_limits<float>::infinity();
    data.reset();

    GDALDataset *p_dataset = static_cast<GDALDataset*>(
        GDALOpen(file_path.c_str(), GA_ReadOnly)
    );
    if (!p_dataset)
    {
        cerr << "ERROR: Cannot open file: \"" << file_path << "\"!" << endl;
        timer.tock();
        return;
    }

    rows = static_cast<size_t>(p_dataset->GetRasterYSize());
    cols = static_cast<size_t>(p_dataset->GetRasterXSize());
    const int nBands = p_dataset->GetRasterCount();
    if (nBands < 1) {
        cerr << "ERROR: Raster has no bands!" << endl;
        GDALClose(p_dataset);
        timer.tock();
        return;
    }

    const char* pj = p_dataset->GetProjectionRef();
    projection = (pj ? pj : "");

    {
        for (int i = 0; i < 6; ++i) geo_transform[i] = 0.0;

        CPLErr err = GDALGetGeoTransform(static_cast<GDALDatasetH>(p_dataset),
                                         geo_transform);

        // std::cerr << "[DEBUG] GDALGetGeoTransform() return code: "
        //           << err << " (0 = CE_None)" << std::endl;

        // if (err == CE_None) {
        //     std::cerr << "[DEBUG] GeoTransform from GDAL: "
        //               << std::fixed << std::setprecision(12)
        //               << geo_transform[0] << " " << geo_transform[1] << " " << geo_transform[2] << " "
        //               << geo_transform[3] << " " << geo_transform[4] << " " << geo_transform[5]
        //               << std::endl;
        // } else {
        //     std::cerr << "ERROR: GetGeoTransform failed for file: \""
        //               << file_path << "\" (err=" << err << ")\n"
        //               << "       GDAL last error: " << CPLGetLastErrorMsg() << std::endl;
        //     GDALClose(p_dataset);
        //     timer.tock();
        //     return;
        // }
    }

    if (std::abs(geo_transform[2]) > 1e-12 || std::abs(geo_transform[4]) > 1e-12)
    {
        cerr << "ERROR: Not support rotation (GT[2]/GT[4] must be 0)!" << endl;
        GDALClose(p_dataset);
        timer.tock();
        return;
    }
    if (std::abs(geo_transform[1] + geo_transform[5]) > 1e-6)
    {
        cerr << "ERROR: Not support different resolution in x and y direction!" << endl;
        GDALClose(p_dataset);
        timer.tock();
        return;
    }

    {
    int ok = 0;
    double nd = p_dataset->GetRasterBand(1)->GetNoDataValue(&ok);
    if (ok) {
        no_data_value = static_cast<float>(nd);
        nodata_write_value = no_data_value;
        has_nodata_write_value = true;
    } else {
        no_data_value = std::numeric_limits<float>::quiet_NaN();
    }
    }

    const size_t N = rows * cols;
    data = shared_ptr<float>(new float[N], default_delete<float[]>());
    CPLErr err = p_dataset->GetRasterBand(1)->RasterIO(
        GF_Read, 0, 0,
        static_cast<int>(cols), static_cast<int>(rows),
        data.get(),
        static_cast<int>(cols), static_cast<int>(rows),
        GDT_Float32, 0, 0
    );
    if (err != CE_None) {
        cerr << "ERROR: RasterIO read failed!" << endl;
        GDALClose(p_dataset);
        timer.tock();
        return;
    }

    const bool nd_is_nan = std::isnan(no_data_value);
    for (size_t i = 0; i < N; ++i) {
        float v = data.get()[i];

        bool is_nodata = false;
        if (!std::isfinite(v)) is_nodata = true;
        else if (nd_is_nan) {
            if (v <= -9000.0f) is_nodata = true;  
        } else {
            if (v == no_data_value) is_nodata = true;
        }

        if (is_nodata) {
            if (!has_nodata_write_value) {
            nodata_write_value = -9999.0f;   
            has_nodata_write_value = true;
            }
            data.get()[i] = std::numeric_limits<float>::quiet_NaN();
            continue;
        }

        if (v > max_value) max_value = v;
        if (v < min_value) min_value = v;
    }

    GDALClose(p_dataset);
    timer.tock();
}

Raster::Raster(size_t rows, size_t cols)
{
    this->rows = rows;
    this->cols = cols;
    const size_t N = rows * cols;
    data = shared_ptr<float>(new float[N], default_delete<float[]>());

    for (int i = 0; i < 6; ++i) geo_transform[i] = 0.0;
    projection.clear();
    no_data_value = std::numeric_limits<float>::quiet_NaN();
    max_value = -std::numeric_limits<float>::infinity();
    min_value =  std::numeric_limits<float>::infinity();
}

void Raster::copyGeoTransformFrom(const Raster &raster)
{
    for (int i = 0; i < 6; i++)
    {
        geo_transform[i] = raster.geo_transform[i];
    }
    projection = raster.projection;
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
        cout.setf(std::ios::fixed);
        cout << std::setprecision(12) << geo_transform[i] << (i==5? "" : " ");
    }
    cout << endl;
    cout << "no_data_value: " << no_data_value << endl;
    cout << "max_value: " << max_value << " min_value: " << min_value << endl;
}

void Raster::save(const string &file_path)
{
    namespace fs = std::filesystem;

    fs::path save_file_path(file_path);
    fs::path save_dir = save_file_path.parent_path();
    if (!save_dir.empty() && !fs::exists(save_dir))
    {
        fs::create_directories(save_dir);
    }

    GDALDriver *p_driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!p_driver)
    {
        cerr << "ERROR: Cannot get GTiff driver!" << endl;
        return;
    }

    char* creationOpts[] = {
        (char*)"TILED=YES",
        (char*)"BLOCKXSIZE=256",
        (char*)"BLOCKYSIZE=256",
        (char*)"COMPRESS=DEFLATE",
        (char*)"PREDICTOR=3",
        (char*)"ZLEVEL=6",
        (char*)"NUM_THREADS=ALL_CPUS",
        nullptr
    };

    GDALDataset *p_dataset = p_driver->Create(
        file_path.c_str(),
        static_cast<int>(cols),
        static_cast<int>(rows),
        1, GDT_Float32, creationOpts
    );
    if (!p_dataset)
    {
        cerr << "ERROR: Cannot create file: \"" << file_path << "\"!" << endl;
        return;
    }

    GDALSetProjection(p_dataset, projection.c_str());

    if (!(sane_deg(geo_transform[0]) && sane_deg(geo_transform[3]) &&
          std::abs(geo_transform[1]) < 1.0 && std::abs(geo_transform[5]) < 1.0 &&
          geo_transform[5] < 0.0))
    {
        std::cerr << "[WARN] Suspicious GT to write: ["
                  << geo_transform[0] << ", " << geo_transform[1] << ", " << geo_transform[2] << ", "
                  << geo_transform[3] << ", " << geo_transform[4] << ", " << geo_transform[5] << "]\n";
    }

    // std::cerr << "[DEBUG] Raster::save() writing GeoTransform: "
    //           << std::fixed << std::setprecision(12)
    //           << geo_transform[0] << " " << geo_transform[1] << " " << geo_transform[2] << " "
    //           << geo_transform[3] << " " << geo_transform[4] << " " << geo_transform[5]
    //           << std::endl;

    CPLErr gtErr = GDALSetGeoTransform(static_cast<GDALDatasetH>(p_dataset),
                                       geo_transform);
    if (gtErr != CE_None) {
        std::cerr << "ERROR: GDALSetGeoTransform failed when saving \""
                  << file_path << "\" (err=" << gtErr << ")\n"
                  << "       GDAL last error: " << CPLGetLastErrorMsg() << std::endl;
    }

    GDALRasterBand *p_band = p_dataset->GetRasterBand(1);
        if (!std::isnan(no_data_value)) {
        p_band->SetNoDataValue(static_cast<double>(no_data_value));
    }
    const float nd_out =
        (has_nodata_write_value && std::isfinite(nodata_write_value))
        ? nodata_write_value
        : -9999.0f;   
    p_band->SetNoDataValue((double)nd_out);

    std::vector<size_t> nan_idx;
    nan_idx.reserve(1024);

    const size_t N = rows * cols;
    float* ptr = data.get();
    for (size_t i = 0; i < N; ++i) {
        if (std::isnan(ptr[i])) {
            nan_idx.push_back(i);
            ptr[i] = nd_out;
        }
    }
    CPLErr err = p_band->RasterIO(
        GF_Write, 0, 0,
        static_cast<int>(cols), static_cast<int>(rows),
        (void*)data.get(),
        static_cast<int>(cols), static_cast<int>(rows),
        GDT_Float32, 0, 0
    );
    if (err != CE_None) {
        cerr << "ERROR: RasterIO write failed!" << endl;
    }

    GDALClose(p_dataset);

    // float nd_out = has_nodata_write_value ? nodata_write_value : -9999.0f;
    // p_band->SetNoDataValue((double)nd_out);

    // const int chunk_rows = 256; 
    // std::vector<float> buf((size_t)cols * chunk_rows);

    // for (int y0 = 0; y0 < (int)rows; y0 += chunk_rows) {
    //     int h = std::min(chunk_rows, (int)rows - y0);

    //     for (int y = 0; y < h; ++y) {
    //         const float* src = data.get() + (size_t)(y0 + y) * cols;
    //         float* dst = buf.data() + (size_t)y * cols;
    //         for (size_t x = 0; x < cols; ++x) {
    //             float v = src[x];
    //             dst[x] = std::isnan(v) ? nd_out : v;
    //         }
    //     }

    //     CPLErr err = p_band->RasterIO(
    //         GF_Write, 0, y0,
    //         (int)cols, h,
    //         (void*)buf.data(),
    //         (int)cols, h,
    //         GDT_Float32, 0, 0
    //     );
    //     if (err != CE_None) {
    //         cerr << "ERROR: RasterIO write failed at y=" << y0 << endl;
    //         break;
    //     }
    // }
}
