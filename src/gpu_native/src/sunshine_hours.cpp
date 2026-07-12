#include <iostream>
#include <string>
#include <stdexcept>

#include <gdal_priv.h>

#include "timer.h"
#include "raster.h"
#include "sunshine_hours_cuda.cuh"

using namespace std;

class SunshineHours
{
public:
    SunshineHours(const RasterWithLatitude &target_dem, const Raster &obstacle_dem, const int day_of_year, const int time_step);
    SunshineHours(const RasterWithLatitude &dem, const int day_of_year, const int time_step);
    void calculate();
    Raster& getResult();
    void save(const string &file_path);
    void printCertainResult(int i, int j);
    void printFirstNResult(int n);

private:
    enum CalculationMode
    {
        SINGLE_DEM,
        TARGET_AND_OBSTACLE_DEM
    };

    CalculationMode calculate_mode;
    RasterWithLatitude target_dem;
    Raster obstacle_dem;
    int day_of_year;
    int time_step;

    Raster result;
};

static void print_usage(const char* prog)
{
    cerr << "Usage:\n"
         << "  Single-DEM mode:\n"
         << "    " << prog << " <target_dem.tif> <day_of_year> <time_step_minutes> <output.tif>\n\n"
         << "  Target + obstacle DEM mode:\n"
         << "    " << prog << " <target_dem.tif> <obstacle_dem.tif> <day_of_year> <time_step_minutes> <output.tif>\n";
}

static int parse_int(const char* s, const char* name)
{
    try {
        size_t idx = 0;
        int v = stoi(string(s), &idx);
        if (idx != string(s).size()) {
            throw invalid_argument("extra chars");
        }
        return v;
    } catch (const exception& e) {
        cerr << "Invalid argument: " << name << " = \"" << s << "\" cannot be parsed as an integer\n";
        throw;
    }
}

int main(int argc, char** argv)
{
    if (argc != 5 && argc != 6) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        timer.tick("total");

        GDALAllRegister();

        string target_dem_file_path;
        string obstacle_dem_file_path;
        string result_file_path;
        int day_of_year = 0;
        int time_step   = 0;

        bool use_obstacle_dem = (argc == 6);

        if (!use_obstacle_dem) {
            // Single DEM:
            // argv[1] = target_dem.tif
            // argv[2] = day_of_year
            // argv[3] = time_step_minutes
            // argv[4] = output.tif
            target_dem_file_path = argv[1];
            day_of_year          = parse_int(argv[2], "day_of_year");
            time_step            = parse_int(argv[3], "time_step_minutes");
            result_file_path     = argv[4];
        } else {
            // Target + obstacle DEM:
            // argv[1] = target_dem.tif
            // argv[2] = obstacle_dem.tif
            // argv[3] = day_of_year
            // argv[4] = time_step_minutes
            // argv[5] = output.tif
            target_dem_file_path  = argv[1];
            obstacle_dem_file_path = argv[2];
            day_of_year           = parse_int(argv[3], "day_of_year");
            time_step             = parse_int(argv[4], "time_step_minutes");
            result_file_path      = argv[5];
        }

        timer.tick("Read and prepare DEM");
        RasterWithLatitude target_dem(target_dem_file_path);
        target_dem.printInfo();

        Raster obstacle_dem;
        if (use_obstacle_dem) {
            obstacle_dem = Raster(obstacle_dem_file_path);
        }
        timer.tock(); // Read and prepare DEM

        timer.tick("SunshineHours");
        SunshineHours sunshine_hours = use_obstacle_dem
            ? SunshineHours(target_dem, obstacle_dem, day_of_year, time_step)
            : SunshineHours(target_dem, day_of_year, time_step);

        timer.tick("SunshineHours::calculate");
        sunshine_hours.calculate();
        timer.tock(); // SunshineHours::calculate
        timer.tock(); // SunshineHours

        timer.tick("Save result");
        Raster result = sunshine_hours.getResult();
        result.save(result_file_path);
        timer.tock(); // Save result

        timer.tock(); // total
        timer.print_records();

        // Optional: print the first N valid results for a quick check
        sunshine_hours.printFirstNResult(20);

        return 0;
    }
    catch (const exception& e) {
        cerr << "Runtime error: " << e.what() << endl;
        return 1;
    }
}

/* ================= SunshineHours member implementations ================= */

SunshineHours::SunshineHours(const RasterWithLatitude &target_dem, const Raster &obstacle_dem, const int day_of_year, const int time_step)
{
    this->target_dem = target_dem;
    this->obstacle_dem = obstacle_dem;
    this->day_of_year = day_of_year;
    this->time_step = time_step;
    this->calculate_mode = CalculationMode::TARGET_AND_OBSTACLE_DEM;

    result = Raster(target_dem.rows, target_dem.cols);
    result.projection = target_dem.projection;
    result.copyGeoTransformFrom(target_dem);
    result.no_data_value = NAN;
}

SunshineHours::SunshineHours(const RasterWithLatitude &dem, const int day_of_year, const int time_step)
{
    this->target_dem = dem;
    this->day_of_year = day_of_year;
    this->time_step = time_step;
    this->calculate_mode = CalculationMode::SINGLE_DEM;

    result = Raster(dem.rows, dem.cols);
    result.projection = dem.projection;
    result.copyGeoTransformFrom(dem);
    result.no_data_value = NAN;
}

void SunshineHours::calculate()
{
    switch (calculate_mode)
    {
        case CalculationMode::SINGLE_DEM:
            calculateSunshineHoursCuda(result, target_dem, day_of_year, time_step);
            break;
        case CalculationMode::TARGET_AND_OBSTACLE_DEM:
            calculateSunshineHoursCuda(result, target_dem, obstacle_dem, day_of_year, time_step);
            break;
        default:
            break;
    }
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
    x = result.geo_transform[0] + j * result.geo_transform[1] + i * result.geo_transform[2]
        + 0.5 * result.geo_transform[1] + 0.5 * result.geo_transform[2];
    y = result.geo_transform[3] + j * result.geo_transform[4] + i * result.geo_transform[5]
        + 0.5 * result.geo_transform[4] + 0.5 * result.geo_transform[5];
    cout << "result.data[" << i << "][" << j << "] = "
         << result.data.get()[i * result.cols + j]
         << " at (" << x << ", " << y << ")" << endl;
}

void SunshineHours::printFirstNResult(int n)
{
    int count = 0;
    for (int i = 0; i < result.rows; i++)
    {
        for (int j = 0; j < result.cols; j++)
        {
            double v = result.data.get()[i * result.cols + j];
            if (!isnan(v) && fabs(v) > 0.0001)
            {
                printCertainResult(i, j);
                count++;
            }
            if (count == n) break;
        }
        if (count == n) break;
    }
}
