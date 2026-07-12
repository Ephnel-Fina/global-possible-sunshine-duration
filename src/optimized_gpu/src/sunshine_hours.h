#pragma once

#include "raster.h"

using namespace std;

class SunshineHours
{
public:
    /**
     * @brief Constructor
     * @param dem The DEM raster
     * @param padding_degree The padding degree
     * @param day_of_year The day of year
     * @param time_step The time step
     */
    SunshineHours(const Raster &dem, const float padding_degree, const int day_of_year, const int time_step);

    /**
     * @brief Calculate the sunshine hours
     */
    void calculate();

    /**
     * @brief Get the result raster
     */
    Raster& getResult();

    /**
     * @brief Save the result raster to a file
     * @param file_path The path to the file
     */
    void save(const string &file_path);

    /**
     * @brief Print the first n result
     * @param n The number of results to print
     */
    void printFirstNResult(int n);

    /**
     * @brief Print the certain result
     * @param i The row index
     * @param j The column index
     */
    void printCertainResult(int i, int j);

private:
    /**
     * @brief Calculate the index range
     */
    IndexRange calculateIndexRange();

    Raster dem;
    float padding_degree;
    IndexRange index_range;
    int day_of_year;
    int time_step;

    Raster result;
};