#pragma once

#include "raster.h"

// void calculateSunshineHoursCuda(Raster & sunshine_hours, const RasterWithLatitude &target_dem, const Raster &obstacle_dem, int day_of_year, int time_step, float search_radius);
void calculateSunshineHoursCuda(Raster & sunshine_hours, const RasterWithLatitude &target_dem, const Raster &obstacle_dem, const int day_of_year, const int time_step);
void calculateSunshineHoursCuda(Raster & sunshine_hours, const RasterWithLatitude &dem, const int day_of_year, const int time_step);