#pragma once
#include "raster.h"

// Single DEM: compute theoretical day length without terrain shadowing
void calculateSunshineHoursCpu(
    Raster &sunshine_hours,
    const RasterWithLatitude &dem,
    int day_of_year,
    int time_step_minutes);
