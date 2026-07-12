#pragma once
#include <cuda_runtime.h>

// Global configuration parameters; concrete values are defined in main.cpp
extern int BATCH;            // Rows processed per batch
extern int NUM_STREAMS;      // Number of CUDA streams
extern dim3 BLOCK_SIZE;      // Block dimensions
extern int CUDA_DEVICE_ID;   // CUDA device ID

// Sunshine calculation parameters
extern int DAY_OF_YEAR;      // Day of year to compute
extern int TIME_STEP;        // Time step in minutes
extern float PADDING_DEGREE; // Output edge padding from the DEM boundary in degrees 
extern std::string FILE_PATH;
extern std::string OUTPUT_PATH;