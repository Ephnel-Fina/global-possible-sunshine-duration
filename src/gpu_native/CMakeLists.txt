cmake_minimum_required(VERSION 3.21)

if(NOT DEFINED CUDAToolkit_ROOT)
    if(DEFINED ENV{CUDA_HOME})
        set(CUDAToolkit_ROOT "$ENV{CUDA_HOME}" CACHE PATH "CUDA Toolkit root")
    elseif(EXISTS "/usr/local/cuda")
        set(CUDAToolkit_ROOT "/usr/local/cuda" CACHE PATH "CUDA Toolkit root")
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_COMPILER)
    if(DEFINED CUDAToolkit_ROOT AND EXISTS "${CUDAToolkit_ROOT}/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc")
    endif()
endif()

project(sunshine_hours LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Configure CUDA package discovery.
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 NEW)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH ON)
pkg_check_modules(GDAL REQUIRED IMPORTED_TARGET gdal)

add_executable(sunshine_hours
    src/timer.h
    src/raster.h
    src/raster.cpp
    src/sunshine_hours.cpp
    src/sunshine_hours_cuda.cu
    src/sunshine_hours_cuda.cuh
)

target_compile_options(sunshine_hours PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-O2>
)

target_link_libraries(sunshine_hours PRIVATE
    PkgConfig::GDAL
    CUDA::cudart
    OpenMP::OpenMP_CXX
)
