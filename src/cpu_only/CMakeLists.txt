cmake_minimum_required(VERSION 3.21)
project(sunshine_hours LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH ON)
pkg_check_modules(GDAL REQUIRED IMPORTED_TARGET gdal)

add_executable(sunshine_hours
    src/timer.h
    src/raster.h
    src/raster.cpp
    src/sunshine_hours.cpp
    src/sunshine_hours_cpu.cpp
    src/sunshine_hours_cpu.h
)

target_compile_options(sunshine_hours PRIVATE -O2)

target_link_libraries(sunshine_hours PRIVATE
    PkgConfig::GDAL
    OpenMP::OpenMP_CXX
)
