#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

gdal.UseExceptions()


def read_raster(path):
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"Cannot open raster: {path}")

    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray().astype(float)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        array[array == nodata] = np.nan

    transform = dataset.GetGeoTransform()
    extent = (
        transform[0],
        transform[0] + dataset.RasterXSize * transform[1],
        transform[3] + dataset.RasterYSize * transform[5],
        transform[3],
    )
    return array, extent


def annotate_cells(axis, values, extent):
    rows, cols = values.shape
    if rows * cols > 400:
        return
    x_min, x_max, y_min, y_max = extent
    x_step = (x_max - x_min) / cols
    y_step = (y_max - y_min) / rows
    value_min = np.nanmin(values)
    value_max = np.nanmax(values)
    value_span = max(value_max - value_min, 1.0)
    for row in range(rows):
        for col in range(cols):
            value = values[row, col]
            if np.isnan(value):
                continue
            x = x_min + (col + 0.5) * x_step
            y = y_max - (row + 0.5) * y_step
            normalized = (value - value_min) / value_span
            text_color = "white" if normalized < 0.32 else "black"
            axis.text(
                x,
                y,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=6,
                color=text_color,
            )


def main():
    parser = argparse.ArgumentParser(description="Plot the DEM and GPU test result.")
    parser.add_argument("dem", type=Path, help="Input DEM GeoTIFF.")
    parser.add_argument("result", type=Path, help="GPU sunshine-duration GeoTIFF.")
    parser.add_argument("output", type=Path, help="Output PNG file.")
    arguments = parser.parse_args()

    dem, dem_extent = read_raster(arguments.dem)
    result, result_extent = read_raster(arguments.result)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    figure.suptitle("Paper DEM GPU example: day 266, 5-minute sampling", fontsize=14)

    dem_image = axes[0].imshow(
        dem,
        cmap="terrain",
        interpolation="nearest",
        extent=dem_extent,
        vmin=np.nanmin(dem),
        vmax=np.nanmax(dem),
    )
    axes[0].set_title("3-degree input DEM")
    axes[0].set_xlabel("Longitude (degrees)")
    axes[0].set_ylabel("Latitude (degrees)")
    annotate_cells(axes[0], dem, dem_extent)
    figure.colorbar(dem_image, ax=axes[0], label="Elevation")

    result_image = axes[1].imshow(
        result,
        cmap="viridis",
        interpolation="nearest",
        extent=result_extent,
        vmin=np.nanmin(result),
        vmax=np.nanmax(result),
    )
    axes[1].set_title("1-degree possible sunshine duration")
    axes[1].set_xlabel("Longitude (degrees)")
    axes[1].set_ylabel("Latitude (degrees)")
    annotate_cells(axes[1], result, result_extent)
    figure.colorbar(result_image, ax=axes[1], label="Minutes")

    for axis in axes:
        axis.set_aspect("equal")
        axis.tick_params(labelsize=8)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
