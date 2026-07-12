#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
from osgeo import gdal, osr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop a GeoTIFF by trimming a border width in degrees."
    )
    parser.add_argument("input", help="Input TIFF file path")
    parser.add_argument("pad_deg", type=float, help="Border width in degrees")
    parser.add_argument("output", help="Output TIFF file path")
    return parser.parse_args()


def main():
    args = parse_args()

    in_path = args.input
    out_path = args.output
    pad = args.pad_deg

    if pad < 0:
        print("ERROR: pad_deg must not be negative.", file=sys.stderr)
        sys.exit(1)

    gdal.AllRegister()
    ds = gdal.Open(in_path, gdal.GA_ReadOnly)
    if ds is None:
        print(f"ERROR: Cannot open input file: {in_path}", file=sys.stderr)
        sys.exit(1)

    gt = ds.GetGeoTransform()
    # gt: [originX, pixelWidth, 0, originY, 0, pixelHeight]
    px_w = gt[1]
    px_h = gt[5]
    rot_x = gt[2]
    rot_y = gt[4]

    if abs(rot_x) > 1e-12 or abs(rot_y) > 1e-12:
        print("ERROR: Rotated GeoTransforms are not supported; GT[2] and GT[4] must be 0.", file=sys.stderr)
        sys.exit(1)

    cols = ds.RasterXSize
    rows = ds.RasterYSize

    # Original bounds
    left = gt[0]
    top = gt[3]
    right = left + cols * px_w
    bottom = top + rows * px_h  # px_h is usually negative

    # Lightweight coordinate-unit check; warning only
    proj_wkt = ds.GetProjection()
    if proj_wkt:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(proj_wkt)
        try:
            unit_name = srs.GetAngularUnitsName()
        except AttributeError:
            unit_name = None

        if unit_name and unit_name.lower() not in ("degree", "degrees"):
            print(
                f"[WARN] Coordinate unit does not appear to be degrees; detected: {unit_name}. "
                "pad_deg will be interpreted in the same units as the coordinates.",
                file=sys.stderr,
            )

    # Compute cropped bounds by shrinking each side inward by pad
    new_left = left + pad
    new_right = right - pad

    if px_h < 0:
        # y decreases from top to bottom: top > bottom
        new_top = top - pad   # move south by pad
        new_bottom = bottom + pad  # move north by pad; bottom becomes closer to top
    else:
        # Rare case where px_h > 0, south-up raster
        new_top = top + pad
        new_bottom = bottom - pad

    # Check whether pad is too large and makes the bounds invalid
    if new_left >= new_right:
        print("ERROR: pad_deg is too large; cropped east-west extent is zero or negative.", file=sys.stderr)
        sys.exit(1)

    if (px_h < 0 and new_bottom >= new_top) or (px_h > 0 and new_top >= new_bottom):
        print("ERROR: pad_deg is too large; cropped north-south extent is zero or negative.", file=sys.stderr)
        sys.exit(1)

    # Check that the cropped bounds remain inside the source raster
    if not (left <= new_left < new_right <= right):
        print("ERROR: Computed east-west crop bounds are outside the source raster.", file=sys.stderr)
        sys.exit(1)

    if px_h < 0:
        if not (bottom <= new_bottom < new_top <= top):
            print("ERROR: Computed north-south crop bounds are outside the source raster.", file=sys.stderr)
            sys.exit(1)
    else:
        if not (top <= new_top < new_bottom <= bottom):
            print("ERROR: Computed north-south crop bounds are outside the source raster.", file=sys.stderr)
            sys.exit(1)

    # Use gdal.Translate to crop by geographic window
    # projWin = [ulx, uly, lrx, lry]
    proj_win = [new_left, new_top, new_right, new_bottom]

    creation_opts = [
        "TILED=YES",
        "BLOCKXSIZE=256",
        "BLOCKYSIZE=256",
        "COMPRESS=LERC_ZSTD",
        "MAX_Z_ERROR=0",
        "NUM_THREADS=ALL_CPUS",
    ]

    translate_opts = gdal.TranslateOptions(
        format="GTiff",
        projWin=proj_win,
        creationOptions=creation_opts,
    )

    out_ds = gdal.Translate(
        out_path,
        ds,
        options=translate_opts
    )

    if out_ds is None:
        print("ERROR: gdal.Translate crop failed.", file=sys.stderr)
        sys.exit(1)

    out_ds = None
    ds = None

    print("Crop completed.")
    print(f"Input: {in_path}")
    print(f"Output: {out_path}")
    print(f"Original bounds: left={left}, right={right}, top={top}, bottom={bottom}")
    print(f"Cropped bounds: left={new_left}, right={new_right}, top={new_top}, bottom={new_bottom}")


if __name__ == "__main__":
    main()
