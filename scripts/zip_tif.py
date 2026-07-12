import argparse
import os
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from osgeo import gdal, gdalconst

gdal.UseExceptions()

@contextmanager
def gdal_env(**opts):
    old = {}
    for k, v in opts.items():
        old[k] = gdal.GetConfigOption(k)
        gdal.SetConfigOption(k, str(v))
        os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                gdal.SetConfigOption(k, None)
                os.environ.pop(k, None)
            else:
                gdal.SetConfigOption(k, v)
                os.environ[k] = v

def get_global_min_max_fast(tif_path: str):
    ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open: {tif_path}")
    b = ds.GetRasterBand(1)
    mn, mx = b.GetMinimum(), b.GetMaximum()
    if mn is not None and mx is not None:
        return float(mn), float(mx), "minmax_tags"
    md = b.GetMetadata()
    if "STATISTICS_MINIMUM" in md and "STATISTICS_MAXIMUM" in md:
        return float(md["STATISTICS_MINIMUM"]), float(md["STATISTICS_MAXIMUM"]), "cached_stats"
    mn, mx = b.ComputeRasterMinMax(True)
    return float(mn), float(mx), "approx"

def _encode_tile(args):
    path, band_idx, x, y, w, h, vmin, anchor_vmin, use_byte, nd_out = args
    t0 = time.perf_counter()

    ds = gdal.Open(path, gdal.GA_ReadOnly)
    b  = ds.GetRasterBand(band_idx)

    a  = b.ReadAsArray(x, y, w, h, buf_type=gdal.GDT_Int32)
    t_read = time.perf_counter()

    flags = b.GetMaskFlags()
    valid = None
    if not (flags & gdalconst.GMF_ALL_VALID):
        if flags & (gdalconst.GMF_PER_DATASET | gdalconst.GMF_NODATA | gdalconst.GMF_ALPHA):
            m = b.GetMaskBand().ReadAsArray(x, y, w, h)
            valid = (m != 0)
        ndv = b.GetNoDataValue()
        if ndv is not None:
            if valid is None:
                valid = (a != ndv)
            else:
                valid &= (a != ndv)

    if anchor_vmin:
        np.subtract(a, int(vmin), out=a, casting="unsafe")
    np.floor_divide(a, 5, out=a)

    if use_byte:
        out = np.full(a.shape, 255, dtype=np.uint8)
        if valid is None: out[...] = a.astype(np.uint8, copy=False)
        else:             out[valid] = a[valid].astype(np.uint8, copy=False)
    else:
        out = np.full(a.shape, 65535, dtype=np.uint16)
        if valid is None: out[...] = a.astype(np.uint16, copy=False)
        else:             out[valid] = a[valid].astype(np.uint16, copy=False)

    t_comp = time.perf_counter()
    # Return CPU time for this tile
    return x, y, out, (t_read - t0), (t_comp - t_read)

def encode_by5_auto_dtype_stream_vec(
    src_tif: str,
    dst_tif: str,
    zstd_level: int = 19,
    block_size: int = 1024,
    workers: int | None = None,
    anchor_vmin: bool = True,
    write_scale_offset: bool = True
):
    t_total0 = time.perf_counter()

    with gdal_env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_PAM_ENABLED="NO",
        GDAL_CACHEMAX="8192",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1"
    ):
        ds = gdal.Open(src_tif, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"Cannot open: {src_tif}")
        b  = ds.GetRasterBand(1)
        X, Y = ds.RasterXSize, ds.RasterYSize
        gt, prj = ds.GetGeoTransform(), ds.GetProjection()

        bsx, bsy = b.GetBlockSize()
        if (bsx == X and bsy == 1) or (bsx < block_size or bsy < block_size):
            bsx = bsy = int(block_size)

        vmin, vmax, src = get_global_min_max_fast(src_tif)
        if anchor_vmin:
            enc_max = int((vmax - vmin) // 5)
        else:
            enc_max = int(vmax // 5); vmin = 0

        use_byte = (enc_max <= 254)
        nd_out   = 255 if use_byte else 65535
        out_gdt  = gdal.GDT_Byte if use_byte else gdal.GDT_UInt16

        tasks = [(src_tif, 1, x0, y0, min(bsx, X-x0), min(bsy, Y-y0), vmin, anchor_vmin, use_byte, nd_out)
                 for y0 in range(0, Y, bsy)
                 for x0 in range(0, X, bsx)]

        co = [
            "TILED=YES",
            f"BLOCKXSIZE={block_size}",
            f"BLOCKYSIZE={block_size}",
            "COMPRESS=ZSTD",
            f"ZSTD_LEVEL={zstd_level}",
            "PREDICTOR=2",
            "NUM_THREADS=ALL_CPUS",
        ]
        out = gdal.GetDriverByName("GTiff").Create(dst_tif, X, Y, 1, out_gdt, options=co)
        out.SetGeoTransform(gt)
        out.SetProjection(prj)
        ob = out.GetRasterBand(1)
        ob.SetNoDataValue(nd_out)
        if write_scale_offset and anchor_vmin:
            ob.SetScale(5.0)
            ob.SetOffset(float(vmin))

        if workers is None:
            workers = min(32, os.cpu_count() or 8)

        cpu_read_sum = cpu_comp_sum = 0.0
        write_wall   = 0.0

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for x0, y0, tile, rsec, csec in ex.map(_encode_tile, tasks):
                cpu_read_sum += rsec
                cpu_comp_sum += csec
                t0w = time.perf_counter()
                ob.WriteArray(tile, xoff=x0, yoff=y0)
                write_wall += (time.perf_counter() - t0w)

        out.FlushCache()
        out = None

    total_wall = time.perf_counter() - t_total0
    parallel_wall_est = max(total_wall - write_wall, 0.0) 

    print(f"[OK] {src_tif} -> {dst_tif} | dtype={'Byte' if use_byte else 'UInt16'} "
          f"| vmin={vmin} vmax={vmax} ({src}) | enc_max={enc_max} | "
          f"ZSTD_LEVEL={zstd_level} BLOCK={block_size} WORKERS={workers}")
    print(f"[TIME] parallel(read+compute)~{parallel_wall_est:.3f}s (cpu_read_sum={cpu_read_sum:.3f}s, cpu_compute_sum={cpu_comp_sum:.3f}s) "
          f"write/compress={write_wall:.3f}s total={total_wall:.3f}s")

    return {
        "parallel_wall_est": parallel_wall_est,   # Estimated wall time for the parallel stage
        "cpu_read_sum": cpu_read_sum,             # Sum of per-thread read time, CPU time
        "cpu_compute_sum": cpu_comp_sum,          # Sum of per-thread compute time, CPU time
        "write_wall": write_wall,                 # Wall time for write/compression
        "total_wall": total_wall,                 # Total wall time
        "dtype": "Byte" if use_byte else "UInt16",
        "enc_max": enc_max,
        "vmin": vmin,
        "vmax": vmax,
        "minmax_source": src
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compress a GeoTIFF after quantizing elevations to five-unit steps."
    )
    parser.add_argument("source", help="Input GeoTIFF.")
    parser.add_argument("destination", help="Output GeoTIFF.")
    parser.add_argument("--zstd-level", type=int, default=19, help="ZSTD level (default: 19).")
    parser.add_argument("--block-size", type=int, default=1024, help="Tile size (default: 1024).")
    parser.add_argument("--workers", type=int, default=None, help="Worker threads (default: automatic).")
    parser.add_argument(
        "--no-anchor-vmin",
        action="store_true",
        help="Do not subtract the source minimum before quantization.",
    )
    parser.add_argument(
        "--no-scale-offset",
        action="store_true",
        help="Do not write GDAL scale and offset metadata.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    encode_by5_auto_dtype_stream_vec(
        arguments.source,
        arguments.destination,
        zstd_level=arguments.zstd_level,
        block_size=arguments.block_size,
        workers=arguments.workers,
        anchor_vmin=not arguments.no_anchor_vmin,
        write_scale_offset=not arguments.no_scale_offset,
    )
