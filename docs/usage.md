# Implementation Command Reference

The commands below run the three implementations in this repository. Run them from the repository root after completing the relevant build described in `README.md`.

## Optimized GPU

```bash
./build/optimized_gpu/sunshine_hours \
  --file examples/data/test_dem.tif \
  --gpu 0 \
  --batch 16 \
  --streams 16 \
  --blockx 64 \
  --blocky 4 \
  --day 266 \
  --step 5 \
  --pad 1 \
  --output /path/to/output_directory
```

## GPU-Native Reference

```bash
./build/gpu_native/sunshine_hours \
  /path/to/input_dem.tif \
  266 \
  5 \
  /path/to/output.tif
```

## CPU Reference

```bash
./build/cpu_only/sunshine_hours \
  /path/to/input_dem.tif \
  266 \
  5 \
  /path/to/output.tif
```

If GDAL is supplied by an active Conda environment, the runtime library path may need to include the environment library directory:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```
