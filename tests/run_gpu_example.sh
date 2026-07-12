#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU_ID=${GPU_ID:-0}
CUDA_ARCHITECTURE=${CUDA_ARCHITECTURE:-80}
BUILD_DIR=${BUILD_DIR:-"$ROOT_DIR/build/optimized_gpu"}
GPU_EXAMPLE_DIR=${GPU_EXAMPLE_DIR:-"$ROOT_DIR/test-results/gpu-example"}
PAPER_DEM="$ROOT_DIR/examples/data/test_dem.tif"
WORKING_DEM="$GPU_EXAMPLE_DIR/test_dem.tif"
OUTPUT_ROOT="$GPU_EXAMPLE_DIR/output"
RESULT_TIF="$OUTPUT_ROOT/test_dem/test_dem--_result266.tif"

for command_name in cmake pkg-config gdal_translate gdalinfo nvidia-smi; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "[ERROR] Required command not found: $command_name" >&2
        exit 1
    fi
done

if ! pkg-config --exists gdal; then
    echo "[ERROR] GDAL development files are not visible to pkg-config." >&2
    exit 1
fi

if [[ ! -s "$PAPER_DEM" ]] || [[ $(stat -c '%s' "$PAPER_DEM") -lt 1000000 ]]; then
    echo "[ERROR] The paper DEM is missing or is only a Git LFS pointer." >&2
    echo "        Run: git lfs pull --include=examples/data/test_dem.tif" >&2
    exit 1
fi

nvidia-smi --query-gpu=index,name --format=csv,noheader | grep -Eq "^${GPU_ID},"
mkdir -p "$GPU_EXAMPLE_DIR"

echo "[INFO] Preparing the input DEM."
gdal_translate \
    -q \
    -of GTiff \
    -co TILED=YES \
    -co COMPRESS=DEFLATE \
    -co PREDICTOR=3 \
    -co NUM_THREADS=ALL_CPUS \
    "$PAPER_DEM" \
    "$WORKING_DEM"

echo "[INFO] Building the optimized GPU implementation."
cmake \
    -S "$ROOT_DIR/src/optimized_gpu" \
    -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURE"
cmake --build "$BUILD_DIR" --parallel

echo "[INFO] Running the 3-degree paper DEM on GPU $GPU_ID."
"$BUILD_DIR/sunshine_hours" \
    --file "$WORKING_DEM" \
    --gpu "$GPU_ID" \
    --batch 16 \
    --streams 16 \
    --blockx 64 \
    --blocky 4 \
    --day 266 \
    --step 5 \
    --pad 1 \
    --output "$OUTPUT_ROOT"

if [[ ! -s "$RESULT_TIF" ]]; then
    echo "[ERROR] The GPU calculation did not create the expected GeoTIFF." >&2
    exit 1
fi

gdalinfo -stats "$RESULT_TIF"
echo "[PASS] GPU paper example completed: $RESULT_TIF"
