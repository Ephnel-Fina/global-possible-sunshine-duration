#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

for command_name in cmake pkg-config gdal_translate gdalinfo awk grep sed; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "[ERROR] Required command not found: $command_name" >&2
        exit 1
    fi
done

if ! pkg-config --exists gdal; then
    echo "[ERROR] GDAL development files are not visible to pkg-config." >&2
    exit 1
fi

WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/gpsd-quick-test.XXXXXX")
trap 'rm -rf "$WORK_DIR"' EXIT

BUILD_DIR="$WORK_DIR/build"
INPUT_TIF="$WORK_DIR/tiny_dem.tif"
OUTPUT_TIF="$WORK_DIR/sunshine_minutes.tif"
INFO_FILE="$WORK_DIR/output-info.txt"

echo "[INFO] Preparing the example DEM."
gdal_translate \
    -q \
    -of GTiff \
    -a_srs EPSG:4326 \
    "$ROOT_DIR/examples/tiny_dem.asc" \
    "$INPUT_TIF"

echo "[INFO] Building the CPU reference implementation."
cmake \
    -S "$ROOT_DIR/src/cpu_only" \
    -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --parallel

echo "[INFO] Calculating possible sunshine duration."
"$BUILD_DIR/sunshine_hours" "$INPUT_TIF" 80 60 "$OUTPUT_TIF"

if [[ ! -s "$OUTPUT_TIF" ]]; then
    echo "[ERROR] The calculation did not create a non-empty GeoTIFF." >&2
    exit 1
fi

gdalinfo -stats "$OUTPUT_TIF" > "$INFO_FILE"

if ! grep -q "Size is 9, 9" "$INFO_FILE"; then
    echo "[ERROR] The output dimensions are not 9 x 9." >&2
    exit 1
fi

if ! grep -Eq 'ID\["EPSG",4326\]|AUTHORITY\["EPSG","4326"\]' "$INFO_FILE"; then
    echo "[ERROR] The output coordinate reference system is not EPSG:4326." >&2
    exit 1
fi

stats_line=$(grep -m1 'Minimum=.*Maximum=' "$INFO_FILE" || true)
minimum=$(printf '%s\n' "$stats_line" | sed -E 's/.*Minimum=([-+0-9.eE]+), Maximum=.*/\1/')
maximum=$(printf '%s\n' "$stats_line" | sed -E 's/.*Maximum=([-+0-9.eE]+), Mean=.*/\1/')

if [[ -z "$stats_line" || "$minimum" == "$stats_line" || "$maximum" == "$stats_line" ]]; then
    echo "[ERROR] GDAL did not report finite output statistics." >&2
    exit 1
fi

if ! awk -v minimum="$minimum" -v maximum="$maximum" \
    'BEGIN { exit !(minimum >= 0 && maximum >= minimum && maximum <= 1440) }'; then
    echo "[ERROR] Output values are outside the physical range 0 to 1440 minutes." >&2
    exit 1
fi

if ! grep -Eq 'STATISTICS_VALID_PERCENT=(100|100\.0+)' "$INFO_FILE"; then
    echo "[ERROR] The output does not contain the expected valid pixels." >&2
    exit 1
fi

echo "[PASS] CPU quick test completed successfully."
