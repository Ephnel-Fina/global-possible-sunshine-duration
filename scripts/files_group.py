#!/usr/bin/env python3
import argparse
import json
import os
import sys


def get_files(folder):
    files = []
    for root, _, filenames in os.walk(folder):
        for name in filenames:
            if not name.lower().endswith((".tif", ".tiff")):
                continue
            full_path = os.path.join(root, name)
            try:
                size = os.path.getsize(full_path)
            except OSError:
                continue
            rel_path = os.path.relpath(full_path, folder)
            files.append((rel_path, size))
    return files


def partition_files(files, n_parts):
    if n_parts <= 0:
        raise ValueError("n_parts must be a positive integer")

    bins = [{"total": 0, "files": []} for _ in range(n_parts)]
    files_sorted = sorted(files, key=lambda item: item[1], reverse=True)
    for rel_path, size in files_sorted:
        smallest_bin = min(bins, key=lambda item: item["total"])
        smallest_bin["files"].append((rel_path, size))
        smallest_bin["total"] += size
    return bins


def calc_small_ratio(files, threshold=3 * 1024**3):
    total = len(files)
    if total == 0:
        return 0, 0, 0.0
    small_count = sum(1 for _, size in files if size <= threshold)
    return small_count, total, small_count / total


def human_readable(size):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}PB"


def main(folder, n):
    if not os.path.isdir(folder):
        raise ValueError(f"'{folder}' is not a valid directory")

    files = get_files(folder)
    if not files:
        return {}

    bins = partition_files(files, n)
    return {
        index: [os.path.join(folder, rel_path) for rel_path, _ in item["files"]]
        for index, item in enumerate(bins)
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Balance GeoTIFF files among groups using their file sizes."
    )
    parser.add_argument("folder", help="Directory searched recursively for GeoTIFF files.")
    parser.add_argument(
        "--groups",
        type=int,
        default=8,
        help="Number of output groups (default: 8).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        grouped_files = main(arguments.folder, arguments.groups)
    except ValueError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)

    if not grouped_files:
        print("[ERROR] No GeoTIFF files were found.", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(grouped_files, indent=2))
