#!/usr/bin/env python3
import argparse
import os
import pathlib
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from files_group import main as group_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Schedule optimized sunshine-duration jobs by GPU and NUMA node."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing input GeoTIFF files.")
    parser.add_argument("--output-dir", required=True, help="Root directory for outputs and logs.")
    parser.add_argument("--executable", required=True, help="Path to the optimized GPU executable.")
    parser.add_argument("--failure-log", help="Failure log to read in retry mode.")
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Schedule entries from the failure log instead of scanning the input directory.",
    )
    parser.add_argument("--groups", type=int, default=8, help="File-size balancing groups (default: 8).")
    parser.add_argument("--batch", type=int, default=16, help="Rows per GPU batch (default: 16).")
    parser.add_argument("--streams", type=int, default=16, help="CUDA stream count (default: 16).")
    parser.add_argument("--blockx", type=int, default=64, help="CUDA block x dimension (default: 64).")
    parser.add_argument("--blocky", type=int, default=4, help="CUDA block y dimension (default: 4).")
    parser.add_argument("--day", type=int, default=249, help="Day of year, or 365 for annual mode.")
    parser.add_argument("--step", type=int, default=5, help="Time step in minutes (default: 5).")
    parser.add_argument("--pad", type=float, default=1.0, help="Output padding in degrees (default: 1).")
    return parser.parse_args()


def parse_nvidia_topology():
    try:
        raw = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("nvidia-smi topo -m failed") from error

    ansi_re = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
    lines = [ansi_re.sub("", line) for line in raw.splitlines()]
    header = next((line for line in lines if "NUMA Affinity" in line), None)
    if header is None:
        raise RuntimeError("The topology output has no NUMA Affinity column")

    numa_start = header.index("NUMA Affinity")
    numa_end = header.find("GPU NUMA ID", numa_start)
    if numa_end < 0:
        numa_end = len(header)

    mapping = {}
    for line in lines:
        if not re.match(r"^GPU\d+", line):
            continue
        gpu_id = int(re.match(r"^GPU(\d+)", line).group(1))
        value = line[numa_start:numa_end].strip().split()
        if value and re.fullmatch(r"-?\d+", value[0]):
            mapping[gpu_id] = int(value[0])

    if not mapping:
        raise RuntimeError("No GPU-to-NUMA mappings could be parsed")
    return mapping


def cpu_list_for_numa(numa_id):
    try:
        with open(f"/sys/devices/system/node/node{numa_id}/cpulist", encoding="utf-8") as handle:
            return handle.read().strip()
    except FileNotFoundError:
        return ""


def read_retry_tasks(failure_log):
    tasks = []
    with open(failure_log, encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            dem_path = parts[0]
            if len(parts) == 1 or parts[1] == "all":
                tasks.append((dem_path, 365))
                continue
            for value in parts[1].split(","):
                if value.isdigit():
                    tasks.append((dem_path, int(value)))
    return tasks


def build_tasks(arguments):
    failure_log = arguments.failure_log or os.path.join(arguments.output_dir, "faillog.txt")
    if arguments.retry_failures:
        if not os.path.isfile(failure_log):
            raise RuntimeError(f"Failure log does not exist: {failure_log}")
        tasks = read_retry_tasks(failure_log)
        if not tasks:
            raise RuntimeError(f"No retryable records found in: {failure_log}")
        return tasks

    groups = group_files(arguments.input_dir, arguments.groups)
    if not groups:
        raise RuntimeError(f"No GeoTIFF files found under: {arguments.input_dir}")
    return [(file_path, arguments.day) for files in groups.values() for file_path in files]


def run_one(gpu_id, numa_id, file_path, day, arguments):
    cpu_list = cpu_list_for_numa(numa_id)
    command = [
        "numactl",
        f"--cpunodebind={numa_id}",
        f"--membind={numa_id}",
        arguments.executable,
        "--file",
        file_path,
        "--gpu",
        str(gpu_id),
        "--batch",
        str(arguments.batch),
        "--streams",
        str(arguments.streams),
        "--blockx",
        str(arguments.blockx),
        "--blocky",
        str(arguments.blocky),
        "--day",
        str(day),
        "--step",
        str(arguments.step),
        "--pad",
        str(arguments.pad),
        "--output",
        arguments.output_dir,
    ]

    log_dir = pathlib.Path(arguments.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{pathlib.Path(file_path).stem}_gpu{gpu_id}.log"

    print(f"[INFO] GPU{gpu_id} NUMA{numa_id} ({cpu_list}) -> {shlex.join(command)}")
    with open(log_path, "w", encoding="utf-8") as log_file:
        subprocess.run(
            command,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    print(f"[DONE] {file_path} on GPU{gpu_id}; log: {log_path}")


def main():
    arguments = parse_args()
    if arguments.groups <= 0:
        raise RuntimeError("--groups must be a positive integer")
    if not os.path.isfile(arguments.executable):
        raise RuntimeError(f"Executable does not exist: {arguments.executable}")

    topology = parse_nvidia_topology()
    gpu_ids = sorted(topology)
    tasks = build_tasks(arguments)
    print(f"[INFO] Detected GPU-to-NUMA mapping: {topology}")

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = []
        for index, (file_path, day) in enumerate(tasks):
            gpu_id = gpu_ids[index % len(gpu_ids)]
            numa_id = topology[gpu_id]
            futures.append(executor.submit(run_one, gpu_id, numa_id, file_path, day, arguments))
        for future in futures:
            future.result()

    print("[DONE] All tasks completed.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)
