"""CPU (NumPy) vs GPU (CUDA via pybind11) throughput comparison across sensor
counts. This is the actual deliverable the project is built to prove: the
GPU/CPU speedup gap at scale, with real measured numbers.

GPU path requires the anomaly_gpu extension to be built (see
notebooks/colab_dev.ipynb for the build command) and importable. If it's not
available, this script still runs the CPU side and reports that GPU timing
was skipped, rather than failing outright.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpu_baseline import rolling_zscore_cpu  # noqa: E402

# anomaly_gpu (the pybind11 extension) is built in the repo root, not python/ —
# add the current working directory so it's importable regardless of how this
# script is invoked (`python python/benchmark.py` from the repo root, etc.).
sys.path.insert(0, os.getcwd())

try:
    import anomaly_gpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

SENSOR_COUNTS = [1024, 4096, 16384, 65536]
NUM_TIMESTEPS = 2048
WINDOW = 32
THRESHOLD = 3.0
SEED = 42


def make_data(num_sensors, num_timesteps, seed):
    rng = np.random.default_rng(seed)
    data = (rng.random((num_sensors, num_timesteps), dtype=np.float32) - 0.5) * 2.0
    for s in range(0, num_sensors, 512):
        data[s, num_timesteps // 2] = 50.0
    return data


def run_benchmark():
    results = []
    for num_sensors in SENSOR_COUNTS:
        data = make_data(num_sensors, NUM_TIMESTEPS, SEED)

        start = time.perf_counter()
        cpu_anomalies = rolling_zscore_cpu(data, WINDOW, THRESHOLD)
        cpu_ms = (time.perf_counter() - start) * 1000.0

        entry = {
            "num_sensors": num_sensors,
            "num_timesteps": NUM_TIMESTEPS,
            "cpu_ms": cpu_ms,
            "cpu_anomalies": int(cpu_anomalies.sum()),
        }

        if GPU_AVAILABLE:
            gpu_anomalies, gpu_kernel_ms = anomaly_gpu.rolling_zscore_coalesced(data, WINDOW, THRESHOLD)
            mismatches = int((cpu_anomalies != gpu_anomalies).sum())
            entry["gpu_kernel_ms"] = gpu_kernel_ms
            entry["gpu_anomalies"] = int(gpu_anomalies.sum())
            entry["mismatches_vs_cpu"] = mismatches
            entry["speedup_x"] = cpu_ms / gpu_kernel_ms if gpu_kernel_ms > 0 else None
        else:
            entry["gpu_kernel_ms"] = None
            entry["speedup_x"] = None

        results.append(entry)
        print(json.dumps(entry, indent=2))

    return results


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.json"
    if not GPU_AVAILABLE:
        print("anomaly_gpu extension not importable - running CPU-only benchmark.", file=sys.stderr)

    results = run_benchmark()
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
