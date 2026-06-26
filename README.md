# GPU-Accelerated Sensor Anomaly Detector

CUDA-accelerated rolling z-score anomaly detector across many simulated sensor streams, benchmarked against a CPU/NumPy baseline.

Industrial/IoT sensor monitoring (vibration, temperature, pressure, current draw) generates high-frequency data across thousands of streams. A CPU pipeline can't keep up with real-time anomaly detection at scale — you either drop sampling rate or fall behind. This project proves a GPU pipeline can, with measured numbers.

## Status

Week 2 of 4 complete: kernel correctness validated against a NumPy reference, and the memory-access-pattern optimization measured.

GPU development happens on a Colab T4 GPU (`notebooks/colab_dev.ipynb`) rather than a local CUDA install — see "Dev environment" below.

## Structure

- `cuda/kernels.cu`, `cuda/kernels.cuh` — kernel definitions + host launcher functions, shared by the standalone executables and the pybind11 bindings
- `cuda/vector_add.cu` — toolchain sanity check (classic CUDA warm-up)
- `cuda/rolling_zscore.cu` — v1: naive sensor-major layout (uncoalesced warp reads)
- `cuda/rolling_zscore_coalesced.cu` — v2: time-major layout (coalesced warp reads), same algorithm
- `cuda/bindings.cpp` — pybind11 extension exposing both kernels to Python
- `python/generate_data.py` — synthetic sensor data generator, shared input format for CUDA and NumPy
- `python/cpu_baseline.py` — vectorized NumPy reference implementation (cumulative-sum based, O(n) memory)
- `python/validate.py` — exact diff between CPU and GPU anomaly output on identical input
- `python/benchmark.py` — CPU vs GPU throughput comparison across sensor counts
- `backend/main.py` — FastAPI endpoint: generate data, run CPU + GPU, return anomalies + timing
- `notebooks/colab_dev.ipynb` — GPU dev/build/validate/benchmark workflow

## Dev environment

Local CUDA Toolkit install hit disk-space constraints, so kernel development and benchmarking run on a free Colab T4 GPU instead. `CMakeLists.txt` still works as the build description if you have `nvcc` locally — open the notebook for the exact `nvcc` commands used to build the executables and the pybind11 extension in Colab.

## The core technical concept: memory access patterns

Same algorithm, same one-thread-per-sensor mapping, two different on-device data layouts:

- **v1 (sensor-major)**: each sensor's series is contiguous. A thread's own reads are sequential, but at any given timestep, adjacent threads in a warp read addresses `num_timesteps` floats apart — 32 separate memory transactions per warp.
- **v2 (time-major)**: transposed so all sensors' values at a given timestep are contiguous. Adjacent threads read adjacent addresses — the warp's 32 reads land in one coalesced transaction.

Measured on a Colab T4, 2048 timesteps, window=32:

| Sensors | v1 (naive) | v2 (coalesced) | Speedup |
|---|---|---|---|
| 4,096   | 86.3 ms  | 96.2 ms | ~0.9x (overhead-dominated, too small to show the effect) |
| 65,536  | 742.7 ms | 51.1 ms | **~14.5x** |

Validated against a NumPy reference (`python/validate.py`): 0 mismatches on both kernels.

## Build (local, if you have nvcc)

```
cmake -B build -S .
cmake --build build --config Release
```

## Results

CPU (NumPy, cumulative-sum vectorized) vs GPU, 2048 timesteps, window=32:

| Sensors | CPU (ms) | GPU coalesced kernel (ms) | Speedup |
|---|---|---|---|
| 65,536  | 28,099.7 | 51.1 | **~550x** |

(Full sweep across sensor counts in `python/benchmark.py`; CPU side runs locally, GPU side requires the `anomaly_gpu` extension built in a CUDA environment.)
