# GPU-Accelerated Sensor Anomaly Detector

CUDA-accelerated rolling z-score anomaly detector across many simulated sensor streams, benchmarked against a CPU/NumPy baseline.

Industrial/IoT sensor monitoring (vibration, temperature, pressure, current draw) generates high-frequency data across thousands of streams. A CPU pipeline can't keep up with real-time anomaly detection at scale — you either drop sampling rate or fall behind. This project proves a GPU pipeline can, with measured numbers.

## Status

Work in progress (week 1 of 4): CUDA toolchain setup + first kernel versions.

## Structure

- `cuda/vector_add.cu` — toolchain sanity check (classic CUDA warm-up)
- `cuda/rolling_zscore.cu` — v1 rolling mean/std/z-score anomaly kernel, one thread per sensor

## Build

Requires CUDA Toolkit (`nvcc`) and CMake >= 3.18.

```
cmake -B build -S .
cmake --build build --config Release
```

## Results

(Throughput numbers and CPU-vs-GPU comparison go here once benchmarked.)
