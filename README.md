# GPU-Accelerated Sensor Anomaly Detector

CUDA-accelerated rolling z-score anomaly detector across many simulated sensor streams, benchmarked against a CPU/NumPy baseline.

Industrial/IoT sensor monitoring (vibration, temperature, pressure, current draw) generates high-frequency data across thousands of streams. A CPU pipeline can't keep up with real-time anomaly detection at scale — you either drop sampling rate or fall behind. This project proves a GPU pipeline can, with measured numbers.

## Status

Week 4 of 4 in progress: dashboard built, demo deployable via Cloudflare Tunnel.

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
- `backend/main.py` — FastAPI endpoint: generate data, run CPU + GPU, return anomalies + timing; also serves the built frontend (`frontend/dist`) as static files when present, so the whole app runs behind one port
- `frontend/` — React + Vite dashboard (signal chart, flagged-sensor table, CPU/GPU timing bar)
- `notebooks/colab_dev.ipynb` — GPU dev/build/validate/benchmark workflow
- `deploy/start_demo.ps1` — builds the frontend, starts the backend, and opens a Cloudflare quick tunnel

## Dev environment

Local CUDA Toolkit install hit disk-space constraints, so kernel development and benchmarking run on a free Colab T4 GPU instead. `CMakeLists.txt` still works as the build description if you have `nvcc` locally — open the notebook for the exact `nvcc` commands used to build the executables and the pybind11 extension in Colab.

## The core technical concept: memory access patterns

Same algorithm, same one-thread-per-sensor mapping, two different on-device data layouts:

- **v1 (sensor-major)**: each sensor's series is contiguous. A thread's own reads are sequential, but at any given timestep, adjacent threads in a warp read addresses `num_timesteps` floats apart — 32 separate memory transactions per warp.
- **v2 (time-major)**: transposed so all sensors' values at a given timestep are contiguous. Adjacent threads read adjacent addresses — the warp's 32 reads land in one coalesced transaction.

Measured on a Colab T4, 2048 timesteps, window=32 (standalone executables, kernel time only):

| Sensors | v1 (naive) | v2 (coalesced) | Speedup |
|---|---|---|---|
| 4,096   | 147.5 ms | 7.5 ms  | **~19.6x** |
| 65,536  | 746.2 ms | 52.8 ms | **~14.1x** |

Same algorithm, same thread-to-sensor mapping — the only difference is whether a warp's simultaneous reads land in one memory transaction or 32. Validated against a NumPy reference (`python/validate.py`): 0 mismatches on both kernels, at every scale tested.

## Build (local, if you have nvcc)

```
cmake -B build -S .
cmake --build build --config Release
```

## Results

CPU (NumPy, cumulative-sum vectorized) vs GPU (coalesced kernel via the pybind11 extension), 2048 timesteps, window=32, on a Colab T4 — full sweep from `python/benchmark.py`, every row validated against the CPU output (0 mismatches):

| Sensors | CPU (ms) | GPU kernel (ms) | Speedup | Anomalies flagged (CPU == GPU) |
|---|---|---|---|---|
| 1,024   | 256.0    | 4.5  | **57x**  | 30 |
| 4,096   | 1,292.2  | 7.2  | **181x** | 132 |
| 16,384  | 5,687.0  | 12.7 | **449x** | 525 |
| 65,536  | 22,719.2 | 55.5 | **409x** | 2,289 |

The speedup grows with scale because the kernel is memory-bound and the GPU has far more memory bandwidth and parallelism to exploit — the CPU baseline is itself vectorized (not a naive Python loop), so this isn't a "Python is slow" trick, it's NumPy's best vectorized version against the GPU's.

## Running the live demo

This serves CPU-only results (no local CUDA toolkit on the demo machine) behind a [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/), with the demo machine acting as the server:

```
powershell -ExecutionPolicy Bypass -File deploy\start_demo.ps1
```

This builds the frontend if needed, starts the FastAPI backend on port 8010 (serving both the API and the built frontend), and opens a free Cloudflare quick tunnel. The printed `*.trycloudflare.com` URL is public but **only live while this is running** — it's a tunnel to this machine, not hosted infrastructure, and the URL changes on every restart (no Cloudflare account / owned domain needed for this mode). For the live GPU path, run `notebooks/colab_dev.ipynb` instead.

### Getting the link after every restart (sleep, reboot, closed terminal, etc.)

Every time you sleep/reboot this machine or close the terminal, both the backend and the tunnel stop — the old URL goes dead. To get a new one, just re-run the same command and watch the terminal output for a new block like this:

```
powershell -ExecutionPolicy Bypass -File deploy\start_demo.ps1

...
Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):
https://some-random-words.trycloudflare.com
...
```

That `https://...trycloudflare.com` line is the new public link — copy it and share it. If Windows shows a firewall prompt for Python or `cloudflared` asking about public/private network access, click **Allow** — declining it breaks the tunnel or the backend's ability to accept connections.
