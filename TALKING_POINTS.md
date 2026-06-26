# Talking points — GPU-Accelerated Sensor Anomaly Detector

One-page reference for walking through this project in an interview.

## The 30-second pitch

Industrial sensor monitoring (vibration, temperature, pressure) generates high-frequency data across thousands of streams. A CPU pipeline can't keep up with real-time anomaly detection at that scale — you drop sampling rate or fall behind. I built a CUDA kernel that computes a rolling z-score anomaly detector across many sensor streams in parallel on the GPU, and measured a 57x-449x speedup over a vectorized NumPy baseline depending on scale, with the GPU-vs-CPU gap growing as sensor count increases — exactly the trend you'd expect from a memory-bound, embarrassingly parallel workload.

## The algorithm

For each sensor's time series independently: at each timestep `t`, compute the mean and standard deviation over the preceding `window` points, then flag `t` as anomalous if `|x[t] - mean| / std > threshold`. One CUDA thread owns one sensor and scans its own series — no cross-thread communication needed, which is why this maps cleanly to the GPU.

## The one deep technical concept: memory access patterns

This is the thing to be ready to explain in detail, because it's the actual engineering decision in this project, not just "I called a CUDA kernel."

**Setup:** same algorithm, same thread-to-sensor mapping, two different on-device data layouts.

- **v1 (sensor-major)**: `data[sensor * num_timesteps + t]`. Each sensor's full series is contiguous. A thread's own reads within its loop are sequential — looks efficient in isolation. But a GPU doesn't execute one thread at a time; a warp of 32 threads executes in lockstep. At any given loop iteration, thread `sensor` and thread `sensor+1` are reading addresses `num_timesteps` floats apart. The warp's 32 simultaneous reads land in 32 different cache lines — 32 separate memory transactions instead of one.
- **v2 (time-major)**: `data[t * num_sensors + sensor]`. Transposed so all sensors' values at a given timestep are adjacent. Now at iteration `t`, the warp's 32 threads read 32 *adjacent* addresses — one 128-byte coalesced transaction services the whole warp.

**Why it matters in numbers (measured, Colab T4, 2048 timesteps, window=32):**

| Sensors | v1 (naive) | v2 (coalesced) | Speedup |
|---|---|---|---|
| 4,096   | 147.5 ms | 7.5 ms  | ~19.6x |
| 65,536  | 746.2 ms | 52.8 ms | ~14.1x |

Same math, same thread mapping, same output (validated bit-for-bit against each other and against a NumPy reference) — the only change is whether the memory controller can batch the warp's reads. That's the whole lesson: on a GPU, *how* you lay out memory often matters more than the arithmetic itself, because these kernels are memory-bandwidth-bound, not compute-bound.

**If asked "why is the host data sensor-major at all if time-major is faster?"** — be honest: it's because that's how I generate/store the synthetic data and how the file format is shared with the NumPy reference for validation. In production, time-major is *also* the more natural ingestion order anyway — sensors report in at each timestamp, so a real ingestion pipeline would naturally accumulate one row per timestep across all sensors. The transpose in `rolling_zscore_coalesced.cu` happens on the host, outside the timed kernel region, specifically so the benchmark isolates the memory-access effect rather than conflating it with transpose cost.

## CPU baseline — built honestly, not as a strawman

The NumPy baseline isn't a naive Python `for` loop — it's vectorized using cumulative sums (`cumsum`) to get O(n) time and memory rather than materializing an O(n × window) tensor. That fix mattered: the first version OOM'd past ~16k sensors because it built a `(sensors, timesteps, window)` array. Worth mentioning if asked about the implementation, because it shows the comparison is GPU-vs-best-effort-CPU, not GPU-vs-strawman.

## Validation

Correctness isn't asserted, it's diffed: `python/generate_data.py` writes one binary file; the CUDA kernel and the NumPy reference both read that exact file; `python/validate.py` does an exact element-wise comparison of the output anomaly flags. 0 mismatches at every scale tested (1,024 to 65,536 sensors). This was deliberate — comparing two separately-generated random datasets would prove nothing.

## Numbers to have ready (Colab T4, 2048 timesteps, window=32)

| Sensors | CPU (NumPy, ms) | GPU kernel (ms) | Speedup |
|---|---|---|---|
| 1,024   | 256.0    | 4.5  | 57x  |
| 4,096   | 1,292.2  | 7.2  | 181x |
| 16,384  | 5,687.0  | 12.7 | 449x |
| 65,536  | 22,719.2 | 55.5 | 409x |

## Architecture, end to end

`cuda/kernels.cu` (kernels + launchers, shared by everything below) →
- `cuda/rolling_zscore.cu` / `rolling_zscore_coalesced.cu`: standalone executables, file-based I/O, used for validation and the executable-level timing comparison
- `cuda/bindings.cpp`: pybind11 extension exposing the same kernels to Python as NumPy-array-in/out functions
- `backend/main.py`: FastAPI endpoints — `/run` (synthetic benchmark: generates data, runs CPU + GPU, returns anomalies and timing) and `/live` (real hardware metrics, see below)
- `python/live_metrics.py`: background collector sampling this machine's actual per-core CPU usage, memory, disk/network throughput, and GPU temp/utilization into ring buffers
- `frontend/`: React + Vite dashboard with two tabs — the synthetic CPU-vs-GPU benchmark, and a live monitor running the same detector on real, continuously-updating hardware metrics

## Why there's a "live monitor" tab, not just a benchmark

A fair pushback on this project as originally scoped: "we already know GPUs are faster at parallel work, so a benchmark proving that in isolation isn't a *useful* artifact, just a demonstration." That's correct, and worth saying out loud if asked rather than oversell the benchmark as a product. The fix wasn't to abandon the benchmark (it's still the right way to prove the *specific* memory-access-pattern mechanism, with a controlled before/after) — it was to add a second, genuinely practical artifact: the exact same rolling z-score detector, unmodified, running continuously on real data from this machine's own hardware sensors (CPU cores, memory, disk, network, GPU). That's something you'd actually glance at, not just a number you compute once and report. Per-CPU-core usage in particular is a direct, real-world instance of "one independent time series per sensor" — same shape of problem as the original industrial-IoT pitch, just running on a laptop instead of a factory floor.

## Honest caveats, if asked

- Local CUDA Toolkit install hit disk-space constraints on my machine, so GPU development happens on a free Colab T4 runtime rather than a local GPU. The kernel code itself is platform-agnostic; this only affected where I iterated.
- The deployed/CV-link version serves pre-generated results rather than running a live GPU on request — that's intentional (a public-facing demo shouldn't depend on a GPU instance being up), and the live-GPU path is what I'd run during an actual interview demo.
- Scope was deliberately narrow: one signal type, one CUDA kernel concept (memory layout), no FFT/multi-sensor-type/live websocket ingestion. That was a conscious choice to go deep on one real concept rather than spread thin across several shallow ones.
