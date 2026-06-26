# Project: GPU-Accelerated Sensor Anomaly Detector (CUDA)

## Who this is for
Dhruv — B.Tech Instrumentation Engineering, IIT Kharagpur (expected 2028). Preparing for CDC internship season, targeting two resume tracks: data engineering/analytics and software engineering. No standalone technical projects on CV yet. Comfortable with C/C++. Beginner-level Python/SQL. Has a local NVIDIA GPU.

Deadline: ~1 month, hard.

## Why this project (the actual pitch)
Industrial/IoT sensors (vibration, temperature, pressure, current draw) generate high-frequency data, and predictive maintenance depends on catching anomalies before failure. At scale (thousands of sensor streams), a CPU pipeline can't keep up in real time — you either drop sampling rate (miss the anomaly) or fall behind (detect it too late). This is a real, named industry problem.

This project is a CUDA-accelerated pipeline that computes a rolling z-score anomaly detector across many simulated sensor streams in parallel on the GPU, and proves — with real measured numbers — that it's faster than the equivalent CPU/NumPy version. The CPU-vs-GPU throughput gap *is* the demo, not a visualization for its own sake.

Why it's differentiated: combines the Instrumentation Engineering background with a real GPU systems project. Not a generic "I learned CUDA" toy — it's domain-grounded infra work. Straddles both target tracks: real-time data pipeline (data eng) + performance/systems work (SWE).

## Explicit scope decisions (read before expanding scope)
These cuts were deliberate — do not re-add them without good reason:
- **No FFT / vibration signature extraction.** One signal type only, simple rolling stats. DSP is a rabbit hole.
- **No multi-sensor-type simulation** (vibration + temp + pressure together). Just one signal type for v1.
- **No live WebSocket real-time ingestion for v1.** Batch processing first. Live streaming is a stretch goal only if week 4 has slack.
- **One CUDA kernel concept, not three.** Rolling window mean/std → z-score anomaly flag. That's the whole algorithmic core. Resist the urge to add shared-memory tiling, FFT, histograms, etc. unless the core pipeline is done early.
- The "interview-level" CUDA concept to actually go deep on: **memory access patterns** (are sensor streams laid out so threads read contiguous memory, or do they jump around — coalesced vs strided access). This is the one concept worth mastering deeply for this project rather than spreading thin.

## Tech stack
- CUDA C++ (`nvcc`) — kernel comfortable to write since C++ background exists, no need to learn syntax from scratch
- pybind11 to wrap the CUDA kernel for Python (preferred over ctypes given C++ comfort)
- Python (NumPy/pandas) for the CPU baseline comparison
- FastAPI backend — endpoint(s) to generate/upload sensor data, run both CPU and GPU versions, return anomalies + timing
- React + Vite frontend — signal chart with anomalies marked, flagged-sensor table, CPU vs GPU timing comparison
- Deployment: host a version with pre-generated data/results for the CV link (live GPU version runs locally for interview demos)

## 4-week plan

**Week 1 — CUDA mechanics + first real kernel**
- CUDA-specific mental model: `__global__`, thread/block/grid indexing, `cudaMalloc`/`cudaMemcpy`, kernel launch syntax, timing with CUDA Events
- Vector addition kernel as a toolchain sanity check (classic CUDA warm-up)
- First working version of the rolling mean/std/z-score kernel — one thread per sensor, each thread scans its own time series window

**Week 2 — Correctness + the one optimization that matters**
- Validate kernel output numerically against a NumPy reference, sensor by sensor, until they match
- Examine and fix memory access pattern: ensure sensor data layout lets threads read contiguous memory (coalesced access) rather than strided/scattered access
- This is the single deep technical concept for this project — understand and be able to explain *why* the layout matters

**Week 3 — CPU baseline, Python wrapper, backend**
- NumPy/pandas implementation of the same rolling z-score logic (CPU baseline for comparison)
- Wrap the CUDA kernel with pybind11 so Python can call it directly
- FastAPI endpoint(s): generate or upload sensor data → run both CPU and GPU versions → return anomaly results + timing for both

**Week 4 — Dashboard, deploy, writeup**
- Frontend: chart of a sensor's signal with anomalies highlighted, table of flagged sensors, CPU vs GPU timing bar
- Deploy a version with pre-generated data/results for the CV link (since the live version needs a real local GPU)
- README with actual measured numbers (e.g. "Nx faster than CPU baseline at M sensor streams")
- One-page "talking points" note: be ready to explain memory hierarchy, coalescing, why this layout choice mattered, and walk through the project end-to-end if asked in an interview

## Context for whoever picks this up in Claude Code
- Don't suggest re-adding cut scope (FFT, multi-sensor types, live websockets, multiple kernel concepts) unless core pipeline (weeks 1-3) is done with time to spare.
- Don't build a "teaching visualizer" — earlier iteration of this idea was rejected for being a learning tool rather than proof of building ability. The dashboard's job is to show real measured throughput/timing numbers, not to animate concepts for their own sake.
- Other portfolio/CV context (for consistency, not necessarily relevant to this specific repo): Dhruv is also building a collaborative YouTube listening app ("Spotify Jam alternative" — React + Vite, Node.js + Express + Socket.IO, YouTube Data API v3) as a separate CDC portfolio project with its own deployable demo link.
