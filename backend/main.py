"""FastAPI backend: generate synthetic sensor data, run the CPU baseline and
(if the anomaly_gpu extension is importable) the GPU kernel, and return
anomaly results + timing for both. The GPU path is optional by design — the
deployed CV-link demo serves pre-generated results and never needs a live
GPU; the live-GPU path is for running this on a CUDA-capable machine
(currently: a Colab GPU runtime) during development or an interview demo.
"""
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python"))
from cpu_baseline import rolling_zscore_cpu  # noqa: E402
from live_metrics import MetricBuffers  # noqa: E402

# anomaly_gpu (the pybind11 extension) is built in the repo root.
sys.path.insert(0, str(_REPO_ROOT))

try:
    import anomaly_gpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

app = FastAPI(title="Sensor Anomaly Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes live under /api so they coexist with the static frontend mounted at
# "/" below (and so the frontend's default API_BASE of "/api" works
# unmodified whether it's served by this same process or proxied in dev).
api = APIRouter(prefix="/api")

_live_metrics = MetricBuffers()


@app.on_event("startup")
def _start_live_metrics():
    _live_metrics.start()


LIVE_WINDOW = 10
LIVE_THRESHOLD = 3.0


class RunRequest(BaseModel):
    num_sensors: int = Field(default=4096, gt=0, le=200_000)
    num_timesteps: int = Field(default=2048, gt=0, le=50_000)
    window: int = Field(default=32, gt=0)
    threshold: float = Field(default=3.0, gt=0)
    seed: int = Field(default=42)
    sample_sensor_id: int = Field(default=0, ge=0)


def make_data(num_sensors, num_timesteps, seed):
    rng = np.random.default_rng(seed)
    data = (rng.random((num_sensors, num_timesteps), dtype=np.float32) - 0.5) * 2.0
    for s in range(0, num_sensors, 512):
        data[s, num_timesteps // 2] = 50.0
    return data


@api.get("/health")
def health():
    return {"status": "ok", "gpu_available": GPU_AVAILABLE}


@api.post("/run")
def run(req: RunRequest):
    if req.window >= req.num_timesteps:
        raise HTTPException(status_code=400, detail="window must be smaller than num_timesteps")
    if req.sample_sensor_id >= req.num_sensors:
        raise HTTPException(status_code=400, detail="sample_sensor_id out of range")

    data = make_data(req.num_sensors, req.num_timesteps, req.seed)

    start = time.perf_counter()
    cpu_anomalies = rolling_zscore_cpu(data, req.window, req.threshold)
    cpu_ms = (time.perf_counter() - start) * 1000.0

    result = {
        "num_sensors": req.num_sensors,
        "num_timesteps": req.num_timesteps,
        "window": req.window,
        "threshold": req.threshold,
        "cpu_ms": cpu_ms,
        "cpu_anomaly_count": int(cpu_anomalies.sum()),
        "gpu_available": GPU_AVAILABLE,
        "gpu_ms": None,
        "gpu_anomaly_count": None,
        "speedup_x": None,
    }

    anomalies_for_response = cpu_anomalies

    if GPU_AVAILABLE:
        gpu_anomalies, gpu_kernel_ms = anomaly_gpu.rolling_zscore_coalesced(data, req.window, req.threshold)
        result["gpu_ms"] = gpu_kernel_ms
        result["gpu_anomaly_count"] = int(gpu_anomalies.sum())
        result["speedup_x"] = cpu_ms / gpu_kernel_ms if gpu_kernel_ms > 0 else None
        anomalies_for_response = gpu_anomalies

    flagged_sensors = np.where(anomalies_for_response.any(axis=1))[0]
    result["flagged_sensors"] = [
        {"sensor_id": int(s), "anomaly_count": int(anomalies_for_response[s].sum())}
        for s in flagged_sensors
    ]

    sensor_id = req.sample_sensor_id
    sample_values = data[sensor_id]
    sample_flags = anomalies_for_response[sensor_id]
    result["sample_sensor"] = {
        "sensor_id": sensor_id,
        "values": sample_values.tolist(),
        "anomaly_indices": np.where(sample_flags == 1)[0].tolist(),
    }

    return result


@api.get("/live")
def live():
    """Real hardware metrics from this machine (per-core CPU%, memory, disk/net
    throughput, GPU temp/util if available), with the same rolling z-score
    detector applied to each stream — real continuously-updating data, not
    the synthetic benchmark dataset used by /run."""
    snapshot = _live_metrics.snapshot()
    names = sorted(snapshot.keys())

    if not names:
        return {"metrics": [], "window": LIVE_WINDOW, "threshold": LIVE_THRESHOLD}

    min_len = min(len(snapshot[n]) for n in names)
    if min_len <= LIVE_WINDOW:
        return {
            "metrics": [{"name": n, "values": snapshot[n], "anomaly_indices": []} for n in names],
            "window": LIVE_WINDOW,
            "threshold": LIVE_THRESHOLD,
            "note": "collecting history, not enough samples yet for anomaly detection",
        }

    matrix = np.array([snapshot[n][-min_len:] for n in names], dtype=np.float32)
    anomalies = rolling_zscore_cpu(matrix, LIVE_WINDOW, LIVE_THRESHOLD)

    metrics = []
    for i, name in enumerate(names):
        metrics.append({
            "name": name,
            "values": matrix[i].tolist(),
            "anomaly_indices": np.where(anomalies[i] == 1)[0].tolist(),
        })

    return {"metrics": metrics, "window": LIVE_WINDOW, "threshold": LIVE_THRESHOLD}


app.include_router(api)

# Serve the built frontend (frontend/dist) if present, so the whole app is one
# process behind one port — convenient for a single Cloudflare Tunnel. Must be
# mounted last so it doesn't shadow the /api routes above.
_FRONTEND_DIST = _REPO_ROOT / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
