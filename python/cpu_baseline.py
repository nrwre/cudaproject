"""NumPy/CPU baseline for the rolling z-score anomaly detector.

Same algorithm as cuda/rolling_zscore.cu: for each sensor's time series,
anomaly[t] = 1 if |z| > threshold, where z = (x[t] - mean) / std over the
window immediately preceding t (indices [t-window, t-1]). First `window`
points of each series are never flagged (no full window available yet).

Computation is kept in float32 to match the GPU kernel's precision, so
validation differences are due to summation order, not dtype mismatch.
"""
import struct
import sys
import time
import numpy as np


def rolling_zscore_cpu(data: np.ndarray, window: int, threshold: float) -> np.ndarray:
    data = np.ascontiguousarray(data, dtype=np.float32)
    num_sensors, num_timesteps = data.shape
    anomalies = np.zeros((num_sensors, num_timesteps), dtype=np.uint8)

    if num_timesteps <= window:
        return anomalies

    windows = np.lib.stride_tricks.sliding_window_view(data, window, axis=1)
    windows = windows[:, : num_timesteps - window, :]  # i = t - window, t in [window, num_timesteps-1]

    mean = windows.mean(axis=2, dtype=np.float32)
    diff = windows - mean[:, :, np.newaxis]
    std_dev = np.sqrt((diff * diff).mean(axis=2, dtype=np.float32))

    current = data[:, window:]
    z = np.zeros_like(current)
    valid = std_dev > 1e-6
    z[valid] = (current[valid] - mean[valid]) / std_dev[valid]

    flags = (np.abs(z) > threshold).astype(np.uint8)
    anomalies[:, window:] = flags

    return anomalies


def read_input_file(path):
    with open(path, "rb") as f:
        num_sensors, num_timesteps, window, threshold = struct.unpack("<iiif", f.read(16))
        data = np.frombuffer(f.read(num_sensors * num_timesteps * 4), dtype=np.float32)
        data = data.reshape(num_sensors, num_timesteps)
    return data, window, threshold


def write_output_file(path, anomalies):
    num_sensors, num_timesteps = anomalies.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", num_sensors, num_timesteps))
        f.write(np.ascontiguousarray(anomalies, dtype=np.uint8).tobytes())


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "data.bin"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "cpu_anomalies.bin"

    data, window, threshold = read_input_file(in_path)

    start = time.perf_counter()
    anomalies = rolling_zscore_cpu(data, window, threshold)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    write_output_file(out_path, anomalies)

    print(f"Sensors: {data.shape[0]}, timesteps: {data.shape[1]}, window: {window}", file=sys.stderr)
    print(f"CPU time: {elapsed_ms:.3f} ms", file=sys.stderr)
    print(f"Anomalies flagged: {int(anomalies.sum())}", file=sys.stderr)
