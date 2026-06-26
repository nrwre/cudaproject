"""Generate synthetic sensor data and write it in the binary format the CUDA
kernel reads, so the CUDA and NumPy paths run on the exact same input.

Layout: sensor-major, float32, matching cuda/rolling_zscore.cu's InputHeader.
"""
import struct
import sys
import numpy as np

NUM_SENSORS = 4096
NUM_TIMESTEPS = 2048
WINDOW = 32
THRESHOLD = 3.0
SEED = 42
SPIKE_EVERY_N_SENSORS = 512
SPIKE_VALUE = 50.0


def generate(num_sensors=NUM_SENSORS, num_timesteps=NUM_TIMESTEPS,
             window=WINDOW, threshold=THRESHOLD, seed=SEED):
    rng = np.random.default_rng(seed)
    data = (rng.random((num_sensors, num_timesteps), dtype=np.float32) - 0.5) * 2.0

    for s in range(0, num_sensors, SPIKE_EVERY_N_SENSORS):
        data[s, num_timesteps // 2] = SPIKE_VALUE

    return data, window, threshold


def write_input_file(path, data, window, threshold):
    num_sensors, num_timesteps = data.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<iiif", num_sensors, num_timesteps, window, threshold))
        f.write(np.ascontiguousarray(data, dtype=np.float32).tobytes())


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "data.bin"
    data, window, threshold = generate()
    write_input_file(out_path, data, window, threshold)
    print(f"Wrote {out_path}: {data.shape[0]} sensors x {data.shape[1]} timesteps, "
          f"window={window}, threshold={threshold}")
