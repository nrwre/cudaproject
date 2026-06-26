"""Compare the CUDA kernel's anomaly output against the NumPy baseline,
run on the identical input file. Both must read the same data.bin.

Usage: python validate.py <data.bin> <cpu_anomalies.bin> <gpu_anomalies.bin>
"""
import struct
import sys
import numpy as np


def read_anomaly_file(path):
    with open(path, "rb") as f:
        num_sensors, num_timesteps = struct.unpack("<ii", f.read(8))
        flags = np.frombuffer(f.read(num_sensors * num_timesteps), dtype=np.uint8)
        flags = flags.reshape(num_sensors, num_timesteps)
    return flags


def main():
    if len(sys.argv) != 4:
        print("Usage: python validate.py <data.bin> <cpu_anomalies.bin> <gpu_anomalies.bin>", file=sys.stderr)
        sys.exit(1)

    data_path, cpu_path, gpu_path = sys.argv[1], sys.argv[2], sys.argv[3]

    cpu_flags = read_anomaly_file(cpu_path)
    gpu_flags = read_anomaly_file(gpu_path)

    if cpu_flags.shape != gpu_flags.shape:
        print(f"Shape mismatch: CPU {cpu_flags.shape} vs GPU {gpu_flags.shape}")
        sys.exit(1)

    mismatch_mask = cpu_flags != gpu_flags
    num_mismatches = int(mismatch_mask.sum())
    total = cpu_flags.size

    print(f"Total points: {total}")
    print(f"CPU flagged: {int(cpu_flags.sum())}")
    print(f"GPU flagged: {int(gpu_flags.sum())}")
    print(f"Mismatches: {num_mismatches} ({100.0 * num_mismatches / total:.6f}%)")

    if num_mismatches > 0:
        sensor_idx, t_idx = np.where(mismatch_mask)
        print("First few mismatches (sensor, t, cpu_flag, gpu_flag):")
        for i in range(min(10, num_mismatches)):
            s, t = sensor_idx[i], t_idx[i]
            print(f"  ({s}, {t}): cpu={cpu_flags[s, t]}, gpu={gpu_flags[s, t]}")

    sys.exit(0 if num_mismatches == 0 else 2)


if __name__ == "__main__":
    main()
