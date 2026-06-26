"""Real hardware metrics collector: samples actual per-core CPU usage, memory,
disk/network throughput, and GPU temp/utilization (via nvidia-smi, if present)
from this machine, into fixed-size ring buffers. This is the "real data" path
— the same rolling z-score detector used on synthetic data in benchmark.py
runs on these genuine, continuously-updating streams instead.

Each ring buffer is one "sensor" in the same sense as the synthetic dataset:
an independent time series checked for anomalies via its own rolling window.
Per-CPU-core usage in particular is a direct real-world analogue of "one
sensor per machine" — just at the scale of cores instead of a factory floor.
"""
import collections
import subprocess
import threading
import time

import psutil

BUFFER_LEN = 240  # ~4 minutes of history at 1 sample/sec


class MetricBuffers:
    def __init__(self, buffer_len=BUFFER_LEN, interval_s=1.0, gpu_poll_interval_s=5.0):
        self.buffer_len = buffer_len
        self.interval_s = interval_s
        self.gpu_poll_interval_s = gpu_poll_interval_s
        self.lock = threading.Lock()
        self.buffers = {}  # name -> deque
        self._names_initialized = False
        self._stop = False
        self._thread = None
        self._gpu_thread = None
        self._gpu_lock = threading.Lock()
        self._gpu_cache = None  # (temp, util) or None until first successful poll
        self._prev_disk = psutil.disk_io_counters()
        self._prev_net = psutil.net_io_counters()
        self._prev_time = time.time()

    def _ensure_buffer(self, name):
        if name not in self.buffers:
            # No zero-padding: a fake jump from 0 to a real value would look
            # like a spurious anomaly. rolling_zscore_cpu already skips the
            # first `window` points of any series, so starting empty and
            # growing is the correct behavior, not a gap to fill.
            self.buffers[name] = collections.deque(maxlen=self.buffer_len)

    def _gpu_metrics(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            )
            if out.returncode != 0:
                return None
            temp_str, util_str = out.stdout.strip().split(",")
            return float(temp_str.strip()), float(util_str.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            return None

    def _sample_once(self):
        now = time.time()
        elapsed = max(now - self._prev_time, 1e-6)

        per_core = psutil.cpu_percent(percpu=True)
        mem_percent = psutil.virtual_memory().percent

        disk = psutil.disk_io_counters()
        disk_read_mbps = (disk.read_bytes - self._prev_disk.read_bytes) / elapsed / 1e6
        disk_write_mbps = (disk.write_bytes - self._prev_disk.write_bytes) / elapsed / 1e6

        net = psutil.net_io_counters()
        net_sent_mbps = (net.bytes_sent - self._prev_net.bytes_sent) / elapsed / 1e6
        net_recv_mbps = (net.bytes_recv - self._prev_net.bytes_recv) / elapsed / 1e6

        self._prev_disk = disk
        self._prev_net = net
        self._prev_time = now

        sample = {}
        for i, pct in enumerate(per_core):
            sample[f"cpu_core_{i}"] = pct
        sample["memory_percent"] = mem_percent
        sample["disk_read_mbps"] = disk_read_mbps
        sample["disk_write_mbps"] = disk_write_mbps
        sample["net_sent_mbps"] = net_sent_mbps
        sample["net_recv_mbps"] = net_recv_mbps

        with self._gpu_lock:
            gpu = self._gpu_cache
        if gpu is not None:
            sample["gpu_temp_c"], sample["gpu_util_percent"] = gpu

        with self.lock:
            for name, value in sample.items():
                self._ensure_buffer(name)
                self.buffers[name].append(value)

    def _run(self):
        psutil.cpu_percent(percpu=True)  # first call primes the internal counter, discard it
        while not self._stop:
            self._sample_once()
            time.sleep(self.interval_s)

    def _run_gpu_poll(self):
        # nvidia-smi subprocess calls can take well over a second on Windows —
        # polling it on the main 1s loop was stretching every sample's actual
        # period to ~1.6s+. Polling it on its own slower thread keeps the main
        # CPU/memory/disk/net sampling at its intended cadence.
        while not self._stop:
            gpu = self._gpu_metrics()
            if gpu is not None:
                with self._gpu_lock:
                    self._gpu_cache = gpu
            time.sleep(self.gpu_poll_interval_s)

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        if self._gpu_thread is None:
            self._gpu_thread = threading.Thread(target=self._run_gpu_poll, daemon=True)
            self._gpu_thread.start()

    def stop(self):
        self._stop = True

    def snapshot(self):
        with self.lock:
            return {name: list(buf) for name, buf in self.buffers.items()}
