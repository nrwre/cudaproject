#pragma once

// Both launchers take sensor-major host arrays in and out, so callers (the
// standalone executables and the pybind11 bindings) share one API regardless
// of which on-device layout the kernel actually uses internally.

void launchRollingZScore(
    const float* h_data, unsigned char* h_anomalies,
    int num_sensors, int num_timesteps, int window, float threshold,
    float* out_kernel_ms);

void launchRollingZScoreCoalesced(
    const float* h_data, unsigned char* h_anomalies,
    int num_sensors, int num_timesteps, int window, float threshold,
    float* out_kernel_ms);
