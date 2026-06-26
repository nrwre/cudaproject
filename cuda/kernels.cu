// Kernel definitions + host launcher functions, shared by the standalone
// executables (cuda/rolling_zscore.cu, cuda/rolling_zscore_coalesced.cu) and
// the pybind11 bindings (cuda/bindings.cpp). Keeping the launch/malloc/copy/
// timing logic in one place means the executables and the Python extension
// are always exercising identical device code, not two implementations that
// could silently drift apart.
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

// v1: sensor-major layout. Adjacent threads (adjacent sensors) read addresses
// num_timesteps floats apart at every step -> uncoalesced warp reads.
__global__ void rollingZScore(
    const float* data, unsigned char* anomalies,
    int num_sensors, int num_timesteps, int window, float threshold)
{
    int sensor = blockIdx.x * blockDim.x + threadIdx.x;
    if (sensor >= num_sensors) return;

    const float* series = data + (size_t)sensor * num_timesteps;
    unsigned char* flags = anomalies + (size_t)sensor * num_timesteps;

    for (int t = 0; t < num_timesteps; ++t) {
        flags[t] = 0;
        if (t < window) continue;

        float sum = 0.0f;
        for (int w = t - window; w < t; ++w) sum += series[w];
        float mean = sum / window;

        float sq_sum = 0.0f;
        for (int w = t - window; w < t; ++w) {
            float diff = series[w] - mean;
            sq_sum += diff * diff;
        }
        float std_dev = sqrtf(sq_sum / window);

        if (std_dev > 1e-6f) {
            float z = (series[t] - mean) / std_dev;
            if (fabsf(z) > threshold) flags[t] = 1;
        }
    }
}

// v2: time-major layout. Adjacent threads read adjacent addresses at every
// step -> coalesced warp reads. Same algorithm, same thread-to-sensor mapping.
__global__ void rollingZScoreCoalesced(
    const float* data, unsigned char* anomalies,
    int num_sensors, int num_timesteps, int window, float threshold)
{
    int sensor = blockIdx.x * blockDim.x + threadIdx.x;
    if (sensor >= num_sensors) return;

    for (int t = 0; t < num_timesteps; ++t) {
        anomalies[(size_t)t * num_sensors + sensor] = 0;
        if (t < window) continue;

        float sum = 0.0f;
        for (int w = t - window; w < t; ++w) sum += data[(size_t)w * num_sensors + sensor];
        float mean = sum / window;

        float sq_sum = 0.0f;
        for (int w = t - window; w < t; ++w) {
            float diff = data[(size_t)w * num_sensors + sensor] - mean;
            sq_sum += diff * diff;
        }
        float std_dev = sqrtf(sq_sum / window);

        if (std_dev > 1e-6f) {
            float x = data[(size_t)t * num_sensors + sensor];
            float z = (x - mean) / std_dev;
            if (fabsf(z) > threshold) anomalies[(size_t)t * num_sensors + sensor] = 1;
        }
    }
}

void launchRollingZScore(
    const float* h_data, unsigned char* h_anomalies,
    int num_sensors, int num_timesteps, int window, float threshold,
    float* out_kernel_ms)
{
    const size_t n = (size_t)num_sensors * num_timesteps;

    float* d_data;
    unsigned char* d_anomalies;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMalloc(&d_anomalies, n * sizeof(unsigned char));
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

    const int threadsPerBlock = 256;
    const int blocks = (num_sensors + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    rollingZScore<<<blocks, threadsPerBlock>>>(d_data, d_anomalies, num_sensors, num_timesteps, window, threshold);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    if (out_kernel_ms) cudaEventElapsedTime(out_kernel_ms, start, stop);

    cudaMemcpy(h_anomalies, d_anomalies, n * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_anomalies);
}

void launchRollingZScoreCoalesced(
    const float* h_data, unsigned char* h_anomalies,
    int num_sensors, int num_timesteps, int window, float threshold,
    float* out_kernel_ms)
{
    const size_t n = (size_t)num_sensors * num_timesteps;

    // Host-side transpose: sensor-major API -> time-major device layout.
    std::vector<float> h_data_time_major(n);
    for (int s = 0; s < num_sensors; ++s)
        for (int t = 0; t < num_timesteps; ++t)
            h_data_time_major[(size_t)t * num_sensors + s] = h_data[(size_t)s * num_timesteps + t];

    float* d_data;
    unsigned char* d_anomalies;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMalloc(&d_anomalies, n * sizeof(unsigned char));
    cudaMemcpy(d_data, h_data_time_major.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    const int threadsPerBlock = 256;
    const int blocks = (num_sensors + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    rollingZScoreCoalesced<<<blocks, threadsPerBlock>>>(d_data, d_anomalies, num_sensors, num_timesteps, window, threshold);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    if (out_kernel_ms) cudaEventElapsedTime(out_kernel_ms, start, stop);

    std::vector<unsigned char> h_anomalies_time_major(n);
    cudaMemcpy(h_anomalies_time_major.data(), d_anomalies, n * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Transpose back to sensor-major so the API matches launchRollingZScore.
    for (int t = 0; t < num_timesteps; ++t)
        for (int s = 0; s < num_sensors; ++s)
            h_anomalies[(size_t)s * num_timesteps + t] = h_anomalies_time_major[(size_t)t * num_sensors + s];

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_anomalies);
}
