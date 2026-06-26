// v1: rolling-window mean/std/z-score anomaly detector, one thread per sensor.
//
// Layout note (deliberate, revisited in week 2): data is stored sensor-major
// (each sensor's full time series contiguous: data[sensor * num_timesteps + t]).
// That makes each thread's own reads contiguous, but threads in a warp read far-apart
// addresses at every step -> NOT coalesced across the warp. This naive layout is the
// baseline; the week-2 task is to transpose to time-major layout (data[t * num_sensors + sensor])
// so a warp's simultaneous reads at a given t land in one contiguous segment, and measure
// the throughput difference.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

__global__ void rollingZScore(
    const float* data,      // [num_sensors * num_timesteps], sensor-major
    unsigned char* anomalies, // [num_sensors * num_timesteps], 1 = anomaly
    int num_sensors,
    int num_timesteps,
    int window,
    float threshold)
{
    int sensor = blockIdx.x * blockDim.x + threadIdx.x;
    if (sensor >= num_sensors) return;

    const float* series = data + (size_t)sensor * num_timesteps;
    unsigned char* flags = anomalies + (size_t)sensor * num_timesteps;

    for (int t = 0; t < num_timesteps; ++t) {
        flags[t] = 0;
        if (t < window) continue;

        float sum = 0.0f;
        for (int w = t - window; w < t; ++w) {
            sum += series[w];
        }
        float mean = sum / window;

        float sq_sum = 0.0f;
        for (int w = t - window; w < t; ++w) {
            float diff = series[w] - mean;
            sq_sum += diff * diff;
        }
        float std_dev = sqrtf(sq_sum / window);

        if (std_dev > 1e-6f) {
            float z = (series[t] - mean) / std_dev;
            if (fabsf(z) > threshold) {
                flags[t] = 1;
            }
        }
    }
}

int main() {
    const int num_sensors = 4096;
    const int num_timesteps = 2048;
    const int window = 32;
    const float threshold = 3.0f;

    const size_t n = (size_t)num_sensors * num_timesteps;
    std::vector<float> h_data(n);
    std::vector<unsigned char> h_anomalies(n);

    srand(42);
    for (size_t i = 0; i < n; ++i) {
        float noise = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        h_data[i] = noise;
    }
    // inject a handful of obvious spikes to sanity-check detection
    for (int s = 0; s < num_sensors; s += 512) {
        h_data[(size_t)s * num_timesteps + num_timesteps / 2] = 50.0f;
    }

    float* d_data;
    unsigned char* d_anomalies;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMalloc(&d_anomalies, n * sizeof(unsigned char));
    cudaMemcpy(d_data, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    const int threadsPerBlock = 256;
    const int blocks = (num_sensors + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    rollingZScore<<<blocks, threadsPerBlock>>>(d_data, d_anomalies, num_sensors, num_timesteps, window, threshold);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_anomalies.data(), d_anomalies, n * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    int flagged = 0;
    for (size_t i = 0; i < n; ++i) {
        flagged += h_anomalies[i];
    }

    printf("Sensors: %d, timesteps: %d, window: %d\n", num_sensors, num_timesteps, window);
    printf("Kernel time: %.3f ms\n", ms);
    printf("Anomalies flagged: %d\n", flagged);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_anomalies);

    return 0;
}
