// v1: rolling-window mean/std/z-score anomaly detector, one thread per sensor.
//
// Layout note (deliberate, revisited in week 2): data is stored sensor-major
// (each sensor's full time series contiguous: data[sensor * num_timesteps + t]).
// That makes each thread's own reads contiguous, but threads in a warp read far-apart
// addresses at every step -> NOT coalesced across the warp. This naive layout is the
// baseline; the week-2 task is to transpose to time-major layout (data[t * num_sensors + sensor])
// so a warp's simultaneous reads at a given t land in one contiguous segment, and measure
// the throughput difference.
//
// I/O format (so CUDA and the NumPy reference run on identical data, for validation):
// input file header: int32 num_sensors, int32 num_timesteps, int32 window, float32 threshold
//   followed by num_sensors*num_timesteps float32, sensor-major.
// output file header: int32 num_sensors, int32 num_timesteps
//   followed by num_sensors*num_timesteps uint8 (1 = anomaly).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
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

struct InputHeader {
    int32_t num_sensors;
    int32_t num_timesteps;
    int32_t window;
    float threshold;
};

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }

    FILE* fin = fopen(argv[1], "rb");
    if (!fin) {
        fprintf(stderr, "Failed to open input file: %s\n", argv[1]);
        return 1;
    }

    InputHeader header;
    if (fread(&header, sizeof(header), 1, fin) != 1) {
        fprintf(stderr, "Failed to read header\n");
        fclose(fin);
        return 1;
    }

    const int num_sensors = header.num_sensors;
    const int num_timesteps = header.num_timesteps;
    const int window = header.window;
    const float threshold = header.threshold;
    const size_t n = (size_t)num_sensors * num_timesteps;

    std::vector<float> h_data(n);
    if (fread(h_data.data(), sizeof(float), n, fin) != n) {
        fprintf(stderr, "Failed to read data\n");
        fclose(fin);
        return 1;
    }
    fclose(fin);

    std::vector<unsigned char> h_anomalies(n);

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

    fprintf(stderr, "Sensors: %d, timesteps: %d, window: %d\n", num_sensors, num_timesteps, window);
    fprintf(stderr, "Kernel time: %.3f ms\n", ms);
    fprintf(stderr, "Anomalies flagged: %d\n", flagged);

    FILE* fout = fopen(argv[2], "wb");
    if (!fout) {
        fprintf(stderr, "Failed to open output file: %s\n", argv[2]);
        return 1;
    }
    int32_t out_header[2] = { (int32_t)num_sensors, (int32_t)num_timesteps };
    fwrite(out_header, sizeof(int32_t), 2, fout);
    fwrite(h_anomalies.data(), sizeof(unsigned char), n, fout);
    fclose(fout);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_anomalies);

    return 0;
}
