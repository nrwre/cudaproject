// v2: same rolling z-score detector, but the on-device data layout is transposed
// to time-major (data[t * num_sensors + sensor]) instead of v1's sensor-major.
//
// Why this matters: a warp executes 32 threads in lockstep. In v1 (sensor-major),
// at loop iteration `w`, thread `sensor` reads data[sensor * num_timesteps + w] —
// adjacent threads are num_timesteps floats apart in memory, so the warp's 32 reads
// land in 32 different cache lines (strided / uncoalesced). In time-major layout,
// at iteration `t`, thread `sensor` reads data[t * num_sensors + sensor] — adjacent
// threads read adjacent addresses, so the warp's 32 reads land in one contiguous
// 128-byte segment the memory controller can service in a single transaction
// (coalesced). Same algorithm, same thread-to-sensor mapping — only the memory
// layout changes. Host-side I/O still uses the sensor-major file format (matching
// rolling_zscore.cu) so both kernels can be validated against the same NumPy
// reference and the same input/output files; the transpose happens on the host
// before/after the timed kernel region, so it does not count toward the measured
// kernel time.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

__global__ void rollingZScoreCoalesced(
    const float* data,        // [num_timesteps * num_sensors], time-major
    unsigned char* anomalies, // [num_timesteps * num_sensors], time-major, 1 = anomaly
    int num_sensors,
    int num_timesteps,
    int window,
    float threshold)
{
    int sensor = blockIdx.x * blockDim.x + threadIdx.x;
    if (sensor >= num_sensors) return;

    for (int t = 0; t < num_timesteps; ++t) {
        anomalies[(size_t)t * num_sensors + sensor] = 0;
        if (t < window) continue;

        float sum = 0.0f;
        for (int w = t - window; w < t; ++w) {
            sum += data[(size_t)w * num_sensors + sensor];
        }
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
            if (fabsf(z) > threshold) {
                anomalies[(size_t)t * num_sensors + sensor] = 1;
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

    std::vector<float> h_data_sensor_major(n);
    if (fread(h_data_sensor_major.data(), sizeof(float), n, fin) != n) {
        fprintf(stderr, "Failed to read data\n");
        fclose(fin);
        return 1;
    }
    fclose(fin);

    // Host-side transpose: sensor-major file layout -> time-major device layout.
    std::vector<float> h_data_time_major(n);
    for (int s = 0; s < num_sensors; ++s) {
        for (int t = 0; t < num_timesteps; ++t) {
            h_data_time_major[(size_t)t * num_sensors + s] = h_data_sensor_major[(size_t)s * num_timesteps + t];
        }
    }

    std::vector<unsigned char> h_anomalies_time_major(n);

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

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_anomalies_time_major.data(), d_anomalies, n * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Transpose back to sensor-major for the output file, so validate.py is layout-agnostic.
    std::vector<unsigned char> h_anomalies_sensor_major(n);
    for (int t = 0; t < num_timesteps; ++t) {
        for (int s = 0; s < num_sensors; ++s) {
            h_anomalies_sensor_major[(size_t)s * num_timesteps + t] = h_anomalies_time_major[(size_t)t * num_sensors + s];
        }
    }

    int flagged = 0;
    for (size_t i = 0; i < n; ++i) {
        flagged += h_anomalies_sensor_major[i];
    }

    fprintf(stderr, "Sensors: %d, timesteps: %d, window: %d\n", num_sensors, num_timesteps, window);
    fprintf(stderr, "Kernel time (coalesced): %.3f ms\n", ms);
    fprintf(stderr, "Anomalies flagged: %d\n", flagged);

    FILE* fout = fopen(argv[2], "wb");
    if (!fout) {
        fprintf(stderr, "Failed to open output file: %s\n", argv[2]);
        return 1;
    }
    int32_t out_header[2] = { (int32_t)num_sensors, (int32_t)num_timesteps };
    fwrite(out_header, sizeof(int32_t), 2, fout);
    fwrite(h_anomalies_sensor_major.data(), sizeof(unsigned char), n, fout);
    fclose(fout);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_anomalies);

    return 0;
}
