// Standalone executable for the time-major (coalesced-access) kernel. Kernel +
// launcher live in kernels.cu (shared with rolling_zscore.cu and the pybind11
// bindings); this file is just file I/O. Same input/output file format as
// rolling_zscore.cu, so validate.py is layout-agnostic.
#include "kernels.cuh"
#include <cstdio>
#include <cstdint>
#include <vector>

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
    float kernel_ms = 0.0f;
    launchRollingZScoreCoalesced(h_data.data(), h_anomalies.data(), num_sensors, num_timesteps, window, threshold, &kernel_ms);

    int flagged = 0;
    for (size_t i = 0; i < n; ++i) flagged += h_anomalies[i];

    fprintf(stderr, "Sensors: %d, timesteps: %d, window: %d\n", num_sensors, num_timesteps, window);
    fprintf(stderr, "Kernel time (coalesced): %.3f ms\n", kernel_ms);
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

    return 0;
}
