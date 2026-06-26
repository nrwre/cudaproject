// pybind11 bindings exposing the CUDA kernels to Python as NumPy-array-in,
// NumPy-array-out functions. Compiled together with kernels.cu by nvcc (see
// notebooks/colab_dev.ipynb for the exact build command) so the standalone
// executables and this extension always run identical device code.
#include "kernels.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

static py::tuple run(py::array_t<float, py::array::c_style | py::array::forcecast> data,
                      int window, float threshold, bool coalesced)
{
    if (data.ndim() != 2) {
        throw std::runtime_error("data must be a 2D array shaped (num_sensors, num_timesteps)");
    }

    const int num_sensors = static_cast<int>(data.shape(0));
    const int num_timesteps = static_cast<int>(data.shape(1));

    auto anomalies = py::array_t<unsigned char>({num_sensors, num_timesteps});

    float kernel_ms = 0.0f;
    if (coalesced) {
        launchRollingZScoreCoalesced(data.data(), anomalies.mutable_data(), num_sensors, num_timesteps, window, threshold, &kernel_ms);
    } else {
        launchRollingZScore(data.data(), anomalies.mutable_data(), num_sensors, num_timesteps, window, threshold, &kernel_ms);
    }

    return py::make_tuple(anomalies, kernel_ms);
}

PYBIND11_MODULE(anomaly_gpu, m) {
    m.doc() = "CUDA rolling z-score anomaly detector bindings";

    m.def("rolling_zscore", [](py::array_t<float, py::array::c_style | py::array::forcecast> data, int window, float threshold) {
        return run(data, window, threshold, false);
    }, "Naive sensor-major kernel. Returns (anomalies, kernel_time_ms).",
       py::arg("data"), py::arg("window"), py::arg("threshold"));

    m.def("rolling_zscore_coalesced", [](py::array_t<float, py::array::c_style | py::array::forcecast> data, int window, float threshold) {
        return run(data, window, threshold, true);
    }, "Time-major coalesced-access kernel. Returns (anomalies, kernel_time_ms).",
       py::arg("data"), py::arg("window"), py::arg("threshold"));
}
