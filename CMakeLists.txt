cmake_minimum_required(VERSION 3.18)
project(sensor_anomaly_cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89) # RTX 4060 (Ada)
endif()

add_executable(vector_add cuda/vector_add.cu)

add_executable(rolling_zscore cuda/rolling_zscore.cu)

add_executable(rolling_zscore_coalesced cuda/rolling_zscore_coalesced.cu)
