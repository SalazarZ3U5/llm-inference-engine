#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Wrap every CUDA runtime call: CUDA_CHECK(cudaMalloc(...));
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err_ = (call);                                                  \
    if (err_ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), \
                   __FILE__, __LINE__, cudaGetErrorString(err_));               \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// Call after a kernel launch to catch launch-configuration errors immediately.
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())
