#include "cuda/device_check.h"

#include <cmath>
#include <cstdio>
#include <vector>

#include "cuda/cuda_check.cuh"

namespace ie::cuda {

namespace {

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

// Simple grid-stride copy. Real decode is bound by how fast weights stream
// from VRAM, so this number is your tokens/sec ceiling (bandwidth / model bytes).
__global__ void copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    dst[i] = src[i];
}

void print_properties() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  size_t free_b = 0, total_b = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
  int rt = 0, drv = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&rt));
  CUDA_CHECK(cudaDriverGetVersion(&drv));

  std::printf("GPU               : %s\n", p.name);
  std::printf("Compute capability: %d.%d\n", p.major, p.minor);
  std::printf("SMs               : %d\n", p.multiProcessorCount);
  std::printf("VRAM              : %.2f GiB total, %.2f GiB free\n", total_b / 1073741824.0,
              free_b / 1073741824.0);
  std::printf("L2 cache          : %d KiB\n", p.l2CacheSize / 1024);
  std::printf("Shared mem/block  : %zu KiB (opt-in max %zu KiB)\n", p.sharedMemPerBlock / 1024,
              p.sharedMemPerBlockOptin / 1024);
  std::printf("CUDA runtime      : %d.%d, driver supports %d.%d\n\n", rt / 1000, (rt % 1000) / 10,
              drv / 1000, (drv % 1000) / 10);
}

bool test_vector_add() {
  const int n = 1 << 20;
  std::vector<float> a(n), b(n), c(n);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = 2.0f * i;
  }
  float *da, *db, *dc;
  CUDA_CHECK(cudaMalloc(&da, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&db, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dc, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  const int block = 256;
  vector_add<<<(n + block - 1) / block, block>>>(da, db, dc, n);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(c.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));

  int bad = 0;
  for (int i = 0; i < n; ++i)
    if (std::fabs(c[i] - 3.0f * i) > 1e-3f) ++bad;
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dc));
  std::printf("vector_add        : %s (%d mismatches)\n", bad == 0 ? "PASS" : "FAIL", bad);
  return bad == 0;
}

void test_bandwidth() {
  const size_t bytes = size_t(512) << 20;  // 512 MiB
  const size_t n4 = bytes / sizeof(float4);
  float4 *src, *dst;
  CUDA_CHECK(cudaMalloc(&src, bytes));
  CUDA_CHECK(cudaMalloc(&dst, bytes));
  CUDA_CHECK(cudaMemset(src, 0, bytes));

  int sms = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  const int block = 256, grid = sms * 8;
  copy_kernel<<<grid, block>>>(src, dst, n4);  // warm-up
  CUDA_CHECK_LAST();

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  const int iters = 20;
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) copy_kernel<<<grid, block>>>(src, dst, n4);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  // Each copy reads and writes `bytes`.
  const double gbps = (2.0 * bytes * iters) / (ms / 1e3) / 1e9;
  std::printf("D2D bandwidth     : %.1f GB/s (read+write)\n", gbps);
  std::printf("  -> decode ceiling for a 1 GB (0.5B params, BF16) model ~ %.0f tokens/s\n", gbps);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(src));
  CUDA_CHECK(cudaFree(dst));
}

}  // namespace

int run_device_check() {
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  if (count == 0) {
    std::printf("No CUDA devices found.\n");
    return 1;
  }
  print_properties();
  bool ok = test_vector_add();
  test_bandwidth();
  return ok ? 0 : 1;
}

}  // namespace ie::cuda
