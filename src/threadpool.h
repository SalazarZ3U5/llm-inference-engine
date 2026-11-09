#pragma once
// A small fixed-size thread pool for data-parallel loops.
//
//   pool.parallel_for(n, [&](int64_t begin, int64_t end) { ... });
//
// splits [0, n) into one contiguous chunk per thread, runs them at the same
// time (the calling thread does a chunk too), and returns when all are done.
//
// A forward pass makes ~200 of these calls per token, so start-up latency
// matters. Idle workers spin for a short while before going to sleep, so a
// call that follows soon after the previous one starts in well under a
// microsecond instead of waiting for the OS to wake a thread.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace ie {

class ThreadPool {
 public:
  explicit ThreadPool(int num_threads);  // total threads, including the caller
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn);
  int num_threads() const { return num_threads_; }

 private:
  void worker(int index);
  void run_chunk(int index);

  int num_threads_;
  std::vector<std::thread> workers_;

  // Current job. Published by bumping `generation_`.
  const std::function<void(int64_t, int64_t)>* fn_ = nullptr;
  int64_t n_ = 0;
  std::atomic<uint64_t> generation_{0};
  std::atomic<int> pending_{0};  // workers still running the current job
  std::atomic<bool> stop_{false};

  std::mutex mu_;
  std::condition_variable wake_;
};

// Physical CPU cores (not counting hyper-threads / SMT siblings). Falls back to
// the hardware thread count if it can't be determined.
int physical_cores();

// The pool used by the kernels. Defaults to one thread per physical core:
// decode makes ~200 small parallel calls per token, and each waits for its
// slowest thread. Using every hyper-thread leaves the OS no spare CPU, so one
// thread occasionally gets paused and every call waits for it.
ThreadPool& thread_pool();
void set_num_threads(int n);  // recreates the pool; n <= 0 = default. Call before running the model.

}  // namespace ie
