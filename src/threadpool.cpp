#include "threadpool.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define IE_CPU_RELAX() _mm_pause()
#else
#define IE_CPU_RELAX() std::this_thread::yield()
#endif

namespace ie {

ThreadPool::ThreadPool(int num_threads) : num_threads_(std::max(1, num_threads)) {
  for (int i = 1; i < num_threads_; ++i) workers_.emplace_back(&ThreadPool::worker, this, i);
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
    generation_.fetch_add(1);
  }
  wake_.notify_all();
  for (std::thread& t : workers_) t.join();
}

void ThreadPool::run_chunk(int index) {
  // Even split: chunk i is [n*i/T, n*(i+1)/T).
  const int64_t begin = n_ * index / num_threads_;
  const int64_t end = n_ * (index + 1) / num_threads_;
  if (begin < end) (*fn_)(begin, end);
}

void ThreadPool::worker(int index) {
  uint64_t seen = 0;
  while (true) {
    // Wait for a new job: spin briefly (cheap if work is about to arrive), then sleep.
    uint64_t gen = generation_.load(std::memory_order_acquire);
    if (gen == seen) {
      const auto spin_until = std::chrono::steady_clock::now() + std::chrono::microseconds(200);
      while ((gen = generation_.load(std::memory_order_acquire)) == seen &&
             std::chrono::steady_clock::now() < spin_until) {
        for (int i = 0; i < 64; ++i) IE_CPU_RELAX();
      }
      if (gen == seen) {
        std::unique_lock<std::mutex> lock(mu_);
        wake_.wait(lock, [&] { return generation_.load(std::memory_order_acquire) != seen; });
        gen = generation_.load(std::memory_order_acquire);
      }
    }
    seen = gen;
    if (stop_) return;
    run_chunk(index);
    pending_.fetch_sub(1, std::memory_order_acq_rel);
  }
}

void ThreadPool::parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
  if (n <= 0) return;
  if (num_threads_ == 1 || n == 1) {
    fn(0, n);
    return;
  }
  fn_ = &fn;
  n_ = n;
  pending_.store(num_threads_ - 1, std::memory_order_relaxed);
  {
    // Taking the lock orders the bump against a worker that is about to sleep,
    // so no wake-up is lost.
    std::lock_guard<std::mutex> lock(mu_);
    generation_.fetch_add(1, std::memory_order_release);
  }
  wake_.notify_all();
  run_chunk(0);  // the caller does chunk 0 itself
  while (pending_.load(std::memory_order_acquire) != 0) IE_CPU_RELAX();
}

int physical_cores() {
  static const int cores = [] {
    int n = 0;
#ifdef _WIN32
    // One RelationProcessorCore record per physical core.
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<char> buf(len);
    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
    if (len && GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) {
      for (DWORD off = 0; off < len;) {
        auto* rec = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
        if (rec->Relationship == RelationProcessorCore) ++n;
        off += rec->Size;
      }
    }
#else
    // Count distinct (physical id, core id) pairs in /proc/cpuinfo.
    std::ifstream f("/proc/cpuinfo");
    std::set<std::pair<int, int>> seen;
    std::string line;
    int phys = 0, core = -1;
    auto value = [](const std::string& l) { return std::stoi(l.substr(l.find(':') + 1)); };
    while (std::getline(f, line)) {
      if (line.rfind("physical id", 0) == 0) phys = value(line);
      else if (line.rfind("core id", 0) == 0) core = value(line);
      else if (line.empty() && core >= 0) {
        seen.insert({phys, core});
        core = -1;
      }
    }
    if (core >= 0) seen.insert({phys, core});
    n = int(seen.size());
#endif
    const int logical = int(std::max(1u, std::thread::hardware_concurrency()));
    return (n > 0 && n <= logical) ? n : logical;
  }();
  return cores;
}

namespace {
std::unique_ptr<ThreadPool>& pool_slot() {
  static std::unique_ptr<ThreadPool> pool;
  return pool;
}
}  // namespace

ThreadPool& thread_pool() {
  auto& p = pool_slot();
  if (!p) p = std::make_unique<ThreadPool>(physical_cores());
  return *p;
}

void set_num_threads(int n) {
  pool_slot() = std::make_unique<ThreadPool>(n > 0 ? n : physical_cores());
}

}  // namespace ie
