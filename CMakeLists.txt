cmake_minimum_required(VERSION 3.24)

option(IE_ENABLE_CUDA "Build the CUDA backend" ON)
option(IE_AVX2 "Build the AVX2/FMA CPU kernels (any x86 CPU from ~2013 on)" ON)

if(IE_ENABLE_CUDA)
  # RTX 4060 = Ada Lovelace = compute capability 8.9. Override with -DCMAKE_CUDA_ARCHITECTURES=...
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
  endif()
  project(inference_engine LANGUAGES CXX CUDA)
else()
  project(inference_engine LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # clangd reads build/<preset>/compile_commands.json

if(IE_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # Keep include paths visible in compile_commands.json so clangd can find headers.
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
endif()

# --- dependencies -----------------------------------------------------------
# nlohmann/json v3.11.3 (MIT), vendored as a single header: parses safetensors
# headers, config.json and the server's requests.
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json SYSTEM INTERFACE third_party)
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)

# --- engine library ---------------------------------------------------------
add_library(ie_core
  src/safetensors.cpp
  src/npy.cpp
  src/config.cpp
  src/weights.cpp
  src/ops.cpp
  src/model.cpp
  src/tokenizer.cpp
  src/sampler.cpp
  src/threadpool.cpp
  src/matrix.cpp
  src/runner.cpp
)
target_include_directories(ie_core PUBLIC src)
find_package(Threads REQUIRED)
target_link_libraries(ie_core PUBLIC nlohmann_json::nlohmann_json Threads::Threads)

if(IE_AVX2)
  if(MSVC)
    target_compile_options(ie_core PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
  else()
    target_compile_options(ie_core PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>)
  endif()
endif()

if(IE_ENABLE_CUDA)
  target_sources(ie_core PRIVATE src/cuda/device_check.cu src/cuda/kernels.cu src/cuda/cuda_runner.cu)
  target_compile_definitions(ie_core PUBLIC IE_WITH_CUDA=1)
  # cuBLAS: only for the --cublas comparison; the engine's own kernels need just the runtime.
  target_link_libraries(ie_core PUBLIC CUDA::cudart CUDA::cublas)
  # Line info lets Nsight Compute map hot spots back to your .cu source lines.
  target_compile_options(ie_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()

# Warnings + MSVC niceties
if(MSVC)
  target_compile_options(ie_core PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:/W4 /utf-8 /permissive->
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
else()
  target_compile_options(ie_core PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endif()

# --- HTTP server --------------------------------------------------------------
# cpp-httplib v0.66.0 (MIT), vendored as a single header, built without TLS.
# Only server.cpp includes it (on Windows its winsock2.h must precede windows.h).
add_library(httplib INTERFACE)
target_include_directories(httplib SYSTEM INTERFACE third_party)
target_link_libraries(httplib INTERFACE Threads::Threads $<$<PLATFORM_ID:Windows>:ws2_32>)

# No CUDA code: the server only needs a BatchRunner, so it builds in every configuration.
add_library(ie_server src/server.cpp)
target_link_libraries(ie_server PUBLIC ie_core PRIVATE httplib)

# --- CLI --------------------------------------------------------------------
add_executable(ie src/main.cpp)
target_link_libraries(ie PRIVATE ie_core ie_server)
