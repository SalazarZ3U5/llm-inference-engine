#pragma once

namespace ie::cuda {

// Prints GPU properties, runs a vector-add correctness test and a
// device-to-device bandwidth test. Returns 0 on success.
int run_device_check();

}  // namespace ie::cuda
