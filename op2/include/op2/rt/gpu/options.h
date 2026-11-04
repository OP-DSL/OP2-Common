#pragma once

// The CUDA and HIP backend's run-wide options and the device's facts, which
// its part of the run's context holds (op2/rt/gpu/backend.h).

#include <algorithm>
#include <cstddef>
#include <string>

namespace op::rt::gpu {

struct Options {
    // OP_JIT_MAX_THREADS's default: compiles worth running at once.
    int build_threads = 4;

    // OP_CHUNK_COLOURING_WINDOW: chunks a wavefront colours at a time; 0
    // colours a section at once.
    int window = 4096;
};

struct Device {
    // NVRTC's architecture flag; empty under HIP.
    std::string arch;
    std::size_t shared_bytes = 0;
    int multiprocessors = 0;
    int threads_per_multiprocessor = 0;

    // The most blocks of block_size threads the device holds at once.
    int resident_blocks(int block_size) const {
        return multiprocessors *
               std::max(threads_per_multiprocessor / block_size, 1);
    }
};

} // namespace op::rt::gpu
