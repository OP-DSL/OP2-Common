#pragma once

// The CUDA and HIP backend's run-wide options and the device's facts.  Read
// by init, which the backend's op_init calls once the device is set, and
// which also declares the backend's strategy families.

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

    // OP_JIT_DEBUG: also synchronise after every launch, offline ones too,
    // so a fault names its kernel.
    bool sync_launches = false;
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

const Options& options();
const Device& device();
void init();

} // namespace op::rt::gpu
