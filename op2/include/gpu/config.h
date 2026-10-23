#pragma once

// The CUDA and HIP backend's run-wide configuration: the JIT mechanism's
// settings, the device, and the wavefront's window.  Read by init, which the
// backend's op_init calls once the device is set.

#include <algorithm>
#include <cstddef>
#include <string>

namespace op::gpu {

struct Config {
    // OP_JIT_MAX_THREADS's default: compiles worth running at once.
    int build_threads = 4;
    std::string arch;

    // The device.
    std::size_t shared_bytes = 0;
    int multiprocessors = 0;
    int threads_per_multiprocessor = 0;

    // OP_CHUNK_COLOURING_WINDOW: chunks a wavefront colours at a time.
    int window = 0;

    // The most blocks of block_size threads the device holds at once.
    int resident_blocks(int block_size) const {
        return multiprocessors *
               std::max(threads_per_multiprocessor / block_size, 1);
    }
};

const Config& config();
void init();

} // namespace op::gpu
