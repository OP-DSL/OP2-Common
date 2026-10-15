#pragma once

// The CUDA and HIP backend's run-wide configuration: the JIT policy, the
// block sizes, the device, and the chunked strategies' kernel variants.
// Read by init, which the backend's op_init calls once the device is set.

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace op::gpu {

struct Config {
    bool enable = true;         // OP_JIT_ENABLE
    bool seq_compile = false;   // OP_JIT_SEQ_COMPILE: compile on the calling thread
    bool debug = false;         // OP_JIT_DEBUG: print compiles, synchronise launches
    bool force = false;         // OP_JIT_FORCE: specialise every kernel
    // OP_JIT_MAX_THREADS: compiles in flight at once.
    int max_threads = 4;
    // OP_JIT_MIN_CALLS: calls of a configuration before its compile starts,
    // so parameters that change every call never compile.
    int min_calls = 9;
    // OP_JIT_MIN_REGISTERS: an offline kernel using more registers than this
    // is specialised; the prior for when a compile pays.
    int min_registers = 32;
    // OP_AUTOTUNE_BLOCK_SIZES: block sizes registered as variants of every
    // strategy, for the tuner to explore.
    std::vector<int> block_sizes;
    std::string arch;

    // The device.
    std::size_t shared_bytes = 0;
    int multiprocessors = 0;
    int threads_per_multiprocessor = 0;
    int default_block_size = 128;

    // OP_CHUNK_ATOMICS_UNSTAGED_SINGLES: a chunk's only reference to a
    // target increments global memory directly, as staging it would save no
    // global atomic.
    bool unstaged_singles = false;
    // OP_CHUNK_COLOURING_WAVEFRONT: one launch per section, its chunks taken
    // in an order that keeps them after every earlier chunk sharing a
    // target; OP_CHUNK_COLOURING_WINDOW chunks are coloured at a time.
    bool wavefront = false;
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
