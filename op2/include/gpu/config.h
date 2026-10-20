#pragma once

// The CUDA and HIP backend's run-wide configuration: the JIT policy, the
// device, the wavefront's window and lazy execution.  Read by init, which the backend's
// op_init calls once the device is set.

#include <algorithm>
#include <cstddef>
#include <string>

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
    std::string arch;

    // The device.
    std::size_t shared_bytes = 0;
    int multiprocessors = 0;
    int threads_per_multiprocessor = 0;
    int blocks_per_multiprocessor = 0;
    int registers_per_multiprocessor = 0;
    int warp_size = 0;

    // OP_CHUNK_COLOURING_WINDOW: chunks a wavefront colours at a time.
    int window = 0;

    // OP_LAZY: queue calls and run them when the host needs their results.
    bool lazy = true;
    // OP_LAZY_MAX_CALLS: the queue runs once it holds this many calls.
    int lazy_max_calls = 64;
    // OP_LAZY_COPY_BYTES: a READ global up to this size is copied into its
    // queued call; a larger one runs the queue at once.
    int lazy_copy_bytes = 4096;
    // OP_FUSION: run queued calls that may share a kernel as one.
    bool fusion = true;
    // OP_FUSION_MAX_REGISTERS: a call whose kernel uses more registers is
    // never fused; 0 for no limit.
    int fusion_max_registers = 64;
    // OP_FUSION_TUNE: time a fused kernel against its calls run apart and
    // keep the faster.
    bool fusion_tune = true;

    // The most blocks of block_size threads the device holds at once.
    int resident_blocks(int block_size) const {
        return multiprocessors *
               std::max(threads_per_multiprocessor / block_size, 1);
    }

    // Blocks of block_size threads a multiprocessor holds when each thread
    // uses registers, allocated per warp in units of 256.
    int occupancy(int registers, int block_size) const {
        int warps = (block_size + warp_size - 1) / warp_size;
        int per_warp = (registers * warp_size + 255) / 256 * 256;
        int by_registers = per_warp > 0 ? registers_per_multiprocessor / (per_warp * warps)
                                        : blocks_per_multiprocessor;
        return std::min({by_registers, threads_per_multiprocessor / block_size,
                         blocks_per_multiprocessor});
    }
};

const Config& config();
void init();

} // namespace op::gpu
