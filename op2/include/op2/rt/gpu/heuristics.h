#pragma once

// The CUDA and HIP backend's heuristics: what pays on the device before
// anything is timed, as the defaults the families' axes start from, and as
// what a strategy's kernels are expected to cost, from what the backend
// knows of them, in the neutral estimate the ladder orders them by
// (op2/rt/loop/strategy.h).  Every model of what pays on the device belongs
// here.  Defined in src/rt/gpu/op_gpu_heuristics.cpp.

#include <op2/rt/abi/gpu/descriptor.h>
#include <op2/rt/loop/strategy.h>

#include <vector>

namespace op::rt::gpu {

// What pays by default on the device the library is built for: each axis's
// values, the ladder's first and the tuner's others after it, and which
// chunked families the ladder runs.
struct Defaults {
    std::vector<int> block_sizes;
    std::vector<int> unstaged_singles;
    std::vector<int> wavefront;
    bool chunk_atomics;
    bool chunk_colouring;
};

const Defaults& defaults();

struct HeuristicOptions {
    // OP_JIT_MIN_REGISTERS: an offline kernel using more registers than this
    // is expected slower than its specialised kernel.
    int min_registers = 32;
};

// The options the variables set, which the backend's part of the run's
// context holds (op2/rt/gpu/backend.h).
HeuristicOptions read_heuristic_options();

// What the backend knows of a strategy's kernels before any call: whether
// its offline kernel can launch the block size, and its resources, none
// known when zero; whether it has a specialised kernel's source.
struct KernelFacts {
    bool offline = false;
    abi::gpu::KernelResources resources{};
    bool jit = false;
};

// The kernel of kind: an offline kernel at a nominal 1, and a specialised
// one a tenth cheaper when the offline one uses more than
// OP_JIT_MIN_REGISTERS registers (an arbitrary tenth, until timings
// calibrate it), else as cheap.
loop::Estimate estimate(const HeuristicOptions& options, const KernelFacts& facts,
                        loop::KernelKind kind);

} // namespace op::rt::gpu
