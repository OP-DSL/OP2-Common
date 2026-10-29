#pragma once

// The CUDA and HIP backend's heuristics: what a strategy's kernels are
// expected to cost before anything is timed, from what the backend knows of
// them, as the neutral estimate the ladder orders them by
// (op2/rt/loop/strategy.h).  Every model of what pays on the device belongs
// here; the register rule is the first.  Defined in
// src/rt/gpu/op_gpu_heuristics.cpp.

#include <op2/rt/abi/gpu/descriptor.h>
#include <op2/rt/loop/strategy.h>

#include <optional>

namespace op::rt::gpu {

struct HeuristicOptions {
    // OP_JIT_MIN_REGISTERS: an offline kernel using more registers than this
    // is expected slower than its specialised kernel.
    int min_registers = 32;
};

const HeuristicOptions& heuristic_options();

// Read the heuristics' variables; the backend's init calls it.
void init_heuristics();

// What the backend knows of a strategy's kernels before any call: whether
// its offline kernel can launch the block size, and its resources; whether
// it has a specialised kernel's source.
struct KernelFacts {
    bool offline = false;
    std::optional<abi::gpu::KernelResources> resources;
    bool jit = false;
};

// The kernel of kind: an offline kernel at a nominal 1, and a specialised
// one a tenth cheaper when the offline one uses more than
// OP_JIT_MIN_REGISTERS registers (an arbitrary tenth, until timings
// calibrate it), else as cheap.
loop::Estimate estimate(const KernelFacts& facts, loop::KernelKind kind);

} // namespace op::rt::gpu
