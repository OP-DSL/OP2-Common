#pragma once

// The CUDA and HIP backend's part of the run's context (op2/rt/context.h):
// its options, the device's facts, the heuristics' options and the plans its
// loops share.  op::rt::init_gpu makes it once the device is up, from
// src/rt/gpu/op_gpu_init.cpp.

#include <op2/rt/context.h>
#include <op2/rt/gpu/heuristics.h>
#include <op2/rt/gpu/options.h>
#include <op2/rt/plan/cache.h>

namespace op::rt::gpu {

struct ColouringPlan;
struct ChunkEntry;

struct Backend final : op::rt::Backend {
    Options options;
    Device device;
    HeuristicOptions heuristics;
    // The live plans of each kind, which every strategy that would build the
    // same one shares.
    op::rt::plan::SharedPlans<ColouringPlan> colouring_plans;
    op::rt::plan::SharedPlans<ChunkEntry> chunk_plans;

    // Read the variables and the current device's facts.
    Backend();
};

// The GPU part of context, which init_gpu made.
inline Backend& backend(op::rt::Context& context) {
    return static_cast<Backend&>(*context.backend);
}

} // namespace op::rt::gpu
