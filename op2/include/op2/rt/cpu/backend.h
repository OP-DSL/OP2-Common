#pragma once

// The OpenMP backend's part of the run's context (op2/rt/context.h): the
// plans its loops share.  op::rt::init_cpu makes it, from
// src/rt/cpu/op_cpu_init.cpp.

#include <op2/rt/context.h>
#include <op2/rt/cpu/chunk_plan.h>
#include <op2/rt/plan/cache.h>
#include <op2/rt/plan/colouring.h>

namespace op::rt::cpu {

struct Backend final : op::rt::Backend {
    // The live plans of each kind, which every strategy that would build the
    // same one shares.
    op::rt::plan::SharedPlans<op::rt::plan::SetColouring> colouring_plans;
    op::rt::plan::SharedPlans<ChunkResult> chunk_plans;
};

// The CPU part of context, which init_cpu made.
inline Backend& backend(op::rt::Context& context) {
    return static_cast<Backend&>(*context.backend);
}

} // namespace op::rt::cpu
