#include <op2/rt/gpu/heuristics.h>

#include <op2/env.h>

namespace op::rt::gpu {

namespace {

HeuristicOptions options;

} // namespace

const HeuristicOptions& heuristic_options() { return options; }

void init_heuristics() {
    options = {};
    env::parse_number("OP_JIT_MIN_REGISTERS", options.min_registers);
}

loop::Estimate estimate(const KernelFacts& facts, loop::KernelKind kind) {
    if (kind == loop::KernelKind::offline) {
        if (!facts.offline)
            return {.reason = "kernel_unavailable"};
        return {.available = true, .cost = 1.0f};
    }
    if (!facts.jit)
        return {.reason = "kernel_unavailable"};
    bool prior = facts.resources && facts.resources->registers > options.min_registers;
    return {.available = true, .cost = prior ? 0.9f : 1.0f};
}

} // namespace op::rt::gpu
