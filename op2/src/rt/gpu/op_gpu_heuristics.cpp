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
            return {false, "kernel_unavailable", {}};
        return {true, {}, 1.0f};
    }
    if (!facts.jit)
        return {false, "kernel_unavailable", {}};
    bool prior = facts.resources && facts.resources->registers > options.min_registers;
    return {true, {}, prior ? 0.9f : 1.0f};
}

} // namespace op::rt::gpu
