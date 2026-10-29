#include <op2/rt/tune/heuristics.h>

#include <op2/env.h>

namespace op::rt::tune {

namespace {

HeuristicOptions options;

} // namespace

const HeuristicOptions& heuristic_options() { return options; }

void init_heuristics() {
    options = {};
    env::parse_number("OP_JIT_MIN_REGISTERS", options.min_registers);
}

Estimate estimate(const loop::Facts& facts, loop::KernelKind kind) {
    if (kind == loop::KernelKind::offline) {
        if (!facts.offline.available)
            return {false, {}, facts.offline.reason};
        return {true, 1.0f};
    }
    if (!facts.jit.available)
        return {false, {}, facts.jit.reason};
    bool prior = facts.offline.figures &&
                 facts.offline.figures->registers > options.min_registers;
    return {true, prior ? 0.9f : 1.0f};
}

} // namespace op::rt::tune
