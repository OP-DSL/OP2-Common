#pragma once

// The heuristics: what is expected to run fastest before anything is timed.
// The ladder orders a strategy's candidates by them, and the tuner's
// timings overrule them once a configuration is explored
// (op2/rt/tune/autotune.h).  Every model of what pays belongs here, over the
// facts the backends report (op2/rt/loop/strategy.h); the register rule is
// the first.

#include <op2/rt/loop/strategy.h>

#include <optional>
#include <string_view>

namespace op::rt::tune {

struct HeuristicOptions {
    // OP_JIT_MIN_REGISTERS: an offline kernel using more registers than this
    // is expected slower than its specialised kernel.
    int min_registers = 32;
};

const HeuristicOptions& heuristic_options();

// Read the heuristics' variables; op_init calls it through op::rt::init.
void init_heuristics();

// A heuristic's estimate of one kind of a strategy's kernel: whether the
// strategy has it, and its expected time in units comparable only with the
// loop's other estimates, none when no heuristic can say; when it has none,
// why ("kernel_unavailable").
struct Estimate {
    bool available = false;
    std::optional<float> time;
    std::string_view reason{};
};

// The kernel of kind from a strategy's facts: an offline kernel at a nominal
// 1, and a specialised one expected a tenth faster when the offline one uses
// more than OP_JIT_MIN_REGISTERS registers (an arbitrary tenth, until
// timings calibrate it), else as fast.
Estimate estimate(const loop::Facts& facts, loop::KernelKind kind);

} // namespace op::rt::tune
