#pragma once

// What runs for a configuration, as far as heuristics decide it: the kind of
// kernel each strategy runs, and, when they come, the heuristics that order
// the strategies.  The ladder and the tuner ask here; a strategy only
// reports its kernels and prepares what it is asked for.

#include <loop/strategy.h>

namespace op::loop {

// The kernel kind strategy runs for configuration.  The one heuristic so
// far: a specialised kernel when the JIT is on and the strategy can build
// one, and it has no offline kernel that can run, OP_JIT_FORCE is set, or
// the offline kernel uses more than OP_JIT_MIN_REGISTERS registers; else the
// offline one.
KernelKind choose_kernel(const Strategy& strategy, const Configuration& configuration);

} // namespace op::loop
