#pragma once

// What runs for a configuration: the kind of kernel each strategy runs, and,
// when they come, the order the strategies are tried and timed in.  Decided
// here from the backends' estimates alone; a backend only estimates its
// strategies' kernels and prepares what it is asked for.

#include <loop/strategy.h>

namespace op::loop {

// The kernel kind strategy runs for configuration: a specialised kernel when
// the JIT is on and one can run, and the offline one cannot, OP_JIT_FORCE is
// set, or the backend expects it to be faster; else the offline one.  A tie
// or a missing estimate keeps the offline kernel, as a specialised one costs
// a build.
KernelKind choose_kernel(const Strategy& strategy, const Configuration& configuration);

} // namespace op::loop
