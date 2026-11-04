#include <op2/rt/lifecycle.h>

#include <op2/rt/context.h>
#include <op2/rt/cpu/strategy.h>

namespace op::rt::cpu {

void init() { op::rt::context().families.declare(strategy_families()); }

} // namespace op::rt::cpu

namespace op::rt {

void init_cpu() { cpu::init(); }

} // namespace op::rt
