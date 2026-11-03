#include <op2/rt/lifecycle.h>

#include <op2/rt/cpu/strategy.h>
#include <op2/rt/loop/strategy.h>

namespace op::rt::cpu {

void init() { op::rt::loop::declare_families(strategy_families()); }

} // namespace op::rt::cpu

namespace op::rt {

void init_cpu() { cpu::init(); }

} // namespace op::rt
