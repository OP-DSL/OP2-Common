#include <op2/rt/lifecycle.h>

#include <op2/rt/cpu/strategy.h>
#include <op2/rt/loop/strategy.h>

#include <vector>

namespace op::rt::cpu {

void init() {
    auto offered = strategy_families();
    std::vector<op::rt::loop::Family> families(offered.begin(), offered.end());
    op::rt::loop::declare_families(families);
}

} // namespace op::rt::cpu

namespace op::rt {

void init_cpu() { cpu::init(); }

} // namespace op::rt
