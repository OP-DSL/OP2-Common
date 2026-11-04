#include <op2/rt/lifecycle.h>

#include <op2/rt/context.h>
#include <op2/rt/cpu/backend.h>
#include <op2/rt/cpu/strategy.h>

#include <memory>

namespace op::rt {

void init_cpu() {
    auto& run = context();
    run.backend = std::make_unique<cpu::Backend>();
    run.families.declare(cpu::strategy_families());
}

} // namespace op::rt
