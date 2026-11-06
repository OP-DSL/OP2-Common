#include <op2/rt/lifecycle.h>

#include <op2/env.h>
#include <op2/rt/context.h>
#include <op2/rt/cpu/backend.h>
#include <op2/rt/cpu/strategy.h>

#include <memory>

namespace op::rt::cpu {

Backend::Backend() {
    env::parse_number("OP_CHUNK_ATOMICS_MIN_COMPRESSION", options.min_compression);
    env::parse_number("OP_MPI_TEST_INTERVAL", options.mpi_test_interval);
}

} // namespace op::rt::cpu

namespace op::rt {

void init_cpu() {
    auto& run = context();
    run.backend = std::make_unique<cpu::Backend>();
    run.families.declare(cpu::strategy_families());
}

} // namespace op::rt
