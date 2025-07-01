#pragma once

#include <op_mpi_unified_exchanges.h>
#include <tuple>

namespace op::mpi::unified {

std::tuple<void *, void *> alloc_exchange_buffers(size_t gather_size, size_t scatter_size);

void start_gather(const GatherSpec &gather);
void start_scatter(const ScatterSpec &scatter);

void record_gathers();
void wait_gathers();

}
