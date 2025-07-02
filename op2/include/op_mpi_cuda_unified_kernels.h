#pragma once

#include <op_mpi_unified_exchanges.h>

#include <tuple>
#include <map>
#include <vector>

namespace op::mpi::unified {

std::tuple<void *, void *> alloc_exchange_buffers(size_t gather_size, size_t scatter_size);

void initiate_gathers(const std::map<int, std::vector<GatherSpec>> &gathers_for_neighbour);
void initiate_scatters(const std::map<int, std::vector<ScatterSpec>> &scatters_for_neighbour);

void wait_gathers();

}
