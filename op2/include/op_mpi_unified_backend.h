#pragma once

#include <op_mpi_unified_exchanges.h>

#include <unordered_map>
#include <vector>

namespace op::mpi::unified {

struct ExchangeBuffers {
    // Where the gather and scatter kernels write and read.
    void *gather;
    void *scatter;

    // What MPI sends from and receives into. Equal to the pair above unless the
    // backend stages the exchange through separate host memory, which the
    // exchange driver does not need to know about.
    void *gather_mpi;
    void *scatter_mpi;
};

ExchangeBuffers alloc_exchange_buffers(size_t gather_size, size_t scatter_size);

void initiate_gathers(const std::unordered_map<int, std::vector<GatherSpec>> &gathers_for_neighbour);
void initiate_scatters(const std::unordered_map<int, std::vector<ScatterSpec>> &scatters_for_neighbour);

void wait_gathers();
void wait_scatters();

}
