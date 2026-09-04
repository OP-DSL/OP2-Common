#pragma once

#include <op_mpi_unified_exchanges.h>

#include <unordered_map>
#include <vector>

namespace op::unified_exchanges {

template<typename SpecT>
using SpecsByNeighbour = std::unordered_map<int, std::vector<SpecT>>;

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

// One backend per place a dat's data can live. A single binary runs both host
// and device loops, so the backend is selected per exchange from the device
// argument rather than fixed at link time.
class Backend {
public:
    // The gather has completed by the time initiate_gathers() returns, so the
    // sends can be posted straight away instead of waiting until wait-all.
    const bool synchronous;

    explicit Backend(bool synchronous) : synchronous{synchronous} {}
    virtual ~Backend() = default;

    virtual ExchangeBuffers alloc_buffers(size_t gather_size, size_t scatter_size) = 0;

    virtual void initiate_gathers(const SpecsByNeighbour<GatherSpec> &gathers_for_neighbour) = 0;
    virtual void initiate_scatters(const SpecsByNeighbour<ScatterSpec> &scatters_for_neighbour) = 0;

    virtual void wait_gathers() = 0;
    virtual void wait_scatters() = 0;
};

// Null in a library variant built without that backend.
Backend *device_backend();

// Selects from the device argument, aborting if that backend is unavailable.
Backend &backend_for(int device);

}
