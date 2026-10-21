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

// Where a backend finds a dat's data and the halo index lists that go with it.
// Host data is always AoS and its lists live in HaloList::list; a device
// backend keeps its own copies of both.
struct DatPlacement {
    DatAccessor dat;

    // Full exchange, indexed by set.
    int *exec_export = nullptr;
    int *nonexec_export = nullptr;

    // Partial exchange, indexed by map.
    int *nonexec_export_permap = nullptr;
    int *nonexec_import_permap = nullptr;
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

    // Everything about a dat that differs between memory spaces, asked once per
    // exchange. Only the pair matching the exchange kind is filled in.
    virtual DatPlacement placement(op_dat dat, op_map partial_map) const = 0;

    virtual ExchangeBuffers alloc_buffers(size_t gather_size, size_t scatter_size) = 0;

    virtual void initiate_gathers(const SpecsByNeighbour<GatherSpec> &gathers_for_neighbour) = 0;
    virtual void initiate_scatters(const SpecsByNeighbour<ScatterSpec> &scatters_for_neighbour) = 0;

    virtual void wait_gathers() = 0;
    virtual void wait_scatters() = 0;

    // --- Asynchronous exchanges (OP_MPI_ASYNC) ---
    // An exchange in flight holds a slot: buffers, events and a flag of its
    // own.  Its scatter waits on the device for the flag, which the host
    // raises once the receives are in, so the host never blocks on it.  Only
    // a device backend has slots.

    static constexpr int slots = 8;

    virtual bool asynchronous() const { return false; }

    // Whether the slot's buffers would have to grow: done only once nothing
    // is in flight, as freeing staging memory synchronises the device.
    virtual bool grows(int slot, size_t gather_size, size_t scatter_size) const {
        (void) slot; (void) gather_size; (void) scatter_size;
        return false;
    }

    virtual ExchangeBuffers slot_buffers(int slot, size_t gather_size, size_t scatter_size) {
        (void) slot;
        return alloc_buffers(gather_size, scatter_size);
    }

    virtual void slot_gathers(int slot, const SpecsByNeighbour<GatherSpec> &gathers_for_neighbour) {
        (void) slot;
        initiate_gathers(gathers_for_neighbour);
    }

    // The scatter, held on the device until the slot's flag reaches seq.
    virtual void slot_scatters(int slot, unsigned seq,
                               const SpecsByNeighbour<ScatterSpec> &scatters_for_neighbour) {
        (void) slot; (void) seq;
        initiate_scatters(scatters_for_neighbour);
    }

    // Whether the slot's gather (and its copy to the host) or scatter has run.
    virtual bool gathered(int slot) { (void) slot; return true; }
    virtual bool scattered(int slot) { (void) slot; return true; }

    virtual void raise(int slot, unsigned seq) { (void) slot; (void) seq; }
};

// Always available: the host backend has no accelerator dependency.
Backend *host_backend();

// Null in a library variant built without an accelerator backend.
Backend *device_backend();

// Selects from the device argument, aborting if that backend is unavailable.
Backend &backend_for(int device);

}
