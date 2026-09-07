#include <op_mpi_unified_backend.h>

#include <cstdlib>
#include <cstring>

namespace op::unified_exchanges {

class HostBackend final : public Backend {
public:
    // The gather is a plain loop, so it has finished by the time it returns.
    HostBackend() : Backend{true} {}

    DatPlacement placement(op_dat dat, op_map partial_map) const override;

    ExchangeBuffers alloc_buffers(size_t gather_size, size_t scatter_size) override;

    void initiate_gathers(const SpecsByNeighbour<GatherSpec> &gathers_for_neighbour) override;
    void initiate_scatters(const SpecsByNeighbour<ScatterSpec> &scatters_for_neighbour) override;

    void wait_gathers() override {}
    void wait_scatters() override {}

private:
    void *gather_buf = nullptr;
    size_t gather_buf_size = 0;

    void *scatter_buf = nullptr;
    size_t scatter_buf_size = 0;
};

static void ensure_capacity(void **buffer, size_t *size, size_t capacity) {
    if (capacity <= *size) {
        return;
    }

    size_t new_size = capacity * 1.2;

    free(*buffer);
    *buffer = malloc(new_size);

    *size = new_size;
}

DatPlacement HostBackend::placement(op_dat dat, op_map partial_map) const {
    DatPlacement placement;
    // The host copy is always AoS, so the stride is unused.
    placement.dat = DatAccessor((void *) dat->data, dat->dim, 0, dat->size / dat->dim, false);

    if (partial_map != nullptr) {
        placement.nonexec_export_permap = OP_export_nonexec_permap[partial_map->index]->list;
        placement.nonexec_import_permap = OP_import_nonexec_permap[partial_map->index]->list;
    } else {
        placement.exec_export = OP_export_exec_list[dat->set->index]->list;
        placement.nonexec_export = OP_export_nonexec_list[dat->set->index]->list;
    }

    return placement;
}

ExchangeBuffers HostBackend::alloc_buffers(size_t gather_size, size_t scatter_size) {
    ensure_capacity(&gather_buf, &gather_buf_size, gather_size);
    ensure_capacity(&scatter_buf, &scatter_buf_size, scatter_size);

    // MPI sends and receives straight out of the same buffers.
    return {gather_buf, scatter_buf, gather_buf, scatter_buf};
}

template<typename SpecT>
static void run_copies(const SpecsByNeighbour<SpecT> &specs_for_neighbour) {
    for (auto &[neighbour, specs] : specs_for_neighbour) {
        for (auto &spec : specs) {
            for (int i = 0; i < spec.size; ++i) {
                copy_element(spec, i);
            }
        }
    }
}

void HostBackend::initiate_gathers(const SpecsByNeighbour<GatherSpec> &gathers_for_neighbour) {
    run_copies(gathers_for_neighbour);
}

void HostBackend::initiate_scatters(const SpecsByNeighbour<ScatterSpec> &scatters_for_neighbour) {
    run_copies(scatters_for_neighbour);
}

Backend *host_backend() {
    static HostBackend backend;
    return &backend;
}

}
