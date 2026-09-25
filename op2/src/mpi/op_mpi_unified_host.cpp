#include <op_mpi_unified_backend.h>

#include <cstdint>
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
        placement.nonexec_export_permap = OP_map_halos[partial_map->index].export_nonexec.list.get();
        placement.nonexec_import_permap = OP_map_halos[partial_map->index].import_nonexec.list.get();
    } else {
        placement.exec_export = OP_set_halos[dat->set->index].export_exec.list.get();
        placement.nonexec_export = OP_set_halos[dat->set->index].export_nonexec.list.get();
    }

    return placement;
}

ExchangeBuffers HostBackend::alloc_buffers(size_t gather_size, size_t scatter_size) {
    ensure_capacity(&gather_buf, &gather_buf_size, gather_size);
    ensure_capacity(&scatter_buf, &scatter_buf_size, scatter_size);

    // MPI sends and receives straight out of the same buffers.
    return {gather_buf, scatter_buf, gather_buf, scatter_buf};
}

// The host copy is always AoS, so an element is one contiguous row of dim
// components and a spec is a loop of row copies. The element width is chosen once
// per spec, not per element as copy_element does for the device kernel, and the
// spec's fields are read into locals first: a store through T * may alias them,
// which would otherwise force them to be reloaded for every element.
template<typename Spec, typename Copy>
static void by_width(const Spec &spec, Copy copy) {
    switch (spec.dat.elem_size) {
        case 1:  copy(std::uint8_t{});  break;
        case 2:  copy(std::uint16_t{}); break;
        case 4:  copy(std::uint32_t{}); break;
        case 8:  copy(std::uint64_t{}); break;
        default: unsupported_elem_size(spec.dat.elem_size); break;
    }
}

static void gather(const GatherSpec &g) {
    by_width(g, [&](auto width) {
        using T = decltype(width);
        const T *data = static_cast<const T *>(g.dat.data);
        T *target = static_cast<T *>(g.target);
        const int *list = g.list;
        const int size = g.size;
        const std::size_t dim = g.dat.dim;
        for (int i = 0; i < size; ++i)
            for (std::size_t c = 0; c < dim; ++c)
                target[i * dim + c] = data[list[i] * dim + c];
    });
}

static void scatter(const ScatterSpec &s) {
    if (!s.is_indirect()) {
        // A full exchange's import block is contiguous in the dat: one copy.
        const std::size_t row = (std::size_t) s.dat.dim * s.dat.elem_size;
        std::memcpy((char *) s.dat.data + s.offset * row, s.source, s.size * row);
        return;
    }
    by_width(s, [&](auto width) {
        using T = decltype(width);
        T *data = static_cast<T *>(s.dat.data);
        const T *source = static_cast<const T *>(s.source);
        const int *list = s.list;
        const int size = s.size;
        const std::size_t dim = s.dat.dim;
        for (int i = 0; i < size; ++i)
            for (std::size_t c = 0; c < dim; ++c)
                data[list[i] * dim + c] = source[i * dim + c];
    });
}

void HostBackend::initiate_gathers(const SpecsByNeighbour<GatherSpec> &gathers_for_neighbour) {
    for (auto &[neighbour, specs] : gathers_for_neighbour)
        for (auto &spec : specs)
            gather(spec);
}

void HostBackend::initiate_scatters(const SpecsByNeighbour<ScatterSpec> &scatters_for_neighbour) {
    for (auto &[neighbour, specs] : scatters_for_neighbour)
        for (auto &spec : specs)
            scatter(spec);
}

Backend *host_backend() {
    static HostBackend backend;
    return &backend;
}

}
