#include <op2/mpi/exchange_backend.h>
#include <op2/mpi/halo.h>

#include <cstdlib>
#include <cstring>

namespace op::mpi {

class HostBackend final : public Backend {
public:
    // The gather is a plain loop, so it has finished by the time it returns.
    HostBackend() : Backend{true} {}
    ~HostBackend() override;

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
    if (*buffer == nullptr)
        fail("op_mpi_halo_exchanges: cannot allocate a %zu-byte exchange buffer\n", new_size);

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

// The host copy is always AoS, so an element is one contiguous row and the copy needs
// to know nothing of its components: row list[i] of the dat is row i of the buffer.
// Always inlined, so that `row` is a constant where the caller passes one and each
// memcpy is then a move or two.
template<bool to_dat>
[[gnu::always_inline]] static inline void copy_rows(std::size_t row, char *dat, char *buf,
                                                    const int *list, int n) {
    for (int i = 0; i < n; ++i) {
        char *in_dat = dat + list[i] * row, *in_buf = buf + i * row;
        if (to_dat)
            std::memcpy(in_dat, in_buf, row);
        else
            std::memcpy(in_buf, in_dat, row);
    }
}

// Most dats' rows are one of these lengths. The rest pay a memcpy call per row, which
// costs more than copying a short row does.
template<bool to_dat>
static void copy_rows(const DatAccessor &accessor, void *buffer, const int *list, int n) {
    const std::size_t row = (std::size_t) accessor.dim * accessor.elem_size;
    char *dat = static_cast<char *>(accessor.data), *buf = static_cast<char *>(buffer);
    switch (row) {
        case 4:  copy_rows<to_dat>(4, dat, buf, list, n); break;
        case 8:  copy_rows<to_dat>(8, dat, buf, list, n); break;
        case 12: copy_rows<to_dat>(12, dat, buf, list, n); break;
        case 16: copy_rows<to_dat>(16, dat, buf, list, n); break;
        case 24: copy_rows<to_dat>(24, dat, buf, list, n); break;
        case 32: copy_rows<to_dat>(32, dat, buf, list, n); break;
        case 40: copy_rows<to_dat>(40, dat, buf, list, n); break;
        case 48: copy_rows<to_dat>(48, dat, buf, list, n); break;
        default: copy_rows<to_dat>(row, dat, buf, list, n); break;
    }
}

static void gather(const GatherSpec &g) { copy_rows<false>(g.dat, g.target, g.list, g.size); }

static void scatter(const ScatterSpec &s) {
    if (!s.is_indirect()) {
        // A full exchange's import block is contiguous in the dat: one copy.
        const std::size_t row = (std::size_t) s.dat.dim * s.dat.elem_size;
        std::memcpy((char *) s.dat.data + s.offset * row, s.source, s.size * row);
        return;
    }
    copy_rows<true>(s.dat, s.source, s.list, s.size);
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

HostBackend::~HostBackend() {
    free(gather_buf);
    free(scatter_buf);
}

std::unique_ptr<Backend> make_host_backend() { return std::make_unique<HostBackend>(); }

}
