#pragma once

#include <op_lib_core.h>
#include <op_lib_c.h>
#include <op_lib_mpi.h>

#include <cstdio>
#include <cstdint>

// The element copies below run in the gather/scatter kernels on a GPU backend
// and in a plain loop on a CPU one, so they are compiled for both.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define OP2_UNIFIED_HD __host__ __device__
#else
#define OP2_UNIFIED_HD
#endif

namespace op::unified_exchanges {

struct DatAccessor {
    void *data;

    int dim;
    int stride;
    int elem_size;

    bool soa;

    DatAccessor() = default;

    // Deliberately no op_dat constructor: where a dat's data lives and how it is
    // strided is the backend's business, so each one fills this in itself.
    DatAccessor(void *data, int dim, int stride, int elem_size, bool soa)
        : data{data}, dim{dim}, stride{stride}, elem_size{elem_size}, soa{soa} {}

    template<typename T>
    constexpr T& get(std::size_t i, std::size_t j) const {
        if (soa) {
            return ((T *) data)[i + j * stride];
        } else {
            return ((T *) data)[i * dim + j];
        }
    }
};

struct GatherSpec {
    int size;
    int *list;

    DatAccessor dat;
    void *target = nullptr;

    GatherSpec() = default;
    GatherSpec(int size, int *list, DatAccessor dat) :
        size{size}, list{list}, dat{dat} {}

    constexpr size_t gather_size() const {
        return round32(size * dat.elem_size * dat.dim);
    }
};

struct ScatterSpec {
    int size;
    int *list;
    int offset;

    DatAccessor dat;
    void *source = nullptr;

    ScatterSpec() = default;

    ScatterSpec(int size, int *list, DatAccessor dat) :
        size{size}, list{list}, offset{-1}, dat{dat} {}

    ScatterSpec(int size, int offset, DatAccessor dat) :
        size{size}, list{nullptr}, offset{offset}, dat{dat} {}

    constexpr bool is_indirect() const { return list != nullptr; }

    constexpr size_t scatter_size() const {
        return round32(size * dat.elem_size * dat.dim);
    }
};

OP2_UNIFIED_HD inline void unsupported_elem_size(int elem_size) {
    std::printf("op_mpi_unified_exchanges: unsupported element size %d\n", elem_size);
#if defined(__CUDA_ARCH__)
    __trap();
#else
    __builtin_trap();
#endif
}

template<typename T>
OP2_UNIFIED_HD inline void gather_components(const GatherSpec &g, std::size_t index,
                                             std::size_t set_elem) {
    for (int i = 0; i < g.dat.dim; ++i) {
        ((T *) g.target)[index * g.dat.dim + i] = g.dat.template get<T>(set_elem, i);
    }
}

template<typename T>
OP2_UNIFIED_HD inline void scatter_components(const ScatterSpec &s, std::size_t index,
                                              std::size_t set_elem) {
    for (int i = 0; i < s.dat.dim; ++i) {
        s.dat.template get<T>(set_elem, i) = ((const T *) s.source)[index * s.dat.dim + i];
    }
}

// Copy one halo element. index is the element's position within the spec.
OP2_UNIFIED_HD inline void gather_element(const GatherSpec &g, std::size_t index) {
    std::size_t set_elem = g.list[index];

    switch (g.dat.elem_size) {
        case 1:  gather_components<std::uint8_t> (g, index, set_elem); break;
        case 2:  gather_components<std::uint16_t>(g, index, set_elem); break;
        case 4:  gather_components<std::uint32_t>(g, index, set_elem); break;
        case 8:  gather_components<std::uint64_t>(g, index, set_elem); break;
        default: unsupported_elem_size(g.dat.elem_size); break;
    }
}

OP2_UNIFIED_HD inline void scatter_element(const ScatterSpec &s, std::size_t index) {
    std::size_t set_elem = s.is_indirect() ? s.list[index] : s.offset + index;

    switch (s.dat.elem_size) {
        case 1:  scatter_components<std::uint8_t> (s, index, set_elem); break;
        case 2:  scatter_components<std::uint16_t>(s, index, set_elem); break;
        case 4:  scatter_components<std::uint32_t>(s, index, set_elem); break;
        case 8:  scatter_components<std::uint64_t>(s, index, set_elem); break;
        default: unsupported_elem_size(s.dat.elem_size); break;
    }
}

}

// Entry points from op_mpi_util.cpp, global to match the rest of the runtime's
// op_* interface.
int op_mpi_halo_exchanges_unified(op_set set, int nargs, op_arg *args, int device);
void op_mpi_wait_all_unified(int nargs, op_arg *args);
