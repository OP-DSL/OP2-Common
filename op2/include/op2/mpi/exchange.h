#pragma once

#include <op_lib_core.h>
#include <op_lib_c.h>
#include <op_lib_mpi.h>

#include <cstddef>

// The element copy below runs in a GPU backend's gather and scatter kernels; the
// host backend copies whole rows instead.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define OP2_UNIFIED_HD __host__ __device__
#else
#define OP2_UNIFIED_HD
#endif

namespace op::mpi {

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

// The element of its dat that element `index` of a spec is.
OP2_UNIFIED_HD inline std::size_t set_element(const GatherSpec &g, std::size_t index) {
    return g.list[index];
}

OP2_UNIFIED_HD inline std::size_t set_element(const ScatterSpec &s, std::size_t index) {
    return s.is_indirect() ? s.list[index] : s.offset + index;
}

// W bytes between two addresses aligned to W, as every component of a dat and of an
// exchange buffer is. A memcpy copies a dat of any type. The builtins, because nvcc
// moves single bytes without the promise and does not take it from
// std::assume_aligned, and HIP declares no device memcpy here.
template<int W>
OP2_UNIFIED_HD inline void copy_component(void *to, const void *from) {
    __builtin_memcpy(__builtin_assume_aligned(to, W), __builtin_assume_aligned(from, W), W);
}

template<int W>
OP2_UNIFIED_HD inline void copy_components(const GatherSpec &g, std::size_t index) {
    const DatAccessor &dat = g.dat;
    const std::size_t elem = set_element(g, index);
    const std::size_t first = dat.soa ? elem : elem * dat.dim, step = dat.soa ? dat.stride : 1;
    for (int c = 0; c < dat.dim; ++c)
        copy_component<W>((char *) g.target + (index * dat.dim + c) * W,
                          (const char *) dat.data + (first + c * step) * W);
}

template<int W>
OP2_UNIFIED_HD inline void copy_components(const ScatterSpec &s, std::size_t index) {
    const DatAccessor &dat = s.dat;
    const std::size_t elem = set_element(s, index);
    const std::size_t first = dat.soa ? elem : elem * dat.dim, step = dat.soa ? dat.stride : 1;
    for (int c = 0; c < dat.dim; ++c)
        copy_component<W>((char *) dat.data + (first + c * step) * W,
                          (const char *) s.source + (index * dat.dim + c) * W);
}

// Copy one halo element. index is the element's position within the spec.
template<typename Spec>
OP2_UNIFIED_HD inline void copy_element(const Spec &spec, std::size_t index) {
    switch (spec.dat.elem_size) {
        case 1:  copy_components<1>(spec, index); break;
        case 2:  copy_components<2>(spec, index); break;
        case 4:  copy_components<4>(spec, index); break;
        case 8:  copy_components<8>(spec, index); break;
        default: break;  // the exchange takes no other width
    }
}

}
