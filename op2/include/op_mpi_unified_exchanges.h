#pragma once

#include <op_lib_core.h>
#include <op_lib_c.h>
#include <op_lib_mpi.h>

#include <cassert>

struct DatAccessor {
    void *data;

    int dim;
    int stride;
    int elem_size;

    bool soa;

    DatAccessor(op_dat dat) {
        data = (void *) dat->data_d;
        dim = dat->dim;
        stride =  round32(dat->set->size + OP_import_exec_list[dat->set->index]->size
                                         + OP_import_nonexec_list[dat->set->index]->size);
        elem_size = dat->size / dat->dim;
        soa = strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1);
    }

    template<typename T>
    constexpr T& get(std::size_t i, std::size_t j) {
        assert(elem_size == sizeof(T));

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

    ScatterSpec(int size, int *list, DatAccessor dat) :
        size{size}, list{list}, offset{-1}, dat{dat} {}

    ScatterSpec(int size, int offset, DatAccessor dat) :
        size{size}, list{nullptr}, offset{offset}, dat{dat} {}

    constexpr bool is_indirect() const { return list != nullptr; }

    constexpr size_t scatter_size() const {
        return round32(size * dat.elem_size * dat.dim);
    }
};

int op_mpi_halo_exchanges_unified(op_set set, int nargs, op_arg *args);
void op_mpi_wait_all_unified(int nargs, op_arg *args);
