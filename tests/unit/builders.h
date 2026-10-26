#pragma once

// Sets, dats, maps and arguments filled in by hand, as the plans and
// strategies read them, with no op_decl_* calls.

#include <op_lib_core.h>

#include <vector>

namespace op::test {

// size owned elements, core_size of them core (all of them when negative),
// then the halos.
inline void initialize_set(op_set_core& set, int size, int core_size = -1,
                           int exec_size = 0, int nonexec_size = 0) {
    set = {};
    set.size = size;
    set.core_size = core_size < 0 ? size : core_size;
    set.exec_size = exec_size;
    set.nonexec_size = nonexec_size;
}

inline void initialize_dat(op_dat_core& dat, op_set set, int dim, int scalar_size,
                           const char *type) {
    dat = {};
    dat.set = set;
    dat.dim = dim;
    dat.size = dim * scalar_size;
    dat.type = type;
}

inline void initialize_map(op_map_core& map, op_set from, op_set to,
                           std::vector<int>& values, int dim = 1) {
    map = {};
    map.from = from;
    map.to = to;
    map.dim = dim;
    map.map = values.data();
}

// An argument on dat through component idx of map, or directly when map is
// null, typed as the dat is unless type spells it otherwise.
inline op_arg dat_arg(op_dat dat, op_map map, int idx, op_access access,
                      const char *type = nullptr) {
    op_arg arg{};
    arg.opt = 1;
    arg.argtype = OP_ARG_DAT;
    arg.dat = dat;
    arg.map = map;
    arg.map_data = map != nullptr ? map->map : nullptr;
    arg.idx = map != nullptr ? idx : -1;
    arg.dim = dat->dim;
    arg.size = dat->size;
    arg.data = dat->data;
    arg.type = type != nullptr ? type : dat->type;
    arg.acc = access;
    return arg;
}

// Edge i of a chain reaches nodes i and i+1 through a two-component map, so
// neighbouring edges share a node; the exec halo's edges continue it.
struct Chain {
    op_set_core edges{};
    op_set_core nodes{};
    std::vector<int> map_values;
    op_map_core map{};

    explicit Chain(int size, int core = -1, int exec = 0) {
        initialize_set(edges, size, core, exec);
        initialize_set(nodes, size + exec + 1);
        for (int i = 0; i < size + exec; ++i) {
            map_values.push_back(i);
            map_values.push_back(i + 1);
        }
        initialize_map(map, &edges, &nodes, map_values, 2);
    }
};

} // namespace op::test
