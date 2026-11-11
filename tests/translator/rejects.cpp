// A loop the translator refuses, the case chosen by -D: an info argument that
// doesn't match its maximum, one dat passed as two types, a map index a legacy
// target needs as a constant, a non-constant const dim, a recursive helper, a
// function template or a helper in a namespace, an include that isn't there,
// or an info argument, which the legacy GPU targets don't support.

#if defined(MISSING_INCLUDE)
#include <op2_no_such_header.h>
#endif

#include "op_seq.h"

inline void take(const double *v, double *r) { r[0] = v[0]; }

#if defined(RECURSION)
inline double fact(double x) { return x <= 1.0 ? 1.0 : x * fact(x - 1.0); }
#elif defined(TEMPLATE_CALL)
template<typename T> inline T twice(T x) { return x + x; }
#elif defined(NAMESPACED)
namespace helpers {
inline double half(double x) { return 0.5 * x; }
}
#endif

void kernel(const double *q, double *r, double *p) {
#if defined(RECURSION)
    r[0] = fact(q[0]);
#elif defined(TEMPLATE_CALL)
    r[0] = twice(q[0]);
#elif defined(NAMESPACED)
    r[0] = helpers::half(q[0]);
#endif
}

#if defined(INFO_DIM) || defined(INFO_REF) || defined(LEGACY_INFO)
void extreme(const double *q, double *max, int *loc) {
    for (int d = 0; d < 2; ++d)
        if (q[d] > max[d]) {
            max[d] = q[d];
            loc[d] = 0;
        }
}
#endif

int main(int argc, char **argv) {
    op_init(argc, argv, 0);
    op_set nodes = op_decl_set(4, "nodes");
    op_set edges = op_decl_set(4, "edges");
    int edge_nodes[4] = {0, 1, 2, 3};
    op_map map = op_decl_map(edges, nodes, 1, edge_nodes, "edge_nodes");
    double q[8] = {}, r[4] = {}, p[4] = {};
    op_dat q_dat = op_decl_dat(edges, 2, "double", q, "q");
    op_dat r_dat = op_decl_dat(edges, 1, "double", r, "r");
    op_dat p_dat = op_decl_dat(nodes, 1, "double", p, "p");
    op_par_loop(kernel, "kernel", edges,
                op_arg_dat(q_dat, -1, OP_ID, 2, "double", OP_READ),
                op_arg_dat(r_dat, -1, OP_ID, 1, "double", OP_WRITE),
                op_arg_dat(p_dat, 0, map, 1, "double", OP_INC));
#if defined(CONFLICTING_DAT)
    op_par_loop(kernel, "kernel", edges,
                op_arg_dat(q_dat, -1, OP_ID, 2, "double", OP_READ),
                op_arg_dat(q_dat, -1, OP_ID, 1, "float", OP_WRITE),
                op_arg_dat(p_dat, 0, map, 1, "double", OP_INC));
#elif defined(RUNTIME_INDEX)
    int k = argc - 1;
    op_par_loop(kernel, "kernel", edges,
                op_arg_dat(q_dat, -1, OP_ID, 2, "double", OP_READ),
                op_arg_dat(r_dat, -1, OP_ID, 1, "double", OP_WRITE),
                op_arg_dat(p_dat, k, map, 1, "double", OP_INC));
#elif defined(CONST_DIM)
    int n = argc;
    op_decl_const(n, "double", q);
#endif
#if defined(INFO_DIM) || defined(INFO_REF) || defined(LEGACY_INFO)
    double max[2] = {};
    int loc[2] = {};
#if defined(INFO_DIM)
    op_par_loop(extreme, "extreme", edges,
                op_arg_dat(q_dat, -1, OP_ID, 2, "double", OP_READ),
                op_arg_gbl(max, 2, "double", OP_MAX),
                op_arg_info(loc, 1, "int", 1));
#elif defined(LEGACY_INFO)
    op_par_loop(extreme, "extreme", edges,
                op_arg_dat(q_dat, -1, OP_ID, 2, "double", OP_READ),
                op_arg_gbl(max, 2, "double", OP_MAX),
                op_arg_info(loc, 2, "int", 1));
#else
    op_par_loop(extreme, "extreme", edges,
                op_arg_dat(q_dat, -1, OP_ID, 2, "double", OP_READ),
                op_arg_gbl(max, 2, "double", OP_MAX),
                op_arg_info(loc, 2, "int", 0));
#endif
#endif
    op_exit();
}
