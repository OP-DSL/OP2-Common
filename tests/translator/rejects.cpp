// A loop the cuda target refuses to translate, the case chosen by -D: each
// misuses a kernel argument, which the target passes as an op::rt::kernel::Ptr
// (an OP_INC one only to increment), an info argument that doesn't match its
// maximum, a map index a legacy target needs as a constant, a non-constant
// const dim, or a recursive helper.

#include <cstring>

#include "op_seq.h"

inline void take(const double *v, double *r) { r[0] = v[0]; }

#if defined(RECURSION)
inline double fact(double x) { return x <= 1.0 ? 1.0 : x * fact(x - 1.0); }
#endif

void kernel(const double *q, double *r, double *p) {
#if defined(POINTER_ARITHMETIC)
    take(q + 1, r);
#elif defined(ELEMENT_ADDRESS)
    const double *e = &q[1];
    r[0] = e[0];
#elif defined(LOCAL_ALIAS)
    const double *a = q;
    r[0] = a[0];
#elif defined(LIBRARY_CALL)
    std::memcpy(r, q, sizeof(double));
#elif defined(NULL_TEST)
    if (q)
        r[0] = q[0];
#elif defined(INC_ASSIGNED)
    p[0] = q[0];
#elif defined(INC_READ)
    r[0] = p[0];
#elif defined(RECURSION)
    r[0] = fact(q[0]);
#endif
}

#if defined(INFO_DIM) || defined(INFO_REF)
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
#if defined(RUNTIME_INDEX)
    int k = argc - 1;
    op_par_loop(kernel, "kernel", edges,
                op_arg_dat(q_dat, -1, OP_ID, 2, "double", OP_READ),
                op_arg_dat(r_dat, -1, OP_ID, 1, "double", OP_WRITE),
                op_arg_dat(p_dat, k, map, 1, "double", OP_INC));
#elif defined(CONST_DIM)
    int n = argc;
    op_decl_const(n, "double", q);
#endif
#if defined(INFO_DIM) || defined(INFO_REF)
    double max[2] = {};
    int loc[2] = {};
#if defined(INFO_DIM)
    op_par_loop(extreme, "extreme", edges,
                op_arg_dat(q_dat, -1, OP_ID, 2, "double", OP_READ),
                op_arg_gbl(max, 2, "double", OP_MAX),
                op_arg_info(loc, 1, "int", 1));
#else
    op_par_loop(extreme, "extreme", edges,
                op_arg_dat(q_dat, -1, OP_ID, 2, "double", OP_READ),
                op_arg_gbl(max, 2, "double", OP_MAX),
                op_arg_info(loc, 2, "int", 0));
#endif
#endif
    op_exit();
}
