// The translator corpus: kernels written the ways the C++ translator
// rewrites (every accepted increment form, helpers taking arguments,
// overloaded helpers, array parameters, a local array passed where an
// argument is, branches and loops, map indices and dims known only at run
// time, a helper chosen by __cplusplus), each loop checked against its
// kernel called directly, element by element, on the same data.

#include "op_seq.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int nedges = 96;
constexpr int nnodes = 40;

// --- KERNELS ---
inline void add_to(double *t, double v) { t[0] += v; }
inline void add_to(float *t, float v) { t[0] += v; }

void increments(double *a, double *b, const double *w) {
    a[0] += w[0];
    a[1] -= 0.5 * w[0];
    ++a[2];
    b[0]--;
    b[1] = b[1] + w[0] - 1.0;
    b[2] = 2.0 * w[0] + b[2];
    *a += 1.0;
    add_to(b, w[0]);
}

inline double dot(const double *u, const double *v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

#if __cplusplus >= 202002L
inline double standard(double x) { return x; }
#else
inline double standard(double x) { return -x; }
#endif

void arrays(const double q[3], const double *n, double r[]) {
    const double unit[3] = {1.0, 1.0, 1.0};
    r[0] = dot(q, unit);
    r[1] = dot(unit, n);
    r[2] = dot(q, n) + sqrt(fabs(q[0])) + standard(q[1]);
}

inline double weight(double x) { return 2.0 * x; }
inline double weight(int k) { return 0.5 * k; }

void control(const double *x, const int *k, double *y) {
    double acc = 0.0;
    for (int d = 0; d < 3; ++d) {
        if (x[d] > 0.5)
            acc += x[d];
        else if (k[0] % 2 == 0)
            acc -= x[d];
        else
            continue;
    }
    switch (k[0] % 3) {
    case 0:
        y[0] = acc;
        break;
    case 1:
        y[0] = -acc;
        break;
    default:
        y[0] = 2.0 * acc;
    }
    int i = 0;
    while (i < k[0] && i < 5)
        ++i;
    y[1] = (k[0] > 3 ? fmin(acc, x[0]) : fmax(acc, x[1])) + i + weight(acc) + weight(k[0]);
}

void gather(const double *n, const double *g, double *y) {
    for (int d = 0; d < 3; ++d)
        y[d] = n[d] * g[d];
}

// Stop unless every value of a loop's result matches the direct calls'.
void compare(const char *loop, const std::vector<double>& expected, op_dat dat) {
    std::vector<double> fetched(expected.size());
    op_fetch_data(dat, fetched.data());
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (std::fabs(fetched[i] - expected[i]) >
            1e-12 * std::fmax(1.0, std::fabs(expected[i]))) {
            std::printf("ERROR: %s value %zu: %g, expected %g\n", loop, i, fetched[i],
                        expected[i]);
            op_exit();
            std::exit(EXIT_FAILURE);
        }
    std::printf("%s passed\n", loop);
}

int main(int argc, char **argv) {
    op_init(argc, argv, 2);

    std::vector<int> edge_nodes(2 * nedges), k(nedges);
    std::vector<double> x(3 * nedges), w(nedges), node3(3 * nnodes), acc(3 * nnodes, 1.0);
    for (int e = 0; e < nedges; ++e) {
        edge_nodes[2 * e] = e % nnodes;
        edge_nodes[2 * e + 1] = (7 * e + 3) % nnodes;
        x[3 * e] = (5 * e % 11) / 10.0;
        x[3 * e + 1] = (3 * e % 7) / 6.0;
        x[3 * e + 2] = (e % 5) / 4.0 - 0.25;
        w[e] = (13 * e % 17) / 16.0;
        k[e] = 11 * e % 9;
    }
    for (int i = 0; i < 3 * nnodes; ++i)
        node3[i] = (i % 13) / 12.0;
    std::vector<double> y3(3 * nedges, 0.0), y2(2 * nedges, 0.0);

    op_set edges = op_decl_set(nedges, "edges");
    op_set nodes = op_decl_set(nnodes, "nodes");
    op_map pedge = op_decl_map(edges, nodes, 2, edge_nodes.data(), "pedge");
    op_dat p_x = op_decl_dat(edges, 3, "double", x.data(), "x");
    op_dat p_w = op_decl_dat(edges, 1, "double", w.data(), "w");
    op_dat p_k = op_decl_dat(edges, 1, "int", k.data(), "k");
    op_dat p_node3 = op_decl_dat(nodes, 3, "double", node3.data(), "node3");
    op_dat p_acc0 = op_decl_dat(nodes, 3, "double", acc.data(), "acc0");
    op_dat p_acc1 = op_decl_dat(nodes, 3, "double", acc.data(), "acc1");
    op_dat p_y3 = op_decl_dat(edges, 3, "double", y3.data(), "y3");
    op_dat p_y2 = op_decl_dat(edges, 2, "double", y2.data(), "y2");

    op_par_loop(increments, "increments", edges,
                op_arg_dat(p_acc0, 0, pedge, 3, "double", OP_INC),
                op_arg_dat(p_acc1, 1, pedge, 3, "double", OP_INC),
                op_arg_dat(p_w, -1, OP_ID, 1, "double", OP_READ));
    std::vector<double> expected0(acc), expected1(acc);
    for (int e = 0; e < nedges; ++e)
        increments(&expected0[3 * edge_nodes[2 * e]], &expected1[3 * edge_nodes[2 * e + 1]],
                   &w[e]);
    compare("increments", expected0, p_acc0);
    compare("increments", expected1, p_acc1);

    op_par_loop(arrays, "arrays", edges,
                op_arg_dat(p_x, -1, OP_ID, 3, "double", OP_READ),
                op_arg_dat(p_node3, 1, pedge, 3, "double", OP_READ),
                op_arg_dat(p_y3, -1, OP_ID, 3, "double", OP_WRITE));
    std::vector<double> expected(3 * nedges);
    for (int e = 0; e < nedges; ++e)
        arrays(&x[3 * e], &node3[3 * edge_nodes[2 * e + 1]], &expected[3 * e]);
    compare("arrays", expected, p_y3);

    op_par_loop(control, "control", edges,
                op_arg_dat(p_node3, 0, pedge, 3, "double", OP_READ),
                op_arg_dat(p_k, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(p_y2, -1, OP_ID, 2, "double", OP_WRITE));
    expected.assign(2 * nedges, 0.0);
    for (int e = 0; e < nedges; ++e)
        control(&node3[3 * edge_nodes[2 * e]], &k[e], &expected[2 * e]);
    compare("control", expected, p_y2);

    // Not constants, so the translator leaves them to run time.
    int idx = argc > 0 ? 1 : 0;
    int dim = argc > 0 ? 3 : 1;
    double g[3] = {0.5, -2.0, 3.0};
    op_par_loop(gather, "gather", edges,
                op_arg_dat(p_node3, idx, pedge, dim, "double", OP_READ),
                op_arg_gbl(g, dim, "double", OP_READ),
                op_arg_dat(p_y3, -1, OP_ID, dim, "double", OP_WRITE));
    expected.assign(3 * nedges, 0.0);
    for (int e = 0; e < nedges; ++e)
        gather(&node3[3 * edge_nodes[2 * e + 1]], g, &expected[3 * e]);
    compare("gather", expected, p_y3);

    op_exit();
    return 0;
}
