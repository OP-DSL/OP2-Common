// Implicit proxy: backward Euler for coupled heat equations,
// u_t = div(K grad u) - C u with NV components, on the interior nodes of an
// n x n grid of the unit square stored as an unstructured mesh (nodes, and
// edges between neighbours, randomly renumbered), u = 0 on the boundary.
//
// Each step assembles the system into solver memory with par loops, builds
// the right-hand side and solves for u in place. u0 = w sin(pi x) sin(pi y)
// is an eigenvector of the discrete operator, so the discrete solution is
// known exactly, checking assembly and solve; the continuous one checks the
// discretisation.
//
//   implicit [-n N] [-steps S] [-dt DT] [-pc lib|bjacobi] [-op assembled|matfree]
//            [-reassemble 0|1] [-init mode|bump] [-solver "<op_solve options>"] [-diags N]
//            [-pc neumann -terms M] [-pmat decoupled] [-timing 1]
//
// -pc neumann preconditions with M Neumann steps after block Jacobi, each a
// matrix-vector product through the blocks in solver memory; -pmat decoupled
// builds the library's preconditioner from the components' operators alone.
//
// -init bump starts from Gaussian bumps instead of the mode, for a solve with
// work in it; the solution is then checked by its true residual, b - A x
// with A applied by par loops, which checks the assembled matrix too.

#include "op_seq.h"
#include "op_solve.h"

#include <mpi.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#ifndef NV
#define NV 1
#endif

// At least two entries, as a one-entry const is a scalar to the translator.
#define NK (NV * NV > 1 ? NV * NV : 2)

double h2dt, h2;
double Kc[NK], Cc[NK];

#include "implicit_kernels.h"

struct Context {
    op_set nodes, edges;
    op_map edge_nodes;
    op_dat nbnd, dinv, y;
    op_mat A;
    int terms;
};

static void bjacobi(op_dat r, op_dat z, void *ctx) {
    Context *c = (Context *)ctx;
    op_par_loop(apply_block, "apply_block", c->nodes,
                op_arg_dat(c->dinv, -1, OP_ID, NV * NV, "double", OP_READ),
                op_arg_dat(r, -1, OP_ID, NV, "double", OP_READ),
                op_arg_dat(z, -1, OP_ID, NV, "double", OP_WRITE));
}

// y = A x through the assembled matrix, read in solver memory.
static void matvec(Context *c, op_dat x, op_dat y) {
    op_par_loop(mat_diag, "mat_diag", c->nodes,
                op_arg_mat(c->A, -1, OP_ID, -1, OP_ID, NV, "double", OP_READ),
                op_arg_dat(x, -1, OP_ID, NV, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, NV, "double", OP_WRITE));
    op_par_loop(mat_edge, "mat_edge", c->edges,
                op_arg_mat(c->A, 0, c->edge_nodes, 1, c->edge_nodes, NV, "double", OP_READ),
                op_arg_mat(c->A, 1, c->edge_nodes, 0, c->edge_nodes, NV, "double", OP_READ),
                op_arg_dat(x, 0, c->edge_nodes, NV, "double", OP_READ),
                op_arg_dat(x, 1, c->edge_nodes, NV, "double", OP_READ),
                op_arg_dat(y, 0, c->edge_nodes, NV, "double", OP_INC),
                op_arg_dat(y, 1, c->edge_nodes, NV, "double", OP_INC));
}

// z = sum_{i<=m} (I - Dinv A)^i Dinv r: block Jacobi, then m steps that each
// apply A through its blocks in solver memory.
static void neumann(op_dat r, op_dat z, void *ctx) {
    Context *c = (Context *)ctx;
    bjacobi(r, z, ctx);
    for (int i = 0; i < c->terms; i++) {
        matvec(c, z, c->y);
        op_par_loop(neumann_step, "neumann_step", c->nodes,
                    op_arg_dat(c->dinv, -1, OP_ID, NV * NV, "double", OP_READ),
                    op_arg_dat(r, -1, OP_ID, NV, "double", OP_READ),
                    op_arg_dat(c->y, -1, OP_ID, NV, "double", OP_READ),
                    op_arg_dat(z, -1, OP_ID, NV, "double", OP_RW));
    }
}

static void matfree(op_dat x, op_dat y, void *ctx) {
    Context *c = (Context *)ctx;
    op_par_loop(apply_node, "apply_node", c->nodes,
                op_arg_dat(c->nbnd, -1, OP_ID, 1, "int", OP_READ),
                op_arg_dat(x, -1, OP_ID, NV, "double", OP_READ),
                op_arg_dat(y, -1, OP_ID, NV, "double", OP_WRITE));
    op_par_loop(apply_edge, "apply_edge", c->edges,
                op_arg_dat(x, 0, c->edge_nodes, NV, "double", OP_READ),
                op_arg_dat(x, 1, c->edge_nodes, NV, "double", OP_READ),
                op_arg_dat(y, 0, c->edge_nodes, NV, "double", OP_INC),
                op_arg_dat(y, 1, c->edge_nodes, NV, "double", OP_INC));
}

// Small dense helpers for the exact solutions, on the host.
typedef std::vector<double> Matrix;

static Matrix multiply(const Matrix& a, const Matrix& b) {
    Matrix c(NV * NV, 0.0);
    for (int i = 0; i < NV; i++)
        for (int k = 0; k < NV; k++)
            for (int j = 0; j < NV; j++)
                c[i * NV + j] += a[i * NV + k] * b[k * NV + j];
    return c;
}

static Matrix inverse(Matrix a) {
    Matrix x(NV * NV, 0.0);
    for (int k = 0; k < NV; k++)
        x[k * NV + k] = 1.0;
    for (int k = 0; k < NV; k++) {
        double p = 1.0 / a[k * NV + k];
        for (int c = 0; c < NV; c++) {
            a[k * NV + c] *= p;
            x[k * NV + c] *= p;
        }
        for (int r = 0; r < NV; r++)
            if (r != k) {
                double f = a[r * NV + k];
                for (int c = 0; c < NV; c++) {
                    a[r * NV + c] -= f * a[k * NV + c];
                    x[r * NV + c] -= f * x[k * NV + c];
                }
            }
    }
    return x;
}

// exp(-m t) by scaling and squaring a Taylor series.
static Matrix expm(const Matrix& m, double t) {
    double norm = 0.0;
    for (double v : m)
        norm = std::max(norm, fabs(v) * NV * t);
    int squarings = 0;
    while (norm > 0.25) {
        norm /= 2;
        squarings++;
    }
    double scale = -t / pow(2.0, squarings);
    Matrix e(NV * NV, 0.0), term(NV * NV, 0.0);
    for (int k = 0; k < NV; k++)
        e[k * NV + k] = term[k * NV + k] = 1.0;
    for (int n = 1; n < 30; n++) {
        Matrix next = multiply(term, m);
        for (int k = 0; k < NV * NV; k++)
            term[k] = next[k] * scale / n;
        for (int k = 0; k < NV * NV; k++)
            e[k] += term[k];
    }
    for (int s = 0; s < squarings; s++)
        e = multiply(e, e);
    return e;
}

static std::vector<double> mat_vec(const Matrix& a, const std::vector<double>& x) {
    std::vector<double> y(NV, 0.0);
    for (int i = 0; i < NV; i++)
        for (int j = 0; j < NV; j++)
            y[i] += a[i * NV + j] * x[j];
    return y;
}

static const char *option(int argc, char **argv, const char *name, const char *fallback) {
    for (int i = 1; i + 1 < argc; i++)
        if (strcmp(argv[i], name) == 0)
            return argv[i + 1];
    return fallback;
}

int main(int argc, char **argv) {
    op_init(argc, argv, atoi(option(argc, argv, "-diags", "1")));

    const int n = atoi(option(argc, argv, "-n", "64"));
    const int steps = atoi(option(argc, argv, "-steps", "10"));
    const double h = 1.0 / (n + 1);
    const double dt = atof(option(argc, argv, "-dt", "0.001"));
    const std::string pc = option(argc, argv, "-pc", "lib");
    const std::string op = option(argc, argv, "-op", "assembled");
    const int reassemble = atoi(option(argc, argv, "-reassemble", "1"));
    const std::string init = option(argc, argv, "-init", "mode");
    const std::string pmat = option(argc, argv, "-pmat", "A");
    const char *solver_options = option(argc, argv, "-solver", "method=cg pc=jacobi rtol=1e-10");
    h2 = h * h;
    h2dt = h2 / dt;

    // K and C symmetric and diagonally dominant, so SPD.
    for (int i = 0; i < NV; i++)
        for (int j = 0; j < NV; j++) {
            Kc[i * NV + j] = i == j ? 1.0 : 0.2 / (1 + abs(i - j));
            Cc[i * NV + j] = i == j ? 2.0 + i : 0.5 / (1 + abs(i - j));
        }

    int rank = 0, ranks = 1, mpi;
    MPI_Initialized(&mpi);
    if (mpi) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    }

    // The global mesh, randomly renumbered; each rank declares its block.
    const int gnodes = n * n, gedges = 2 * n * (n - 1);
    std::vector<int> perm(gnodes);
    for (int i = 0; i < gnodes; i++)
        perm[i] = i;
    std::mt19937 rng(12345);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int> gedge_nodes;
    gedge_nodes.reserve(2 * gedges);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            if (i + 1 < n)
                gedge_nodes.insert(gedge_nodes.end(), {perm[j * n + i], perm[j * n + i + 1]});
            if (j + 1 < n)
                gedge_nodes.insert(gedge_nodes.end(), {perm[j * n + i], perm[(j + 1) * n + i]});
        }
    std::vector<int> eperm(gedges);
    for (int e = 0; e < gedges; e++)
        eperm[e] = e;
    std::shuffle(eperm.begin(), eperm.end(), rng);

    auto block = [&](int total, int r) { return (int)((long long)total * r / ranks); };
    const int n0 = block(gnodes, rank), n1 = block(gnodes, rank + 1);
    const int e0 = block(gedges, rank), e1 = block(gedges, rank + 1);

    std::vector<int> owner(gnodes);
    for (int i = 0; i < gnodes; i++)
        owner[perm[i]] = i;
    std::vector<double> xy(2 * (n1 - n0));
    std::vector<int> nbnd(n1 - n0);
    for (int g = n0; g < n1; g++) {
        int i = owner[g] % n, j = owner[g] / n;
        xy[2 * (g - n0)] = (i + 1) * h;
        xy[2 * (g - n0) + 1] = (j + 1) * h;
        nbnd[g - n0] = (i == 0) + (i == n - 1) + (j == 0) + (j == n - 1);
    }
    std::vector<int> edge_nodes(2 * (e1 - e0));
    for (int e = e0; e < e1; e++) {
        edge_nodes[2 * (e - e0)] = gedge_nodes[2 * eperm[e]];
        edge_nodes[2 * (e - e0) + 1] = gedge_nodes[2 * eperm[e] + 1];
    }

    Context ctx;
    ctx.nodes = op_decl_set(n1 - n0, "nodes");
    ctx.edges = op_decl_set(e1 - e0, "edges");
    ctx.edge_nodes = op_decl_map(ctx.edges, ctx.nodes, 2, edge_nodes.data(), "edge_nodes");
    op_dat p_xy = op_decl_dat(ctx.nodes, 2, "double", xy.data(), "xy");
    ctx.nbnd = op_decl_dat(ctx.nodes, 1, "int", nbnd.data(), "nbnd");
    op_dat p_u = op_decl_dat(ctx.nodes, NV, "double", (double *)NULL, "u");
    op_dat p_b = op_decl_dat(ctx.nodes, NV, "double", (double *)NULL, "b");
    ctx.dinv = op_decl_dat(ctx.nodes, NV * NV, "double", (double *)NULL, "dinv");
    op_dat p_ax = op_decl_dat(ctx.nodes, NV, "double", (double *)NULL, "ax");
    ctx.y = op_decl_dat(ctx.nodes, NV, "double", (double *)NULL, "y");
    ctx.terms = atoi(option(argc, argv, "-terms", "2"));

    op_decl_const(1, "double", &h2dt);
    op_decl_const(1, "double", &h2);
    op_decl_const(NK, "double", Kc);
    op_decl_const(NK, "double", Cc);

    op_partition("PARMETIS", "KWAY", ctx.edges, ctx.edge_nodes, p_xy);

    op_solve_init(argc, argv, NULL);
    op_sparsity sp = op_decl_sparsity(ctx.nodes, ctx.nodes, 1, &ctx.edge_nodes, &ctx.edge_nodes, "laplacian");
    op_mat A = op_decl_mat(sp, NV, "double", "A");
    ctx.A = A;
    op_mat P = pmat == "decoupled" ? op_decl_mat(sp, NV, "double", "P") : NULL;
    op_solver solver = op_decl_solver(A, "heat", solver_options);
    if (pc == "bjacobi")
        op_solver_set_pc(solver, bjacobi, &ctx);
    if (pc == "neumann")
        op_solver_set_pc(solver, neumann, &ctx);
    if (P != NULL)
        op_solver_set_pc_mat(solver, P);
    if (op == "matfree")
        op_solver_set_operator(solver, matfree, &ctx);

    double w[NV];
    for (int v = 0; v < NV; v++)
        w[v] = 1.0 / (1 + v);
    op_par_loop(initial, "initial", ctx.nodes,
                op_arg_dat(p_xy, -1, OP_ID, 2, "double", OP_READ),
                op_arg_gbl(w, NV, "double", OP_READ),
                op_arg_dat(p_u, -1, OP_ID, NV, "double", OP_WRITE));
    if (init == "bump")
        op_par_loop(bump, "bump", ctx.nodes,
                    op_arg_dat(p_xy, -1, OP_ID, 2, "double", OP_READ),
                    op_arg_gbl(w, NV, "double", OP_READ),
                    op_arg_dat(p_u, -1, OP_ID, NV, "double", OP_WRITE));

    // Device work is asynchronous, so each phase waits for it before its time.
    auto wall = []() {
        deviceSync();
        double cpu, now;
        op_timers(&cpu, &now);
        return now;
    };
    double loops_time = 0, commit_time = 0, solve_time = 0, t0, t1, t2, step_loops = 0, step_commit = 0;
    int iterations = 0, converged = 1;
    const double start = wall();
    for (int step = 0; step < steps; step++) {
        if (step == 0 || reassemble) {
            t0 = wall();
            op_mat_zero(A);
            op_par_loop(diag_block, "diag_block", ctx.nodes,
                        op_arg_dat(ctx.nbnd, -1, OP_ID, 1, "int", OP_READ),
                        op_arg_mat(A, -1, OP_ID, -1, OP_ID, NV, "double", OP_INC));
            op_par_loop(edge_blocks, "edge_blocks", ctx.edges,
                        op_arg_mat(A, 0, ctx.edge_nodes, 0, ctx.edge_nodes, NV, "double", OP_INC),
                        op_arg_mat(A, 0, ctx.edge_nodes, 1, ctx.edge_nodes, NV, "double", OP_INC),
                        op_arg_mat(A, 1, ctx.edge_nodes, 0, ctx.edge_nodes, NV, "double", OP_INC),
                        op_arg_mat(A, 1, ctx.edge_nodes, 1, ctx.edge_nodes, NV, "double", OP_INC));
            t1 = wall();
            op_mat_assemble(A);
            if (P != NULL) {
                op_mat_zero(P);
                op_par_loop(diag_block_decoupled, "diag_block_decoupled", ctx.nodes,
                            op_arg_dat(ctx.nbnd, -1, OP_ID, 1, "int", OP_READ),
                            op_arg_mat(P, -1, OP_ID, -1, OP_ID, NV, "double", OP_INC));
                op_par_loop(edge_blocks_decoupled, "edge_blocks_decoupled", ctx.edges,
                            op_arg_mat(P, 0, ctx.edge_nodes, 0, ctx.edge_nodes, NV, "double", OP_INC),
                            op_arg_mat(P, 0, ctx.edge_nodes, 1, ctx.edge_nodes, NV, "double", OP_INC),
                            op_arg_mat(P, 1, ctx.edge_nodes, 0, ctx.edge_nodes, NV, "double", OP_INC),
                            op_arg_mat(P, 1, ctx.edge_nodes, 1, ctx.edge_nodes, NV, "double", OP_INC));
                op_mat_assemble(P);
            }
            if (pc == "bjacobi" || pc == "neumann")
                op_par_loop(invert_block, "invert_block", ctx.nodes,
                            op_arg_mat(A, -1, OP_ID, -1, OP_ID, NV, "double", OP_READ),
                            op_arg_dat(ctx.dinv, -1, OP_ID, NV * NV, "double", OP_WRITE));
            t2 = wall();
            step_loops = t1 - t0;
            step_commit = t2 - t1;
            loops_time += step_loops;
            commit_time += step_commit;
        }

        op_par_loop(rhs, "rhs", ctx.nodes,
                    op_arg_dat(p_u, -1, OP_ID, NV, "double", OP_READ),
                    op_arg_dat(p_b, -1, OP_ID, NV, "double", OP_WRITE));
        t0 = wall();
        converged &= op_solve(solver, p_b, p_u);
        t1 = wall();
        solve_time += t1 - t0;

        int its;
        double res;
        op_solver_stats(solver, &its, &res);
        iterations += its;
        if (step < 3 || step == steps - 1)
            op_printf("step %4d: %4d iterations, residual %.3e, solve %.4f s, assembly %.2f + %.2f ms\n", step, its,
                      res, t1 - t0, step_loops * 1e3, step_commit * 1e3);
    }
    const double end = wall();

    // The discrete solution: the mode's amplitude after each step is
    // (I + dt (mu K + C))^-1 times the last, mu the discrete Laplacian's
    // eigenvalue; the continuous one decays as exp(-(2 pi^2 K + C) t).
    const double pi = 3.14159265358979323846, mu = 4.0 / h2 * (1.0 - cos(pi * h));
    Matrix step_matrix(NV * NV), generator(NV * NV), K(Kc, Kc + NV * NV), C(Cc, Cc + NV * NV);
    for (int k = 0; k < NV * NV; k++) {
        step_matrix[k] = (k % (NV + 1) == 0 ? 1.0 : 0.0) + dt * (mu * K[k] + C[k]);
        generator[k] = 2 * pi * pi * K[k] + C[k];
    }
    Matrix step_inverse = inverse(step_matrix);
    std::vector<double> discrete(w, w + NV);
    for (int s = 0; s < steps; s++)
        discrete = mat_vec(step_inverse, discrete);
    std::vector<double> continuous = mat_vec(expm(generator, steps * dt), std::vector<double>(w, w + NV));

    // The last step's true residual, through the matrix-free operator.
    matfree(p_u, p_ax, &ctx);
    double r2 = 0, b2 = 0;
    op_par_loop(residual_norms, "residual_norms", ctx.nodes,
                op_arg_dat(p_ax, -1, OP_ID, NV, "double", OP_READ),
                op_arg_dat(p_b, -1, OP_ID, NV, "double", OP_READ),
                op_arg_gbl(&r2, 1, "double", OP_INC),
                op_arg_gbl(&b2, 1, "double", OP_INC));

    double e2 = 0, emax = 0, c2 = 0, cmax = 0;
    op_par_loop(error_norms, "error_norms", ctx.nodes,
                op_arg_dat(p_u, -1, OP_ID, NV, "double", OP_READ),
                op_arg_dat(p_xy, -1, OP_ID, 2, "double", OP_READ),
                op_arg_gbl(discrete.data(), NV, "double", OP_READ),
                op_arg_gbl(&e2, 1, "double", OP_INC),
                op_arg_gbl(&emax, 1, "double", OP_MAX));
    op_par_loop(error_norms, "error_norms", ctx.nodes,
                op_arg_dat(p_u, -1, OP_ID, NV, "double", OP_READ),
                op_arg_dat(p_xy, -1, OP_ID, 2, "double", OP_READ),
                op_arg_gbl(continuous.data(), NV, "double", OP_READ),
                op_arg_gbl(&c2, 1, "double", OP_INC),
                op_arg_gbl(&cmax, 1, "double", OP_MAX));

    op_printf("n %d, NV %d, unknowns %d, steps %d, dt %g: %d iterations, %s\n", n, NV, gnodes * NV, steps, dt,
              iterations, converged ? "converged" : "NOT converged");
    op_printf("true relative residual: %.3e\n", sqrt(r2 / b2));
    if (init == "mode") {
        op_printf("error vs discrete solution:   max %.3e, rms %.3e\n", emax, sqrt(e2 / (gnodes * NV)));
        op_printf("error vs continuous solution: max %.3e, rms %.3e\n", cmax, sqrt(c2 / (gnodes * NV)));
    }
    op_printf("time: total %.4f s, assembly loops %.4f s, commit %.4f s, solve %.4f s\n", end - start, loops_time,
              commit_time, solve_time);

    if (atoi(option(argc, argv, "-timing", "0")))
        op_timing_output();
    op_solve_exit();
    op_exit();
    return converged ? 0 : 1;
}
