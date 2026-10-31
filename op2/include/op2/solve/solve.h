#pragma once

// The solver layer's objects, and the interface each solver library's
// backend implements. Internal: applications include op_solve.h.

#include <op_solve.h>

#include <mpi.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#if defined(OP2_CUDA) || defined(OP2_HIP)
#define OP2_SOLVE_DEVICE 1
#else
#define OP2_SOLVE_DEVICE 0
#endif

namespace op::solve {

// Where this library variant's loops keep dats, and so where the solver
// library works on them.
inline constexpr bool on_device = OP2_SOLVE_DEVICE;

// How a row's blocks are ordered within each part of the matrix.
enum class Order { sorted, diagonal_first };

struct Options {
    std::string method = "gmres";
    std::string pc = "jacobi";
    double rtol = 1e-8;
    double atol = 0.0;
    int maxit = 1000;
    int restart = 30;
    bool monitor = false;
    std::string library;
};

Options parse_options(char const *text);

// A library's matrix over an op_mat's values.
class Matrix {
public:
    virtual ~Matrix() = default;
    virtual void assemble() = 0;
};

class Solver {
public:
    virtual ~Solver() = default;
    virtual bool solve(op_dat b, op_dat x) = 0;

    int iterations = 0;
    double residual = 0.0;
};

class Library {
public:
    virtual ~Library() = default;
    virtual Order order() const = 0;
    virtual std::unique_ptr<Matrix> matrix(op_mat A) = 0;
    virtual std::unique_ptr<Solver> solver(op_solver s) = 0;
};

std::unique_ptr<Library> make_petsc(int argc, char **argv);
std::unique_ptr<Library> make_hypre(int argc, char **argv);
std::unique_ptr<Library> make_ginkgo(int argc, char **argv);

[[noreturn]] void fail(char const *format, ...);

// OP2's communicator, or MPI_COMM_SELF in a single-node variant.
MPI_Comm comm();

// A dat's memory in the space this variant's loops use.
char *memory(op_dat dat);
void copy(void *dst, const void *src, std::size_t bytes);
void zero(void *dst, std::size_t bytes);

// Bring direct args on set into the loops' space, as a loop would, and mark
// the written ones after.
void enter(op_set set, op_arg *args, int nargs);
void leave(op_arg *args, int nargs);

// z = apply(r) for a library: r and z are the library's vectors over the
// owned rows, run through the solver's dats, which have room for halos.
void apply_through(op_solver s, op_apply_fn apply, void *ctx, const double *r, double *z);

} // namespace op::solve

struct op_sparsity_core {
    struct Pair {
        op_map rowmap;          // null for OP_ID
        op_map colmap;
        op_set from;
        int rowdim;
        int coldim;
        std::vector<op_map> slots;  // by row index, to the slots of its blocks
    };

    op_set rows;
    op_set cols;
    std::string name;
    std::vector<Pair> pairs;

    // One element per slot: the diagonal part's blocks, the off-diagonal
    // part's, then scratch slots for blocks of rows another rank owns.
    op_set nnz = nullptr;
    int diag_blocks = 0;
    int offd_blocks = 0;
    int scratch_blocks = 0;

    // Block CSR of each part: diagonal columns local, off-diagonal ones into
    // garray, the global block columns ascending.
    std::vector<int> diag_ptr, diag_col;
    std::vector<int> offd_ptr, offd_col;
    std::vector<idx_g_t> garray;

    idx_g_t row_start = 0;
    idx_g_t col_start = 0;
    idx_g_t global_rows = 0;
    idx_g_t global_cols = 0;
};

struct op_mat_core {
    op_sparsity sparsity;
    int dim;
    std::string name;
    op_dat values;              // on the slots, dim * dim each, row major
    int state = 0;              // assemblies so far
    std::unique_ptr<op::solve::Matrix> lib;
};

struct op_solver_core {
    op_mat A;
    op_mat P = nullptr;
    std::string name;
    op::solve::Options options;
    op_apply_fn pc = nullptr;
    void *pc_ctx = nullptr;
    op_apply_fn op = nullptr;
    void *op_ctx = nullptr;
    op_dat in = nullptr;        // the callbacks' dats, on the rows
    op_dat out = nullptr;
    std::unique_ptr<op::solve::Solver> lib;
};
