#pragma once

// The solver layer's objects, the matrix storage loops share with a solver
// library, and the interface each library's backend implements. Internal:
// applications include op_solve.h.

#include <op_solve.h>

#include <op2/rt/abi/matrix.h>

#include <mpi.h>

#include <array>
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

namespace mat = op::rt::abi::mat;

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

// The layer's variables, which op_init reads (read_env) with the rest.
struct Settings {
    std::string lib;               // OP_SOLVE_LIB
    std::string format;            // OP_SOLVE_FORMAT
    bool memory = false;           // OP_SOLVE_MEMORY
    int hypre_pool = 0;            // OP_SOLVE_HYPRE_POOL, MiB
};

extern Settings settings;
void read_env();

// A library's matrix over an op_mat's storage.
class Matrix {
public:
    virtual ~Matrix() = default;
    // The loops have written the storage: the library takes the values.
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
    // The library's matrix in format (its default when empty), over storage
    // it sets up: the layout of the owned slots, and the regions they are in
    // (lend, borrow).
    virtual std::unique_ptr<Matrix> matrix(op_mat A, const std::string& format) = 0;
    virtual std::unique_ptr<Solver> solver(op_solver s) = 0;
};

std::unique_ptr<Library> make_petsc(int argc, char **argv);
std::unique_ptr<Library> make_hypre(int argc, char **argv);
std::unique_ptr<Library> make_ginkgo(int argc, char **argv);
std::unique_ptr<Library> make_amgx(int argc, char **argv);
std::unique_ptr<Library> make_rocalution(int argc, char **argv);
std::unique_ptr<Library> make_mock(int argc, char **argv);

[[noreturn]] void fail(char const *format, ...);

// OP2's communicator, or MPI_COMM_SELF in a single-node variant.
MPI_Comm comm();

// A dat's memory in the space this variant's loops use.
char *memory(op_dat dat);
void copy(void *dst, const void *src, std::size_t bytes);
void zero(void *dst, std::size_t bytes);
// From the loops' space to the host.
void download(void *dst, const void *src, std::size_t bytes);

// An int array in the loops' space, and its release.
int *upload(const std::vector<int>& values);
void release(void *memory);

// With OP_SOLVE_MEMORY=1, what a stage leaves in use: device memory in a GPU
// variant, and the layer's own arrays.
void report_memory(char const *stage, std::size_t own = 0);

// Bring direct args on set into the loops' space, as a loop would, and mark
// the written ones after.
void enter(op_set set, op_arg *args, int nargs);
void leave(op_arg *args, int nargs);

// z = apply(r) for a library: r and z are the library's vectors over the
// owned rows, run through the solver's dats, which have room for halos.
void apply_through(op_solver s, op_apply_fn apply, void *ctx, const double *r, double *z);

// --- Matrix storage -----------------------------------------------------------

// A slot's layout entry (op2/rt/abi/matrix.h).
using Entry = std::array<int, mat::entry>;

// The memory a region of a matrix's values is in, and whether the layer
// frees it.
struct Region {
    char *data = nullptr;
    std::size_t bytes = 0;
    bool owned = false;
};

// Region `region` of A: zeroed memory in the loops' space the layer
// allocates and frees (lend), or the library's own (borrow).
char *lend(op_mat A, int region, std::size_t bytes);
void borrow(op_mat A, int region, void *data, std::size_t bytes);

// Where entry (r, c) of a block with layout entry e is in its region, as
// BlockPtr finds it, for a diagonal block or not.
std::size_t position(const Entry& e, int d, int r, int c, bool diagonal);

// --- Layouts: where a part's blocks are in a library's format --------------
//
// Each builder sets the layout entries of one part's slots (the diagonal
// part's blocks, with owned columns, or the off-diagonal part's) for its
// values in region from offset, and returns what the library needs of it.
// Scalar columns are J * d + c, a diagonal part's J local, an off-diagonal
// part's an index into the sparsity's garray.

enum class Part { diag, offd };

// How a scalar row's diagonal entry is kept: among the others (none), first
// in its row (first), or apart, at diagonal_offset + the row (separate).
enum class Diagonal { none, first, separate };

// A block's entries.
enum class Major { row, column };

// A part's scalar rows and columns, and its blocks.
int part_rows(op_mat A);
int part_cols(op_mat A, Part part);
int part_blocks(op_mat A, Part part);

// Scalar CSR: block row I's d rows each hold its blocks' columns in order.
// COO keeps the same order.
struct ScalarCsr {
    std::vector<int> ptr, col;
};
ScalarCsr csr_layout(op_mat A, Part part, int region, int offset, Diagonal diagonal = Diagonal::none,
                     int diagonal_offset = 0);

// Block CSR: the part's blocks in order, d * d values each, row or column
// major; the diagonal part's diagonal blocks apart at diagonal_offset +
// I * d * d when separate_diagonal (then left out of ptr and col).
struct BlockCsr {
    std::vector<int> ptr, col;
};
BlockCsr bcsr_layout(op_mat A, Part part, int region, int offset, Major major, bool separate_diagonal = false,
                     int diagonal_offset = 0);

// ELL: the part's scalar rows padded to the longest (width), entry k of row
// i at k * rows + i; padding has column pad.
struct Ell {
    int width = 0;
    std::vector<int> col;
};
Ell ell_layout(op_mat A, Part part, int region, int offset, int pad);

// SELL-P: rows in slices of `slice` rows, each padded to its longest rounded
// up to a multiple of `factor`, entry k of row i at (sets[i / slice] + k) *
// slice + i % slice.  slice must be a multiple of d.
struct Sell {
    int slice = 0;
    std::vector<int> lengths, sets, col;
};
Sell sell_layout(op_mat A, Part part, int region, int offset, int slice, int factor, int pad);

// Block ELL: block rows padded to the widest (width), block k of block row I
// at block k * rows + I with its values contiguous (row or column major), or
// interleaved: entry e of the block (row or column major) at (k * d * d + e)
// * rows + I.  Padding has block column pad.
struct BlockEll {
    int width = 0;
    std::vector<int> col;
};
BlockEll bell_layout(op_mat A, Part part, int region, int offset, Major major, bool interleaved, int pad);

// A part a library laid out itself as scalar CSR, ptr and col as it holds
// them (host copies, its columns as csr_layout's): each block found by its
// first entry's column, its rows checked to be evenly spaced.
void csr_located(op_mat A, Part part, int region, const std::vector<int>& ptr, const std::vector<int>& col);

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

    // Whether an element reaches a diagonal block through other indices than
    // the same map's same one, which generated code takes as any block: no
    // format may then move a diagonal block's diagonal entries.
    bool diagonal_elsewhere = false;
};

struct op_mat_core {
    op_sparsity sparsity;
    int dim;
    std::string name;
    std::string format;

    // What loops bind (op2/rt/abi/matrix.h), over the regions and layout.
    op_mat_storage view{};
    op::solve::Region regions[op::solve::mat::regions];
    std::vector<op::solve::Entry> layout;
    int *layout_d = nullptr;

    // The blocks untranslated loops reach, one after another, and whether
    // they hold increments the storage has yet to take.
    std::vector<double> untranslated;
    bool pending = false;

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
