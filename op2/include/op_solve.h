#pragma once

// Linear solvers on OP2 data: matrices whose blocks par loops write and read
// through op_arg_mat, in the solver library's own storage, solved by the
// library (PETSc, hypre, Ginkgo, AmgX, rocALUTION) in the memory the loops
// run in. See SOLVERS_DESIGN.md.

#include <op_lib_core.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct op_sparsity_core *op_sparsity;
typedef struct op_mat_core *op_mat;
typedef struct op_solver_core *op_solver;

/* y = op(x) for a par-loop operator, or z = M^-1 r for a par-loop
   preconditioner: in and out are dats on the matrix's column and row sets. */
typedef void (*op_apply_fn)(op_dat in, op_dat out, void *ctx);

/* After op_init, and op_partition under MPI. lib names a library built in
   ("petsc", "hypre", "ginkgo", "amgx", "rocalution", "mock"), or is NULL for
   OP_SOLVE_LIB, else the first built in. argc and argv reach the library's own
   options (PETSc's database). */
void op_solve_init(int argc, char **argv, char const *lib);
void op_solve_exit(void);

/* The block pattern of a matrix with rows on `rows` and columns on `cols`: for
   each pair k, block (rowmaps[k][e][i], colmaps[k][e][j]) for every element e
   of the set the two maps share and every i, j. OP_ID is the identity of its
   side's set. With rows == cols the diagonal is always in it. */
op_sparsity op_decl_sparsity(op_set rows, op_set cols, int npairs, op_map *rowmaps,
                             op_map *colmaps, char const *name);

/* dim x dim blocks of type ("double") on the pattern, zero, stored in the
   library's format (a name it knows, such as "csr", "bcsr" or "ell"), or NULL
   for OP_SOLVE_FORMAT, else the library's default. */
op_mat op_decl_mat(op_sparsity sparsity, int dim, char const *type, char const *name,
                   char const *format);

/* Block (rowmap[e][rowidx], colmap[e][colidx]) of A as dim * dim values, row
   major, in a loop over the maps' set; OP_ID with index -1 for the element
   itself. acc is OP_INC or OP_READ. */
op_arg op_arg_mat(op_mat A, int rowidx, op_map rowmap, int colidx, op_map colmap, int dim,
                  char const *type, op_access acc);

void op_mat_zero(op_mat A);
/* After the loops that write A and before it is solved with or read: the
   values are final, so the library takes them and rebuilds what depends on
   them (its preconditioner). */
void op_mat_assemble(op_mat A);

/* A solver for A x = b. options: "key=value" settings (method, pc, rtol, atol,
   maxit, restart, monitor), then after "--" the library's own; PETSc also
   reads its database under the prefix "<name>_". */
op_solver op_decl_solver(op_mat A, char const *name, char const *options);

/* Precondition with P (assembled by par loops) instead of A. */
void op_solver_set_pc_mat(op_solver s, op_mat P);
/* Precondition with apply(r, z, ctx), par loops computing z = M^-1 r. */
void op_solver_set_pc(op_solver s, op_apply_fn apply, void *ctx);
/* Apply A by apply(x, y, ctx), par loops computing y = A x; the matrix given to
   op_decl_solver then serves only the preconditioner. */
void op_solver_set_operator(op_solver s, op_apply_fn apply, void *ctx);

/* x holds the initial guess and receives the solution; b and x are dim-dim
   "double" dats on A's row and column sets. Returns 1 when converged. */
int op_solve(op_solver s, op_dat b, op_dat x);
void op_solver_stats(op_solver s, int *iterations, double *residual);

#ifdef __cplusplus
}
#endif
