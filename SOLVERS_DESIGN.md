# Linear solvers in OP2: design

Goal: implicit solvers in OP2 applications without exporting the system from
OP2 data structures into a solver library. Par loops write matrix entries and
right-hand sides into the memory the solver library reads (on the GPU when the
loops run there), the library solves, and par loops read the solution in place.
PETSc first, then hypre and Ginkgo, behind one API.

## 1. What Hydra does now, and what changes

Hydra assembles its Jacobian into its own edge-based storage (two off-diagonal
blocks per edge, a diagonal block per node), then a GPU kernel exports A and b
into the solver's CSR arrays, the library solves, and x is copied back. That
costs a second copy of the matrix values, one pass over them per assembly, and
two vector copies per solve, plus code that knows the library's layout.

Here the matrix is a first-class OP2 object. A par loop names a matrix block as
it names a dat: `op_arg_mat(A, 0, edge_nodes, 1, edge_nodes, ...)` is the
(row node 0, column node 1) block of each edge's contribution. The runtime
resolves it to the block's place in the matrix storage, so the kernel writes
`A[r * d + c] += ...` straight into it, with the usual OP2 handling of
increment conflicts (atomics, colouring, chunked staging). Vectors are op_dats
the library wraps in place.

## 2. API (C, in `op_solve.h`)

```c
/* After op_init (and op_partition, under MPI); lib is "petsc", "hypre" or
   "ginkgo", or NULL for OP_SOLVE_LIB or the first one built in. argc/argv
   reach the library's options database (PETSc). */
void op_solve_init(int argc, char **argv, char const *lib);
void op_solve_exit(void);

/* The block sparsity of a matrix with rows on `rows` and columns on `cols`:
   for each pair k, every (rowmaps[k][e][i], colmaps[k][e][j]) block over the
   iteration set the two maps share. OP_ID (NULL) is the identity of its side's
   set. The diagonal is always in it when rows == cols. */
op_sparsity op_decl_sparsity(op_set rows, op_set cols, int npairs,
                             op_map *rowmaps, op_map *colmaps, char const *name);

/* dim x dim blocks of `type` ("double" only for now) on the sparsity. */
op_mat op_decl_mat(op_sparsity sparsity, int dim, char const *type, char const *name);

/* Block (rowmap[e][rowidx], colmap[e][colidx]) of A, dim x dim, row-major:
   the kernel sees dim * dim values. acc is OP_INC, OP_WRITE (only through
   OP_ID x OP_ID, a set's own diagonal block) or OP_READ. */
op_arg op_arg_mat(op_mat A, int rowidx, op_map rowmap, int colidx, op_map colmap,
                  int dim, char const *type, op_access acc);

void op_mat_zero(op_mat A);      /* every entry to zero, before an assembly */
void op_mat_assemble(op_mat A);  /* the entries are final: commit, new state */

/* A solver for A; options are the library-neutral settings below, then the
   library's own (PETSc's database options with prefix "<name>_"). */
op_solver op_decl_solver(op_mat A, char const *name, char const *options);
/* x is the initial guess; b and x are dim-dim double dats on A's rows/cols. */
int op_solve(op_solver s, op_dat b, op_dat x);  /* 1 when converged */
void op_solver_stats(op_solver s, int *iterations, double *residual);

/* Custom preconditioning and operators (section 6). */
void op_solver_set_pc_mat(op_solver s, op_mat P);
typedef void (*op_apply_fn)(op_dat in, op_dat out, void *ctx);
void op_solver_set_pc(op_solver s, op_apply_fn apply, void *ctx);
void op_solver_set_operator(op_solver s, op_apply_fn apply, void *ctx);
```

Library-neutral options (space separated `key=value`): `method=cg|gmres|
bicgstab|fgmres`, `pc=none|jacobi|bjacobi|ilu|amg|custom`, `rtol=`, `atol=`,
`maxit=`, `restart=`, `monitor=0|1`. Anything after `--` goes to the library
untouched (PETSc options, or `key=value` settings of the hypre/Ginkgo
backends).

## 3. Sparsity and slots

The sparsity is computed once, per rank, on the host, from the maps:

- Rows are the owned row-set elements [0, rows->size). Each rank owns a
  contiguous range of global rows, offset by the exclusive prefix sum of
  sizes; a halo element's global number is its owner's, found by exchanging a
  global-id dat once.
- For each pair, every iteration element OP2 executes in an increment loop
  (owned and exec halo) and every (i, j) gives the block (I, J) with
  I = rowmap[e][i], J = colmap[e][j]. Blocks with an owned row go into the
  pattern; blocks with a halo row (from exec-halo elements, whose rows another
  rank owns and assembles itself) go to scratch slots, one each, that nothing
  reads: the same "write it, then throw it away" OP2 does for increments to
  halo elements of dats.
- Each owned row's blocks split into the diagonal part (owned columns) and the
  off-diagonal part (halo columns, by global number), as PETSc's MPIAIJ,
  hypre's ParCSR and Ginkgo's distributed matrix all store them.
- Blocks are numbered ("slots"): the diagonal part's rows in order, then the
  off-diagonal part's, then the scratch slots. Within a row, columns are
  ordered as the library wants (sorted; hypre wants the diagonal first).

The slots become ordinary OP2 objects: an internal set `nnz` (one element per
slot) and, for each pair k and row index i, an op_map slot_{k,i} from the
iteration set to `nnz` with arity colmap->dim, so slot_{k,i}[e][j] is the
slot of block (rowmap[e][i], colmap[e][j]). A matrix's values are an op_dat
on `nnz` with dim d * d. `op_arg_mat(A, i, rowmap, j, colmap, d, ...)` is then

    op_arg_dat(A->values, j, slot_{k,i}, d * d, type, acc)

and everything downstream is unchanged: the translator treats it as an
indirect dat argument, the plans colour by slot (two edges conflict only on a
shared diagonal block), atomics, colouring and the chunked staging all apply,
and the JIT specialises as for any loop. Matrices on one sparsity share its
slot maps.

Under MPI the internal set and maps are declared after partitioning, with no
halos; the MPI layer gets `op_mpi_declare_local` to give them empty halo
entries.

## 4. Storage layouts

The kernel always sees a block as d * d values, row-major. Where they live:

| layout | block (I, J) entry (r, c) | used for |
|---|---|---|
| blocked | values[slot * d * d + r * d + c] | v1, every library |
| row-strided (later) | part[off(slot) + r * L(I) + c], L(I) = d * (blocks in row I) | scalar CSR in place |

Blocked is CSR when d == 1, so a scalar matrix is the library's CSR arrays: a
zero-copy view where the library takes user arrays (hypre's ParCSR, Ginkgo's
Csr, PETSc's host AIJ), and one copy into its own arrays where it does not
(PETSc on the GPU, through its COO assembly, `MatSetValuesCOO` from device
memory). For d > 1 blocked is not the libraries' scalar CSR (and Ginkgo's Fbcsr
and PETSc's BAIJ store blocks column-major), so v1 commits it with one
permuting copy, as Hydra's export does, but generated rather than hand-written.

The row-strided layout removes that copy: with each block row's scalar rows
holding the same block columns, entry (r, c) of a block is at a fixed offset
plus r times the row length, so a Ptr with a row stride addresses it and the
kernel writes the library's scalar CSR directly, d > 1 included. That needs
the strided Ptr in the templates (gpu, cpu, seq, and the staging flush), the
offset and row length beside the slot, and, for hypre, the diagonal block's
diagonal entry first in each row. It is the second step.

## 5. Vectors

b and x are op_dats on the row and column sets with dim d (AoS; an SoA dat with
d > 1 is an error for now). Their owned part, size * d values, in the dat's
memory space (the device in a GPU build), is the library vector: PETSc
`VecCreateMPI[CUDA]WithArray`/`VecPlaceArray`, hypre a ParVector over the
pointer, Ginkgo a Dense view. op_solve brings b and x to the right space as a
loop does (`op_mpi_halo_exchanges`), solves, and marks x written
(`op_mpi_set_dirtybit[_cuda]`), so the next loop exchanges its halo.

## 6. Custom preconditioners

The key question. Four mechanisms, from the libraries' own to fully custom:

1. **An assembled preconditioning matrix**: `op_solver_set_pc_mat(s, P)`,
   where P is another op_mat (say a first-order Jacobian beside a higher-order
   A, or a matrix-free A) assembled by par loops. The library builds its
   preconditioner (AMG, ILU, block Jacobi) from P. Every library supports
   (A, P) pairs.
2. **A par-loop preconditioner**: `op_solver_set_pc(s, apply, ctx)`. Each
   application of M^-1, the library calls `apply(r, z, ctx)` with r and z as
   op_dats on the row set; the user's code is ordinary op_par_loops (and halo
   exchanges, colouring, GPU execution are OP2's as always). PETSc's PCSHELL,
   hypre's SetPrecond function pointers and a Ginkgo LinOp all take a callback.
   The library's vectors have no room for OP2's halo, so r and z are OP2 dats
   (with halos) the backend copies the library vectors into and out of: two
   vector copies per application, negligible beside a preconditioner, and
   avoidable where the library allows its work vectors to be OP2's (PETSc
   through MatCreateVecs) - a later optimisation.
3. **Reading the matrix in those loops**: `op_arg_mat(..., OP_READ)` gives a
   par loop the assembled entries in solver memory, through the same maps:
   block Jacobi (read and invert the diagonal blocks into a dat, then apply),
   Gauss-Seidel or line-implicit sweeps (edge loops reading off-diagonal
   blocks), polynomial preconditioners (matrix-vector products as edge loops).
   This is what makes (2) useful for Hydra's own preconditioners.
4. **A par-loop operator**: `op_solver_set_operator(s, apply, ctx)` replaces
   y = A x by par loops (matrix-free or Jacobian-free Newton-Krylov), with the
   preconditioner from (1) or (2). PETSc MATSHELL, Ginkgo LinOp; hypre's Krylov
   solvers take a matvec through their generic interface.

Callbacks may call op_solve themselves (an AMG solve on a scalar sub-system
inside a block preconditioner), so field-split and multilevel preconditioners
compose from the same pieces. Nothing else is needed: the preconditioner is
ordinary OP2 code with matrix access, and the library only sees an operator.

## 7. Code layout

- `op2/include/op_solve.h`: the C API.
- `op2/include/op2/solve/`: the objects (`op::solve::Sparsity`, `Matrix`,
  `Solver`) and the backend interface (`op::solve::Backend`: make a matrix
  from a Matrix's storage, commit, make a solver, solve; vectors from dats).
- `op2/src/solve/`: sparsity and slot construction (hardware-free), the API,
  and one file per library (`petsc.cpp`, `hypre.cpp`, `ginkgo.cpp`), compiled
  into each C library variant when that library is enabled
  (`OP2_ENABLE_PETSC`, `OP2_ENABLE_HYPRE`, `OP2_ENABLE_GINKGO`), the memory
  space following the variant (host, or the device for the GPU variants).
- Translator: `op_arg_mat` parsed by the C++ front end into an indirect dat
  argument on the matrix, through a slot map named by (matrix, row map, row
  index, column map).
- `apps/c/implicit`: the proxy app (section 8).

## 8. Proxy app

`apps/c/implicit`: backward Euler for coupled heat equations,
u_t = div(K grad u) - C u, with d components (K, C d x d SPD; d = 1 is the
scalar heat equation), on a structured grid stored as an unstructured mesh
(nodes, edges between neighbours, randomly renumbered so partitioning and
colouring see an unstructured problem), Dirichlet zero boundaries. With
u0 = w sin(pi x) sin(pi y) the exact solution is
exp(-(2 pi^2 K + C) t) w sin(pi x) sin(pi y), so the error at the final time
checks the whole pipeline (second order in h). Per step: assemble
(h^2/dt I + h^2 C) diagonal blocks in a node loop and -K / +K blocks in an edge
loop, the right-hand side in a node loop, solve, update. Options pick the
library, method and preconditioner, including a par-loop block-Jacobi
preconditioner reading the diagonal blocks with op_arg_mat(OP_READ).

## 9. Out of scope for now

Rectangular blocks and rectangular matrices (rows != cols sets), single
precision, Fortran bindings (a thin layer over the C API; Hydra needs it next),
HIP (the same code with hipsparse PETSc and HIP hypre/Ginkgo builds), matrices
whose pattern changes after declaration.
