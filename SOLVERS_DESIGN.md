# Linear solvers in OP2: design

Goal: implicit solvers in OP2 applications without exporting the system from
OP2 data structures into a solver library. Par loops write matrix entries
into the solver library's own storage, in its own format (on the GPU when the
loops run there), the library solves, and par loops read the solution in
place. PETSc, hypre, Ginkgo, AmgX and rocALUTION behind one API, and a mock
library that owns its storage as a closed-source one (SPLISS) does.

## 1. What Hydra does now, and what changes

Hydra assembles its Jacobian into its own edge-based storage (two off-diagonal
blocks per edge, a diagonal block per node), then a GPU kernel exports A and b
into the solver's CSR arrays, the library solves, and x is copied back. That
costs a second copy of the matrix values, one pass over them per assembly, and
two vector copies per solve, plus code that knows the library's layout.

Here the matrix is a first-class OP2 object. A par loop names a matrix block as
it names a dat: `op_arg_mat(A, 0, edge_nodes, 1, edge_nodes, ...)` is the
(row node 0, column node 1) block of each edge's contribution. The runtime
resolves it to the block's place in the matrix's storage, wherever and however
the library keeps it, so the kernel writes `A[r * d + c] += ...` straight into
it, with the usual OP2 handling of increment conflicts (atomics, colouring,
chunked staging). Vectors are op_dats the library wraps in place, or copies.

## 2. API (C, in `op_solve.h`)

```c
/* After op_init (and op_partition, under MPI). */
void op_solve_init(void);
void op_solve_exit(void);

/* The configuration below: JSON, or a path to a file of it. */
void op_solve_set_config(char const *config);

/* The block sparsity of a matrix with rows on `rows` and columns on `cols`:
   for each pair k, every (rowmaps[k][e][i], colmaps[k][e][j]) block over the
   iteration set the two maps share. OP_ID (NULL) is the identity of its side's
   set. The diagonal is always in it when rows == cols. */
op_sparsity op_decl_sparsity(op_set rows, op_set cols, int npairs,
                             op_map *rowmaps, op_map *colmaps, char const *name);

/* dim x dim blocks of `type` ("double" only for now) on the sparsity, in the
   library and storage format (section 4) its configuration names. */
op_mat op_decl_mat(op_sparsity sparsity, int dim, char const *type, char const *name);

/* Block (rowmap[e][rowidx], colmap[e][colidx]) of A, dim x dim, row-major as
   the kernel indexes it. acc is OP_INC or OP_READ. */
op_arg op_arg_mat(op_mat A, int rowidx, op_map rowmap, int colidx, op_map colmap,
                  int dim, char const *type, op_access acc);

void op_mat_zero(op_mat A);      /* every entry to zero, before an assembly */
void op_mat_assemble(op_mat A);  /* the entries are final: the library takes them */

/* A solver for A, in A's library, as its configuration says. */
op_solver op_decl_solver(op_mat A, char const *name);
/* x is the initial guess; b and x are dim-dim double dats on A's rows/cols. */
int op_solve(op_solver s, op_dat b, op_dat x);  /* 1 when converged */
void op_solver_stats(op_solver s, int *iterations, double *residual);

/* Custom preconditioning and operators (section 6). */
void op_solver_set_pc_mat(op_solver s, op_mat P);
typedef void (*op_apply_fn)(op_dat in, op_dat out, void *ctx);
void op_solver_set_pc(op_solver s, op_apply_fn apply, void *ctx);
void op_solver_set_operator(op_solver s, op_apply_fn apply, void *ctx);
```

### Configuration

A JSON configuration names each matrix's library and storage and configures
each solver, in each library's own terms: OP2 maps no generic settings onto
the libraries. It comes from `op_solve_set_config` (JSON text, or a path when
it does not start with `{`) and `OP_SOLVE_CONFIG` (the same two forms), each
merged over those before it (JSON merge patch, RFC 7386), the variable last
so a run can override the application. Rank 0 reads files; the layer reads
the whole at the first `op_decl_mat`, after which it is fixed.

```json
{
  "library": "hypre",
  "libraries": { "hypre": { "UmpireDevicePoolSize": 67108864 } },
  "matrices": { "A": { "library": "ginkgo", "format": "fbcsr" } },
  "solvers": {
    "heat": {
      "hypre": { "solver": "PCG", "Tol": 1e-8, "precond": "BoomerAMG",
                 "BoomerAMG": { "CoarsenType": 10, "StrongThreshold": 0.5 } },
      "ginkgo": { "type": "solver::Cg",
                  "criteria": { "iteration": 1000, "relative_residual_norm": 1e-8 },
                  "preconditioner": { "type": "preconditioner::Jacobi" } }
    }
  }
}
```

- A matrix is in its entry's `library`, else the top-level `library`, else
  the first library built in (not the mock); the rest of its entry is that
  library's, its storage. Each library is started the first time a matrix
  asks for it, with its `libraries` section, so one run can use several.
- A solver runs in its matrix's library with that library's section of its
  entry, so one file can hold a setup for each library a solver may run
  with. A preconditioning matrix must be in the same library.
- Each backend maps its own sections onto its library behind the same three
  calls (`make_<library>(config)`, `Library::matrix(A, config)`,
  `Library::solver(s, config)`) and fails on anything it does not know; PETSc
  must use every option it is given.
- What the layer adds is the same for every library: a par-loop
  preconditioner (`op_solver_set_pc`) replaces the library's, and the
  solver's section then names none; a solver with no section for its
  library runs that library's default (GMRES with Jacobi to 1e-8, in its own
  terms), with a note when the configuration has solvers at all.

| library | matrix | solver | library |
|---|---|---|---|
| PETSc | `format`: `aij` (host default) or `coo` (device default); `owned` | PETSc options without `-`, put under the solver's prefix `<name>_` | global options, set before `PetscInitialize` |
| hypre | `format`: `csr` | `solver` (`PCG`, `GMRES`, `FlexGMRES`, `BiCGSTAB`) and its setters by name (`Tol`, `MaxIter`, `KDim`, ...); `precond` (`BoomerAMG`, `ILU`, `DiagScale`) and an object of its setters under its name | its global setters (`UmpireDevicePoolSize`, `SpMVUseVendor`, ...) |
| Ginkgo | `format`: `csr` (with a `strategy`: `cusparse`, `classical`, `merge_path`), `coo`, `ell`, `sellp`, `fbcsr` | a Ginkgo file configuration (`gko::config`); a preconditioner built apart (from P, or the par-loop one) is its `generated_preconditioner` | none |
| AmgX | `format`: `bcsr` (default for 1, 3 and 4 components; `external_diagonal`) or `csr` | an AmgX configuration (`config_version` 2); its outer solver keeps the residual history the layer reports from | `print`, `resources` |
| rocALUTION | `format`: `csr`, `mcsr`, `bcsr`, `coo`, `ell` | `solver` (`CG`, `BiCGStab`, `GMRES`, `FGMRES`), `Init`'s `abs_tol`, `rel_tol`, `div_tol`, `max_iter`, `basis_size`, `verbose`, `preconditioner` (`Jacobi`, `PairwiseAMG`, `none`) | none |
| mock | `format`: `compact` or `bell`; `major`: `row` or `column`; `interleaved` | `solver` (`cg`), `rtol`, `atol`, `maxit`, `monitor`, `preconditioner` (`jacobi`, `none`) | none |

`OP_SOLVE_MEMORY=1` reports memory per stage and each matrix's storage form.

## 3. Sparsity, slots and matrix arguments

The sparsity is computed once, per rank, on the host, from the maps:

- Rows are the owned row-set elements [0, rows->size). Each rank owns a
  contiguous range of global rows, offset by the exclusive prefix sum of
  sizes; a halo element's global number is its owner's, found by exchanging a
  global-id dat once (64-bit numbers).
- For each pair, every iteration element OP2 executes in an increment loop
  (owned and exec halo) and every (i, j) gives the block (I, J) with
  I = rowmap[e][i], J = colmap[e][j]. Blocks with an owned row go into the
  pattern; blocks with a halo row (from exec-halo elements, whose rows another
  rank owns and assembles itself) go to scratch slots, one each, that nothing
  reads: the same "write it, then throw it away" OP2 does for increments to
  halo elements of dats.
- Each owned row's blocks split into the diagonal part (owned columns) and the
  off-diagonal part (halo columns, by global number), as PETSc's MPIAIJ,
  hypre's ParCSR, Ginkgo's distributed matrix, rocALUTION's GlobalMatrix and
  SPLISS's local and non-local parts all store them.
- Blocks are numbered ("slots"): the diagonal part's rows in order, then the
  off-diagonal part's, then the scratch slots. Within a row, columns are
  sorted, whichever library a matrix on the sparsity is in: a format that
  orders a row otherwise (hypre's diagonal first) says so in its layout.

The slots become ordinary OP2 objects: an internal set `nnz` (one element per
slot) and, for each pair k and row index i, an op_map slot_{k,i} from the
iteration set to `nnz`, so slot_{k,i}[e][j] is the slot of block
(rowmap[e][i], colmap[e][j]). Under MPI they are declared after partitioning,
with no halos (`op_mpi_local_objects`).

`op_arg_mat(A, i, rowmap, j, colmap, d, ...)` is an `OP_ARG_MAT`: an op_arg
whose map is slot_{k,i} with index j and whose `mat` points at the matrix's
storage (`op_mat_storage`, `op2/rt/abi/matrix.h`). There is no op_dat: the
loop runtime (plans, colouring, chunked staging, the MPI exec-halo rule) takes
a target as a dat or a matrix (`op::rt::loop::target`, `target_set`, the slot
set), so every strategy assembles, and MPI runs a matrix-incrementing loop
over the exec halo but never exchanges a matrix. The translator gives each
matrix argument an `op::rt::kernel::BlockPtr<T, d>` kernel parameter, built by
`mat_block` from the matrix's regions and layout (section 4); the generated
stubs bind the storage's regions and layout from `args[i].mat`.

## 4. Storage

A matrix's values are in up to three regions: say a library's local and
non-local parts, and OP2's scratch for blocks of halo rows, which follows the
values of a region 0 OP2 lends and is otherwise region 2. Who allocates a
region is the backend's choice:

- **lend**: OP2 allocates it (`op::solve::lend`) and the library views it as
  its own arrays (hypre's ParCSR data; Ginkgo's Csr, Coo, Ell and Fbcsr
  values; PETSc's host AIJ parts);
- **borrow**: the library allocates it in its own format and OP2 takes its
  pointer (`op::solve::borrow`): the mock library, rocALUTION, Ginkgo's
  Sellp, PETSc's owned AIJ;
- **copy**: OP2 lends a region in the order the library takes values in, and
  hands it over at each assembly: AmgX (`AMGX_matrix_replace_coefficients`),
  PETSc's `coo` (`MatSetValuesCOO`, PETSc's default on the device).

The layout holds, per slot, four ints: entry (r, c) of the slot's block is at
offset + r * row_stride + c * col_stride in region (aux & 3). That covers
scalar CSR rows (row stride the row's length, column stride 1), contiguous
blocks row or column major (d, 1 or 1, d), ELL and SELL-P (row stride 1,
column stride the rows or the slice), and block ELL with its entries
interleaved across block rows for coalescing (d * rows, rows). A diagonal
block may move its diagonal entries by a rule in aux: diagonal first in each
row (hypre), or apart at col_stride + r (MCSR, rocALUTION). Only a diagonal
argument (the same map and index on both sides) is typed
`BlockPtr<T, d, true>` and checks the rule, so other blocks pay nothing; a
format with a rule rejects a sparsity that reaches a diagonal block any other
way.

Kernels find a slot's block by the least general form its matrix's layout
fits (`op_mat_storage::form`), which `op_decl_mat` works out from the layout
once the backend has set it: where the blocks are, **contiguous** (block s
at s * d * d of region 0, row major: the layout is not read; PETSc's COO
order, scalar CSR at d = 1), **one region** (through the layout, every slot
in region 0: every format OP2 lends, its scratch included) or **any region**
(the slot's own: a library's local and non-local arrays apart, as the mock,
rocALUTION, Ginkgo's Sellp and PETSc's owned AIJ have them on more than
one rank); whether every block's column stride is 1; and the one diagonal
rule its slots may have. A JIT kernel bakes the form in, as it does a dat's
run-time dimension (`abi::mat_form`), so what the layout does not need
compiles out: one base and no region choice for the libraries that need no
separate parts, constant offsets along rows, no rule tests where there is
no rule. An offline or CPU kernel takes the form as an argument and
branches on it, the same way for every element. A new library needs
nothing for this: its layout determines its form.

The backends lay out their formats with the layer's builders
(`op2/src/solve/op_solve_layout.cpp`), each setting a part's slots' entries
and returning the format's index arrays: `csr_layout` (with a diagonal rule;
COO keeps its order), `bcsr_layout` (row or column major, the diagonal blocks
apart), `ell_layout`, `sell_layout`, `bell_layout` (contiguous or
interleaved), and `csr_located`, which finds each block in CSR arrays the
library laid out itself and checks its rows are evenly spaced. A library that
lays out its matrix by its own rules is checked against them: rocALUTION
converts the pattern itself and its index arrays must equal the builder's.

| library | formats (default first) | regions |
|---|---|---|
| hypre | csr (diagonal first) | lend |
| Ginkgo | csr (with a strategy), coo, ell, fbcsr; sellp | lend; borrow |
| PETSc | aij (host), coo (device default); owned aij | lend; copy; borrow |
| AmgX | bcsr (the diagonal blocks apart on one rank), csr | copy |
| rocALUTION | csr, mcsr, bcsr, coo, ell | borrow |
| mock | compact, bell (row or column major, bell interleaved) | borrow, local and non-local apart |

Untranslated loops (op_seq.h) know no layout: they reach a matrix's blocks
one after another in a buffer the matrix allocates the first time one needs
it (`op_mat_untranslated`), which `op_mat_assemble` adds into the storage
through the layout (and fills from it for OP_READ). The staged slots of the
chunked strategies are contiguous BlockPtrs over shared memory; their flush
and seed reach the target through the layout, as a possibly diagonal block.

### Integrating a library that owns its storage (SPLISS)

What the backend needs from the library: the device (or host) pointers to
each part's values, that they stay put between assembly and solve, and where
block (I, J) is, by a lookup or index arrays it can read once; a way to tell
it the values changed; vectors it can wrap or copy into. A format whose blocks
are not at fixed strides (entries of one block in different, irregularly
spaced places) would need a new layout rule.

### Index widths

Offsets within a region are 32-bit (the layout's ints, and the libraries'
local CSR: PETSc's default PetscInt, HYPRE_Int, Ginkgo's int32 local index),
so a region holds at most 2^31 values, which op_decl_mat checks. At d = 6
that is 59.6M blocks, 16 GiB of values: 8.5M nodes of a hex mesh per rank,
4.0M of a tet mesh. The kernels compute each block's address from its
region's base; wider offsets were measured to cost registers (+16 on sm_90 for
a 64-bit form).

Globally, rows are numbered in 64 bits (OP2's idx_g_t), as Ginkgo's,
AmgX's and rocALUTION's global indices are. hypre needs HYPRE_ENABLE_MIXEDINT
(64-bit HYPRE_BigInt, 32-bit HYPRE_Int; its all-64-bit build has no GPU
support). PETSc's default 32-bit PetscInt caps a matrix at 2^31 global rows,
358M nodes at d = 6. The backends check both and say which build is needed.

## 5. Vectors

b and x are op_dats on the row and column sets with dim d (AoS; an SoA dat with
d > 1 is an error for now). Their owned part, size * d values, in the dat's
memory space (the device in a GPU build), is the library vector: PETSc
`VecCreateMPI[CUDA]WithArray`/`VecPlaceArray`, hypre a ParVector over the
pointer, Ginkgo a Dense view, the mock library the dat's memory itself; AmgX
and rocALUTION copy into and out of their own vectors, vector-sized copies per
solve. op_solve brings b and x to the right space as a
loop does (`op_mpi_halo_exchanges`), solves, and marks x written
(`op_mpi_set_dirtybit[_cuda]`), so the next loop exchanges its halo.

## 6. Custom preconditioners

The key question. Four mechanisms, from the libraries' own to fully custom.
PETSc, Ginkgo and the mock library take all four; hypre and rocALUTION all but
the par-loop operator; AmgX only the assembled matrix, A itself (its C API
takes no callbacks):

1. **An assembled preconditioning matrix**: `op_solver_set_pc_mat(s, P)`,
   where P is another op_mat (say a first-order Jacobian beside a higher-order
   A, or a matrix-free A) assembled by par loops. The library builds its
   preconditioner (AMG, ILU, block Jacobi) from P. The proxy's
   `-pmat decoupled` drops the components' coupling from P.
2. **A par-loop preconditioner**: `op_solver_set_pc(s, apply, ctx)`. Each
   application of M^-1, the library calls `apply(r, z, ctx)` with r and z as
   op_dats on the row set; the user's code is ordinary op_par_loops (halo
   exchanges, colouring, GPU execution are OP2's as always). PETSc's PCSHELL,
   hypre's SetPrecond function pointers and a Ginkgo LinOp take the callback.
   The library's vectors have no room for OP2's halo, so r and z are OP2 dats
   (with halos) the backend copies the library vectors into and out of: two
   vector copies per application.
3. **Reading the matrix in those loops**: `op_arg_mat(..., OP_READ)` gives a
   par loop the assembled entries in solver memory, through the same maps,
   in whichever layout the library keeps. The proxy's preconditioners: block
   Jacobi (`-pc bjacobi`: invert the diagonal blocks into a dat, apply them)
   and a Neumann series (`-pc neumann`: block Jacobi plus m matrix-vector
   products, each an edge loop reading the off-diagonal blocks and a node
   loop reading the diagonal ones).
4. **A par-loop operator**: `op_solver_set_operator(s, apply, ctx)` replaces
   y = A x by par loops (matrix-free or Jacobian-free Newton-Krylov), with the
   preconditioner from (1) or (2). PETSc MATSHELL, Ginkgo LinOp; hypre's
   ParCSR Krylov solvers would need its generic interface (not done).

Callbacks may call op_solve themselves (an AMG solve on a scalar sub-system
inside a block preconditioner), so field-split and multilevel preconditioners
compose from the same pieces. Nothing else is needed: the preconditioner is
ordinary OP2 code with matrix access, and the library only sees an operator.

## 7. Code layout

- `op2/include/op_solve.h`: the C API.
- `op2/include/op2/rt/abi/matrix.h`: a matrix's storage as generated code
  binds it, and the layout entry's contract; `op2/include/op2/rt/kernel/
  prelude.h`: `BlockPtr` and `mat_block`; `op2/src/rt/op_rt_matrix.cpp`:
  untranslated loops' way in.
- `op2/include/op2/solve/solve.h`: the layer's objects, the backend interface
  (`make_<library>`, `Library`, `Matrix`, `Solver`, each taking its part of
  the configuration; `Section` reads one strictly), storage (`lend`,
  `borrow`) and the layout builders.
- `op2/src/solve/`: sparsity and slot construction, the API, the builders,
  and one file per library (`op_solve_petsc.cpp`, `_hypre`, `_ginkgo`,
  `_amgx`, `_rocalution`, `_mock`), compiled into each C library variant when
  that library is enabled (`OP2_ENABLE_PETSC`, `_HYPRE`, `_GINKGO`, `_AMGX`,
  `_ROCALUTION`, `_SOLVE_MOCK`), the memory space following the variant.
- Translator: `op_arg_mat` parsed by the C++ front end into an argument on the
  matrix (a `Dat` marked `mat`, never an op_dat at run time) through a slot
  map named by (matrix, row map, row index, column map), with a `BlockPtr`
  kernel parameter (seq, openmp, cuda, hip; the legacy targets reject matrix
  blocks).
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
loop, the right-hand side in a node loop, solve, update. `-config FILE` is
its configuration (`op_solve_set_config`): matrices `A` and `P`, the solver
`heat`, or `heat_bjacobi` and `heat_neumann` under the par-loop
preconditioners (block Jacobi reading the diagonal blocks with
op_arg_mat(OP_READ), and a Neumann series). The tests read
`tests/solve/config.json` and pick the library and storage by
`OP_SOLVE_CONFIG` over it.
`-partition random` scatters the nodes over the ranks (OP2's random
partitioner) in place of ParMETIS, so on two ranks a row has as many blocks
in the other rank's columns as in its own, and as many scratch slots.

## 9. Out of scope for now

Rectangular blocks and rectangular matrices (rows != cols sets), single
precision, Fortran bindings (a thin layer over the C API; Hydra needs it next),
HIP runs (the same code with HIP builds of the libraries), matrices whose
pattern changes after declaration, more than 2^31 values in a region.
