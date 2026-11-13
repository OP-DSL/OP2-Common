# OP2 solver layer: summary

Par loops assemble sparse block matrices straight into a linear-solver
library's own storage, on the host or the GPU, and the library solves with
them. The full design is in `SOLVERS_DESIGN.md` and the measurements are in
`SOLVERS_RESULTS.md`.

Backends: PETSc, hypre, Ginkgo, AmgX (device only), rocALUTION, and a mock
library that owns its storage. The mock stands in for a closed-source library
such as SPLISS.

## API (`op2/include/op_solve.h`, C)

```c
op_solve_init();                                   /* after op_init / op_partition */
op_solve_set_config("solve.json");                 /* JSON text or a path */

op_sparsity sp = op_decl_sparsity(nodes, nodes, 1, &edge_nodes, &edge_nodes, "lap");
op_mat A = op_decl_mat(sp, NV, "double", "A");     /* NV x NV blocks */
op_solver s = op_decl_solver(A, "heat");           /* configured by name */

op_mat_zero(A);
op_par_loop(diag_block, "diag_block", nodes,       /* OP_ID, -1: the element itself */
            op_arg_dat(nbnd, -1, OP_ID, 1, "int", OP_READ),
            op_arg_mat(A, -1, OP_ID, -1, OP_ID, NV, "double", OP_INC));
op_par_loop(edge_blocks, "edge_blocks", edges,     /* block (map[e][i], map[e][j]) */
            op_arg_mat(A, 0, edge_nodes, 0, edge_nodes, NV, "double", OP_INC),
            op_arg_mat(A, 0, edge_nodes, 1, edge_nodes, NV, "double", OP_INC),
            op_arg_mat(A, 1, edge_nodes, 0, edge_nodes, NV, "double", OP_INC),
            op_arg_mat(A, 1, edge_nodes, 1, edge_nodes, NV, "double", OP_INC));
op_mat_assemble(A);                                /* values final; library rebuilds */

op_solve(s, b, x);                                 /* dats on A's rows/cols; 1 = converged */
op_solver_stats(s, &iterations, &residual);
op_solve_exit();
```

A kernel sees each block as `double *A`, indexed `A[r * NV + c]`. For
`OP_READ`, the same `op_arg_mat` reads blocks. Custom solving:

- `op_solver_set_pc_mat(s, P)` preconditions with another assembled matrix.
- `op_solver_set_pc(s, fn, ctx)` uses par loops as the preconditioner.
- `op_solver_set_operator(s, fn, ctx)` applies A matrix-free.

## Configuration (JSON)

The sources merge as a JSON merge patch, in order: each
`op_solve_set_config` call, then `OP_SOLVE_CONFIG` last.

```json
{
  "library": "hypre",
  "libraries": { "petsc": { } },
  "matrices": { "A": { "library": "ginkgo", "format": "sellp" } },
  "solvers":  { "heat": { "hypre":  { "solver": "PCG", "Tol": 1e-10, "precond": "BoomerAMG" },
                          "ginkgo": { "type": "solver::Cg", "criteria": { "iteration": 500 } } } }
}
```

- **`library`**: the default library for matrices.
- **`libraries`**: each library's own settings.
- **`matrices`**: a matrix's library and storage format.
- **`solvers`**: one section per library, written in that library's own
  vocabulary. A solver runs with its matrix's library.

A backend fails on settings it does not read. `tests/solve/config.json` is a
complete example.

## How it works

- **Sparsity** (`op_decl_sparsity`): built once per rank from the maps. Each
  row's blocks are split into a diagonal part (owned columns) and an
  off-diagonal part (halo columns), as the libraries split them. Each block
  becomes a **slot**, and each (map, index) pair becomes a slot map, an
  ordinary op_map onto an internal set of slots.
- **Matrix arguments**: `op_arg_mat` is an `OP_ARG_MAT` that increments
  through a slot map, so every loop strategy (atomics, colouring, chunked
  staging) and the MPI exec-halo rule work unchanged. Blocks in halo rows go
  to scratch slots that are thrown away.
- **Storage** (`op2/rt/abi/matrix.h`):
  - A matrix's values live in up to 3 regions, and a layout entry per slot
    (offset, row stride, column stride, region and diagonal rule) says where
    each block's entries are.
  - Either OP2 **lends** a region to the library or **borrows** the library's
    own, so loops write the library's format in place.
  - A **form** (contiguous / one region / any region, unit columns, diagonal
    rule), worked out from the layout, is baked into JIT kernels.
- **Formats**:

  | library | formats (default first) |
  |---|---|
  | hypre | csr, diagonal first |
  | Ginkgo | csr, coo, ell, fbcsr, sellp |
  | PETSc | aij on the host, coo on the device; owned aij |
  | AmgX | bcsr, csr |
  | rocALUTION | csr, mcsr, bcsr, coo, ell |
  | mock | compact, bell (row or column major, interleaved) |

## Code layout

| path | what |
|---|---|
| `op2/include/op_solve.h` | the C API |
| `op2/include/op2/solve/solve.h` | internal: the layer's objects, the backend interface (`make_<lib>`, `Library`, `Matrix`, `Solver`, a `Section` that reads its configuration strictly), storage (`lend`, `borrow`), layout builders |
| `op2/src/solve/op_solve.cpp` | the API, sparsity and slot construction, the configuration and the library registry |
| `op2/src/solve/op_solve_layout.cpp` | layout builders (`csr_layout`, `bcsr_layout`, `ell_layout`, `sell_layout`, `bell_layout`, `csr_located`) |
| `op2/src/solve/op_solve_<lib>.cpp` | one backend per library: petsc, hypre, ginkgo, amgx, rocalution, mock (+ `mock.h`, `*_device.cu`) |
| `op2/include/op2/rt/abi/matrix.h` | the storage and layout contract between storage and kernels |
| `op2/include/op2/rt/kernel/prelude.h` | `BlockPtr<T, d, diagonal>` and `mat_block`: how kernels find a block |
| `op2/src/rt/op_rt_matrix.cpp` | the path for untranslated loops (`op_seq.h`) |
| `op2/src/rt/plan/*`, `rt/loop/call.h`, `src/mpi/*` | the loop runtime and MPI handling a matrix as an increment target |
| `translator-v2/.../cpp/parser.py`, `kernels.py`, `op.py`, `templates/{cpp/seq,cpu,gpu}/loop_host*.jinja` | the translator: `op_arg_mat` parsing and `BlockPtr` kernel parameters for seq, openmp, cuda and hip (legacy targets reject matrix blocks) |
| `cmake/OP2Solve.cmake` | `OP2_ENABLE_{PETSC,HYPRE,GINKGO,AMGX,ROCALUTION,SOLVE_MOCK}` |
| `apps/c/implicit/` | proxy app: implicit coupled heat equations with NV components (`OP2_IMPLICIT_NV`). Options: `-mesh`, `-reorder`, `-pc`, `-partition`, `-check`, `-timing` |
| `tests/solve/` | CTest (label `solve`): each library × variant × case, every format, `config.json` |

## Build and test

- Configure with the `OP2_ENABLE_*` options above, plus `OP2_ENABLE_TESTS` and
  `OP2_IMPLICIT_NV="1;4;6"`. Then run `ctest -L solve`.
- Environment variables: `OP_SOLVE_CONFIG`, and `OP_SOLVE_MEMORY=1` to report
  device memory at each stage.

## Open items

- Fortran `op_arg_mat`.
- Ordering a sparsity for its first matrix's library, so hypre's
  diagonal-first matrices can use the contiguous form again.
- Segregated or multiple-block-size solves.
- HIP builds of the libraries.
- The SPLISS backend.
