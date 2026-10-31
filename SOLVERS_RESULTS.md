# Linear solvers in OP2: results

Branch `feature/solvers` in `/home/jk-ext/op2-solvers` (on `feature/rt-redesign`
77ea9858). Design: `SOLVERS_DESIGN.md`. Solver library builds:
`/home/jk-ext/solvers` (scripts in `scripts/`, logs in `logs/`).

## What exists

- `op_solve.h`: `op_decl_sparsity`, `op_decl_mat`, `op_arg_mat`, `op_mat_zero`,
  `op_mat_assemble`, `op_decl_solver`, `op_solve`, `op_solver_stats`, and the
  custom preconditioning hooks `op_solver_set_pc_mat` (assembled P),
  `op_solver_set_pc` (par-loop preconditioner) and `op_solver_set_operator`
  (par-loop operator).
- `op_arg_mat(A, i, rowmap, j, colmap, d, "double", OP_INC|OP_READ)` is an
  indirect increment (or read) of the matrix's values through an internal slot
  map, so every existing strategy assembles: seq, OpenMP colouring, CUDA
  atomics/colouring/chunked staging, JIT, MPI with the exec halo. The kernel
  writes `A[r * d + c] += ...`; the translator parses `op_arg_mat` and nothing
  in the templates changed.
- Backends, chosen at run time (`OP_SOLVE_LIB`, or `op_solve_init`'s argument):
  - PETSc 3.26: AIJ (AIJCUSPARSE on the GPU) through COO assembly from the
    values in place (`MatSetValuesCOO` from device memory), KSP over vectors
    wrapping the dats' device arrays; PCSHELL and MATSHELL for the callbacks.
  - hypre 3.2: ParCSR built around the values: for scalar blocks hypre's
    diagonal and off-diagonal data are the OP2 values (zero copy, nothing to
    commit); larger blocks fill hypre's arrays with one device permuting copy.
    PCG/GMRES/FlexGMRES/BiCGSTAB with BoomerAMG (systems AMG for d > 1), ILU,
    Jacobi or the par-loop preconditioner.
  - Ginkgo 1.11: distributed Matrix whose local and non-local Csr are views of
    the values (scalar blocks; a permuting copy otherwise), distributed
    vectors over views of the dats, Schwarz-Jacobi/ILU and Pgm multigrid,
    callback LinOps for the par-loop preconditioner and operator.
- MPI: global numbering by exchanging global ids once, blocks of halo rows
  (from exec-halo elements) to scratch slots, diagonal/off-diagonal split with
  columns compacted to a global column map; the sets and maps the layer
  declares after partitioning get empty halos (`op_mpi_local_objects`).
- `apps/c/implicit`: backward Euler for u_t = div(K grad u) - C u with NV = 1
  or 4 components on an n x n grid stored as a randomly renumbered
  unstructured mesh; options for the library's or a par-loop block-Jacobi
  preconditioner and an assembled or par-loop (matrix-free) operator.

## Correctness

All on the same problem (n = 64, 5 steps, bump start, CG, rtol 1e-10):

- seq, genseq (translated), OpenMP, CUDA, MPI on 1-4 ranks, MPI+CUDA on 1-4
  ranks: identical iteration counts and true relative residual (8e-11) for
  NV = 1 and 4, for every library.
- The par-loop block-Jacobi preconditioner (reads the diagonal blocks in
  solver memory with `op_arg_mat(..., OP_READ)`, inverts them into a dat,
  applies them in a direct loop) reproduces PETSc's PCPBJACOBI iteration for
  iteration; the par-loop operator reproduces the assembled one.
- The mode start: error against the exact discrete solution ~1e-13 at every
  size (the solver tolerance), and second order against the continuous one:

  | n | NV=1 max error | NV=4 max error |
  |---|---|---|
  | 16 | 1.10e-2 | 1.24e-2 |
  | 32 | 2.98e-3 | 3.38e-3 |
  | 64 | 7.69e-4 | 8.72e-4 |
  | 128 | 1.96e-4 | 2.23e-4 |
  | 256 | 4.95e-5 | 5.62e-5 |

  (dt = h^2, t = 0.02; ratios 3.7-3.97.)

## Performance and memory (H100 PCIe, ~4.2M unknowns, 10 steps, reassembled each step)

NV = 1: n = 2048 (21M nonzeros). NV = 4: n = 1024 (1.05M nodes, 67M scalar
nonzeros). CG to a true relative residual of 1e-8. Logs:
`/home/jk-ext/solvers/logs/perf_{petsc_2,hypre_1,ginkgo_1}.log`.

Assembly (both par loops, per step): 2.6 ms (NV = 1), 6.0 ms (NV = 4) for
every library. Commit per step:

| | NV = 1 | NV = 4 |
|---|---|---|
| PETSc (`MatSetValuesCOO`) | 2.6 ms | 3.5 ms |
| hypre | 0 (values are hypre's) | 0.9 ms (permute) |
| Ginkgo | 0 (views) | 0.9 ms (permute) |

Matrix memory on the device, beyond OP2's values (160 / 640 MiB) and slot maps
(144 / 36 MiB):

| | NV = 1 | NV = 4 |
|---|---|---|
| PETSc (CSR + COO maps) | 582 MiB | 2262 MiB |
| hypre (index arrays; + its scalar values for NV = 4) | 148 MiB | 1350 MiB |
| Ginkgo | 118 MiB | 1318 MiB |

Solve (steady state, per CG iteration):

| | NV = 1 Jacobi | NV = 1 par-loop bjacobi | NV = 4 lib Jacobi | NV = 4 par-loop bjacobi |
|---|---|---|---|---|
| PETSc | 0.87 ms | 0.94 ms | 2.96 ms (PCPBJACOBI) | 1.30 ms |
| hypre | 0.97 ms | 0.97 ms | 1.42 ms (point) | 1.35 ms |
| Ginkgo | 1.31 ms | 1.38 ms | 1.93 ms (block) | 1.85 ms |

AMG (10 steps, setup every step):

| | NV = 1 | NV = 4 |
|---|---|---|
| PETSc GAMG | 12 its/step, 0.15 s/step after an 8.1 s first setup; 8.4 GB | out of memory (80 GB) |
| hypre BoomerAMG | 15 its/step, 0.13-0.20 s/step, 1.44 s in all; 6.9 GB | 16 its/step, 0.31 s/step; 23 GB |
| Ginkgo Pgm | 59-64 its/step, 1.17 s/step | does not converge (scalar aggregation) |

Ginkgo's first solve costs 8-26 s more (its CUDA modules loading; 172 sm_90
images, no JIT), so its numbers above are steady state.

Observations:
- Assembly into solver memory is cheap beside the solve; the staged commits
  cost about as much as the assembly loops themselves, and are what the direct
  layout removes.
- PETSc's COO path doubles the values and adds 8-byte maps (3.4x a CSR); hypre
  and Ginkgo with views hold only their index arrays.
- Par-loop preconditioners cost nothing extra worth measuring (two vector
  copies per application), and the par-loop block Jacobi is 2.3x faster than
  PETSc's own PCPBJACOBI on the GPU.
- hypre's BoomerAMG on the GPU is the fastest AMG here by far, and the only one
  that handles the coupled 4-component system.

## Custom preconditioners: the answer

op_par_loops plus `op_arg_mat` are enough. The four mechanisms in the design
(assembled P, par-loop preconditioner callback, `op_arg_mat(OP_READ)` in those
loops, par-loop operator) are implemented and tested on all three libraries,
CPU and GPU. Nothing else was needed: the library sees an operator, and the
preconditioner is ordinary OP2 code with access to the assembled matrix.

## Known limitations

- d > 1 commits through a permuting copy (hypre, Ginkgo) or COO (PETSc); the
  direct layout (rows of each block at a row stride in the library's scalar
  CSR) is the next step.
- The par-loop operator is not supported by the hypre backend.
- Only square matrices on one set, "double", AoS vectors.
- C++ only (no Fortran bindings yet), CUDA tested (HIP untested).
