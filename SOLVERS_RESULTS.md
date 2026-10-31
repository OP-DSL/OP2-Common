# Linear solvers in OP2: results

Branch `feature/solvers` in `/home/jk-ext/op2-solvers`, on `feature/rt-redesign`
77ea9858 (op2-loop is untouched; the clone has no remote). Design:
`SOLVERS_DESIGN.md`. Solver library builds, scripts and logs:
`/home/jk-ext/solvers` (`scripts/`, `logs/`).

## What exists

- `op_solve.h`: `op_decl_sparsity`, `op_decl_mat`, `op_arg_mat`, `op_mat_zero`,
  `op_mat_assemble`, `op_decl_solver`, `op_solve`, `op_solver_stats`, and the
  custom preconditioning hooks `op_solver_set_pc_mat` (assembled P),
  `op_solver_set_pc` (par-loop preconditioner) and `op_solver_set_operator`
  (par-loop operator).
- `op_arg_mat(A, i, rowmap, j, colmap, d, "double", OP_INC|OP_READ)` is an
  indirect increment (or read) of the matrix's values through a slot map, so
  every strategy assembles: seq, OpenMP colouring and chunks, CUDA
  atomics/colouring/chunked staging, the JIT, MPI with the exec halo. The kernel
  writes `A[r * d + c] += ...`.
- For d > 1 the loops write the library's own scalar CSR in place: the kernel
  parameter is an `op::rt::kernel::BlockPtr<T, d>` that finds each block row at
  an offset and row stride from a layout table (a hidden argument the
  translator adds). Diagonal blocks under hypre's diagonal-first rows are a
  compile-time variant, so other blocks pay nothing for it. Loops the
  translator never saw (op_seq.h) fall back to blocked storage and a permuting
  commit.
- Backends, chosen at run time (`OP_SOLVE_LIB`, or `op_solve_init`'s argument):
  - PETSc 3.26: on the host, AIJ whose parts' arrays are the values (written in
    place); on the GPU, AIJCUSPARSE through COO assembly from the values in
    device memory (`MatSetValuesCOO`). KSP over vectors wrapping the dats;
    PCSHELL and MATSHELL for the callbacks.
  - hypre 3.2: ParCSR whose diagonal and off-diagonal data are the values
    (written in place). PCG/GMRES/FlexGMRES/BiCGSTAB with BoomerAMG (systems
    AMG for d > 1), ILU, Jacobi or the par-loop preconditioner.
  - Ginkgo 1.11: distributed Matrix whose local and non-local Csr are views of
    the values, distributed vectors over views of the dats, Schwarz-Jacobi/ILU
    and Pgm multigrid, callback LinOps for the par-loop preconditioner and
    operator.
- MPI: global numbering by exchanging global ids once, blocks of halo rows
  (from exec-halo elements) to scratch slots, diagonal/off-diagonal split with
  columns compacted to a global column map; the layer's sets and maps get
  empty halos after partitioning (`op_mpi_local_objects`).
- `apps/c/implicit`: backward Euler for u_t = div(K grad u) - C u with NV
  components (`OP2_IMPLICIT_NV`, default 1 and 4) on an n x n grid stored as a
  randomly renumbered unstructured mesh. Options: the library's
  preconditioner, a par-loop block Jacobi (`-pc bjacobi`) or Neumann series
  (`-pc neumann`) preconditioner reading the matrix in solver memory, a
  decoupled assembled P (`-pmat decoupled`), an assembled or par-loop operator
  (`-op matfree`), `-check TOL`, `-timing 1`.
- Tests: `ctest -L solve` runs the proxy for every variant and library built
  (mode start against the discrete solution, bump start by its true residual,
  the Neumann preconditioner for d > 1). 144 pass on CPU (NV 1, 4 and 7;
  seq, genseq, openmp, and their MPI forms on two ranks); 120 pass on the H100
  build (NV 1 and 4, with cuda and mpi_cuda added). The rest of the suite is
  unchanged (78/78 on CPU).

## Correctness

- Every variant (seq, genseq, openmp, cuda, MPI on 1-4 ranks, MPI+CUDA on 1-4
  ranks), library, layout (blocked or in place) and GPU strategy (atomics,
  colouring, chunk_colouring forced by `OP_STRATEGY`) gives the same iteration
  counts and true residual on the same problem.
- The par-loop block Jacobi reproduces PETSc's PCPBJACOBI iteration for
  iteration; the par-loop operator reproduces the assembled one; the Neumann
  preconditioner gives the same counts on all three libraries, whose layouts
  differ (hypre diagonal-first, Ginkgo sorted, PETSc blocked on the GPU).
- From the mode, the error against the exact discrete solution is ~1e-13 at
  every size; against the continuous solution it falls at second order:

  | n | NV=1 max error | NV=4 max error |
  |---|---|---|
  | 16 | 1.10e-2 | 1.24e-2 |
  | 32 | 2.98e-3 | 3.38e-3 |
  | 64 | 7.69e-4 | 8.72e-4 |
  | 128 | 1.96e-4 | 2.23e-4 |
  | 256 | 4.95e-5 | 5.62e-5 |

  (dt = h^2, t = 0.02; ratios 3.7-3.97.)

## Performance and memory (H100 PCIe, ~4.2M unknowns, reassembled every step)

NV = 1: n = 2048 (21M nonzeros). NV = 4: n = 1024 (1.05M nodes, 67M scalar
nonzeros). CG to a true relative residual of 1e-8, steady-state steps (the
first carries one-off setup). Logs: `logs/perf_{petsc,hypre,ginkgo}_final.log`.

### Assembly (per step)

| | NV = 1 loops + commit | NV = 4 loops + commit |
|---|---|---|
| PETSc (GPU: blocked, COO) | 2.6 + 0.4 ms | 4.7 + 1.6 ms |
| hypre (in place) | 2.7 + 0 ms | 5.9 + 0 ms |
| Ginkgo (in place) | 2.6 + 0 ms | 5.8 + 0 ms |

The in-place layout writes each block row to a different scalar row, so its
edge kernel is ~15% slower than blocked storage (5.1 against 4.4 ms, nsys), and
saves the commit (0.9 ms permuting, 1.6 ms COO) and the second copy of the
values. Before the diagonal-first variant was made compile-time the in-place
kernel was 3x slower: the per-access sign test cost more than the atomics.

### Matrix memory on the device

OP2 holds the values (160 / 640 MiB), the slot maps (144 / 36 MiB) and for
NV = 4 the layout table (40 MiB). Beyond that, the library's matrix:

| | NV = 1 | NV = 4 |
|---|---|---|
| PETSc (its CSR + COO maps) | 582 MiB | 2262 MiB |
| hypre (index arrays; the values are OP2's) | 148 MiB | 388 MiB |
| Ginkgo (index arrays; the values are OP2's) | 118 MiB | 358 MiB |

With the staged commit, hypre's and Ginkgo's NV = 4 matrices were 1350 and
1318 MiB: in place saves ~1 GB per 4.2M unknowns at d = 4, and the saving
grows as d^2.

### Solve (per CG iteration, steady state)

| | NV=1 Jacobi | NV=1 par-loop bjacobi | NV=4 lib Jacobi | NV=4 par-loop bjacobi |
|---|---|---|---|---|
| PETSc | 0.86 ms | 0.93 ms | 2.86 ms (PCPBJACOBI) | 1.30 ms |
| hypre | 0.98 ms | 0.97 ms | 1.42 ms (point) | 1.35 ms |
| Ginkgo | 1.31 ms | 1.39 ms | 1.93 ms (block) | 1.86 ms |

### AMG (setup every step)

| | NV = 1 | NV = 4 |
|---|---|---|
| PETSc GAMG | 12 its, 0.15 s/step after an 8.1 s first setup; 8.4 GB | out of memory (80 GB) |
| hypre BoomerAMG | 15 its, 0.137 s/step; 6.9 GB | 16 its, 0.31 s/step; 22 GB |
| Ginkgo Pgm | 59 its, 1.14 s/step | does not converge (scalar aggregation) |

Ginkgo's first solve costs a further 3-26 s (loading its CUDA modules), so its
figures are steady state.

## Observations

- Assembly into solver memory costs a few milliseconds beside solves of
  ~0.1-2 s per step here; for Hydra's flux Jacobians the loops are heavier but
  the commit saved is the same bandwidth-bound pass.
- Writing in place is where the memory goes: PETSc's COO path holds 3.4x a
  CSR; hypre and Ginkgo with the values in place hold only their index arrays.
- Par-loop preconditioners cost two vector copies per application, nothing
  measurable; the par-loop block Jacobi is 2.2x faster than PETSc's own
  PCPBJACOBI on the GPU.
- hypre's BoomerAMG on the GPU is the fastest AMG here, and the only one that
  handles the coupled 4-component system. Ginkgo's Pgm works on the decoupled
  P (`-pmat decoupled`: 88 iterations where A gives no convergence), the
  intended use of a separate preconditioning matrix.
- PETSc GAMG with the decoupled P (explicit zeros inside its blocks) reports
  an indefinite preconditioner on its first setup only; with P = A it is fine.

## Custom preconditioners: the answer

op_par_loops plus `op_arg_mat` are enough. The four mechanisms in the design
(assembled P, par-loop preconditioner callback, `op_arg_mat(OP_READ)` in those
loops, par-loop operator) are implemented and tested on all three libraries,
CPU and GPU, with block Jacobi and a Neumann series as working examples.
Nothing else was needed: the library sees an operator, and the preconditioner
is ordinary OP2 code with access to the assembled matrix in the library's own
storage.

## Limitations and next steps

- Fortran: the API is C, so a binding is thin, but the Fortran translator
  needs `op_arg_mat` (the f2c path would reuse the C++ templates' BlockPtr).
  Hydra needs this next.
- PETSc on the GPU still commits through COO (one copy of the values plus
  8-byte maps). In place there needs PETSc's Kokkos backend (AIJKokkos over
  unmanaged views) or writing into its two CUSPARSE arrays.
- The par-loop operator is not supported by the hypre backend (ParCSR Krylov
  solvers take a matvec only through hypre's generic interface).
- Slot maps cost as much as the scalar values: the diagonal blocks of each
  edge could come from the node map, halving them.
- Only square matrices on one set, "double", AoS vectors, full (not partial)
  halo exchanges, CUDA tested (HIP compiles the same code but needs HIP builds
  of the libraries), the legacy C++ targets reject matrix blocks.
- Ginkgo's distributed multigrid on coupled systems, and PETSc GAMG's memory
  on the GPU, are library limits rather than integration ones.

## Building and running

```
# Solver libraries (already built): /home/jk-ext/solvers/{cpu,cuda},
# PETSc in /home/jk-ext/solvers/src/petsc{,-cuda}/arch-{cpu,cuda}
cmake -S /home/jk-ext/op2-solvers -B build -DOP2_ENABLE_APPS=ON -DOP2_ENABLE_TESTS=ON \
  -DOP2_ENABLE_PETSC=ON -DPETSC_DIR=... -DPETSC_ARCH=... \
  -DOP2_ENABLE_HYPRE=ON -DHYPRE_DIR=.../lib/cmake/HYPRE \
  -DOP2_ENABLE_GINKGO=ON -DGinkgo_DIR=.../lib/cmake/Ginkgo
OP_SOLVE_LIB=hypre ./build/apps/c/implicit/implicit4_cuda -n 1024 -steps 10 -init bump \
  -solver "method=cg pc=amg rtol=1e-8"
ctest --test-dir build -L solve
```

`/home/jk-ext/solvers/scripts/` has the exact configurations: `build_cpu.sh`,
`build_cuda.sh` (the libraries), `op2_cpu_build.sh`, `gpu_configure.sh` (this
tree; the GPU one also needs `CMAKE_CUDA_HOST_COMPILER` and
`CUDA_TOOLKIT_ROOT_DIR` for hypre's Umpire/BLT config), `perf_lib.sh`,
`assembly.sh`, `nsys_assembly.sh`, `gpu_solve_tests.sh`.
Variables: `OP_SOLVE_LIB`, `OP_SOLVE_LAYOUT=blocked`, `OP_SOLVE_MEMORY=1`,
`OP_SOLVE_HYPRE_POOL` (MiB; Umpire reserves 4 GiB by default).
