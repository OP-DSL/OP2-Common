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
  (`-op matfree`), OP2's random partitioner in place of ParMETIS
  (`-partition random`), `-check TOL`, `-timing 1`.
- Tests: `ctest -L solve` runs the proxy for every variant and library built
  (mode start against the discrete solution, bump start by its true residual,
  the Neumann preconditioner for d > 1, and on two ranks a randomly
  partitioned mesh, whose rows have as many blocks in the other rank's columns
  as in their own). 171 pass on CPU (NV 1, 4 and 7; seq, genseq, openmp, and
  their MPI forms on two ranks); 228 pass on the H100 build (NV 1, 4 and 6,
  with cuda and mpi_cuda added). The rest of the suite is unchanged (78/78 on
  CPU).

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

NV = 1: n = 2048 (4.2M nodes, 21M values). NV = 4: n = 1024 (1.05M nodes,
5.2M blocks, 84M values). NV = 6: n = 836 (0.70M nodes, 3.5M blocks, 126M
values). CG to a true relative residual of 1e-8; solve figures are from step
9 of 10 (the first carries one-off setup). Rerun on 2026-10-31 after the
index-width changes below, against a baseline of the previous build in the
same session: assembly kernels, solve times and memory all agreed (kernels
within 0.5%). Logs: `logs/bench_base`, `logs/bench_after`.

### Assembly (per step)

| | NV = 1 loops + commit | NV = 4 loops + commit | NV = 6 loops + commit |
|---|---|---|---|
| PETSc (GPU: blocked, COO) | 2.6 + 0.4 ms | 4.7 + 1.6 ms (9.9) | 23.1 + 2.4 ms (7.7) |
| hypre (in place) | 2.7 + 0 ms | 6.0 + 0 ms (8.9) | 21.2 + 0 ms (9.8) |
| Ginkgo (in place) | 2.6 + 0 ms | 5.8 + 0 ms (8.2) | 18.9 + 0 ms (9.2) |

Steps 1-2, with step 9's loops in brackets. The default edge loop times as
forced chunk_atomics does, and after a few calls (the JIT off too) both take
forced atomics' time: slower at d = 4 (5.9 against 8.6-9.0 ms), faster at
d = 6 (21 against 9.7 ms; forced chunk_colouring 14, colouring 22). That is
OP2's loop machinery, not the solver layer, and has not been looked into
further.

The in-place layout writes each block row to a different scalar row, so its
edge kernel is ~15% slower than blocked storage (5.1 against 4.4 ms, nsys), and
saves the commit (0.9 ms permuting, 1.6 ms COO) and the second copy of the
values. Before the diagonal-first variant was made compile-time the in-place
kernel was 3x slower: the per-access sign test cost more than the atomics.

### Matrix memory on the device

OP2 holds the values (160 / 640 / 959 MiB), the slot maps (144 / 36 / 24 MiB)
and for NV > 1 the layout table (40 / 27 MiB). Beyond that, the library's
matrix:

| | NV = 1 | NV = 4 | NV = 6 | per value at NV = 6 |
|---|---|---|---|---|
| PETSc (its CSR + COO maps) | 582 MiB | 2262 MiB | 3380 MiB | 28.2 B |
| hypre (index arrays; the values are OP2's) | 148 MiB | 388 MiB | 548 MiB | 4.6 B |
| Ginkgo (index arrays; the values are OP2's) | 118 MiB | 358 MiB | 514 MiB | 4.3 B |

With the staged commit, hypre's and Ginkgo's NV = 4 matrices were 1350 and
1318 MiB: in place saves ~1 GB per 4.2M unknowns at d = 4, and the saving
grows as d^2.

### Solve (per CG iteration, steady state)

| | NV=1 Jacobi | NV=1 par-loop bjacobi | NV=4 lib Jacobi | NV=4 par-loop bjacobi | NV=6 lib Jacobi | NV=6 par-loop bjacobi |
|---|---|---|---|---|---|---|
| PETSc | 0.87 ms | 0.93 ms | 3.09 ms (PCPBJACOBI) | 1.30 ms | 3.58 ms (PCPBJACOBI) | 1.78 ms |
| hypre | 0.98 ms | 0.97 ms | 1.42 ms (point) | 1.35 ms | 1.72 ms (point) | 1.83 ms |
| Ginkgo | 1.31 ms | 1.38 ms | 1.75 ms (block) | 1.86 ms | 2.19 ms (block) | 2.38 ms |

Ginkgo's block Jacobi now skips sorting the local matrix (its columns are
sorted already); before, it built a sorted copy of the matrix at every setup,
which cost 1.93 against 1.75 ms per iteration at NV = 4 and ran out of memory
near the 32-bit limit (below).

### AMG (setup every step)

| | NV = 1 | NV = 4 | NV = 6 |
|---|---|---|---|
| PETSc GAMG | 12 its, 0.15 s/step after an 8.1 s first setup; 8.4 GB | out of memory (80 GB) | out of memory |
| hypre BoomerAMG | 14 its, 0.13 s/step; 6.9 GB | 15 its, 0.31 s/step; 22 GB | 15 its, 0.43 s/step; 32 GB |
| Ginkgo Pgm | 56 its, 1.1 s/step | does not converge (scalar aggregation) | does not converge |

The memory is the device's in use after the solve, with hypre's Umpire pool
holding its high-water mark: 260-330 bytes per matrix value with BoomerAMG.
So AMG fills a GPU long before a rank reaches the 32-bit limit below.

Ginkgo's first solve costs a further 3-26 s (loading its CUDA modules), so its
figures are steady state.

## Index widths and the per-rank limit

A rank holds at most 2^31 values of a matrix (owned blocks and scratch slots):
generated code and the libraries' local CSR address them with 32-bit offsets
(design, section 4). `op_decl_mat` now stops with a message past it. At d = 6
that is 59.6M blocks, 16 GiB of values: 8.5M hex-mesh or 4.0M tet-mesh nodes
per rank.

One H100 rank just under it, d = 6, n = 3400 (11.6M nodes, 57.8M blocks,
2.08e9 values, 97% of 2^31), CG + Jacobi from the bump, dt = 1e-4
(`logs/bench_after/limit*.log`):

| | hypre | Ginkgo | PETSc |
|---|---|---|---|
| result | converged, true residual 9.9e-9 | converged, 9.8e-9 | out of memory |
| per CG iteration | 28 ms | 31 ms | |
| assembly loops per step | 366 ms | 325 ms | |
| first commit (one-off) | 9.9 s | 9.6 s | |
| OP2: values + layout, slot maps | 15.9 GiB, 0.4 GiB | the same | |
| library's matrix | 8.3 GiB (4.3 B per value) | 8.3 GiB (4.3 B per value) | ~55 GiB more (COO, 28 B per value) |
| device in use after the solve | 32.3 GiB | 35.9 GiB | |

The rest of the device memory is the proxy's own dats and the CUDA context
(6 GiB, half of it the d x d inverse blocks) and the solvers' vectors. The first commit builds the
library's index arrays on the host and copies them over. At n = 3500
(2.20e9 values) the run stops at `op_decl_mat`: "2204496000 values on a rank
in 6 x 6 blocks, past the 32-bit offsets loops and solver libraries take".

Changes behind these figures:

- Kernels keep 32-bit offsets. 64-bit ones were tried and measured on the JIT
  kernels for d = 1, 4 and 6 (sm_90 and gfx942, `logs/regs`,
  `scripts/regtable.py`). Every 64-bit form of `BlockPtr` raised some matrix
  kernel: computing slot * d * d in 64 bits plus a relative offset raised the
  d = 6 colouring edge kernel from 80 to 96 registers and gfx942 kernels by up
  to 8 VGPRs. The nearest to neutral, signed offsets from a base 2^31 values
  in, still took the d = 6 chunk_colouring wavefront kernels from 32 to 40.
  The libraries cap each part at 2^31 anyway, so wider kernel offsets would
  only have added a rank's off-diagonal and scratch blocks. Generated code is
  unchanged (offline kernels' registers compared equal).
- In the same study, indexing a block as `data[r * stride + c]` instead of
  through row pointers took the d = 6 atomics edge kernel from 128 to 40
  registers on sm_90 (colouring 80 to 32), but raised gfx942's diag_block
  kernels by 8 and d = 6 chunk_colouring and invert_block on sm_90 by 8. Not
  taken; worth timing.
- hypre is built with `HYPRE_ENABLE_MIXEDINT` (64-bit global indices; its
  all-64-bit build has no GPU support), for meshes past 358M nodes at d = 6.
  It cost nothing measurable: per-iteration times, BoomerAMG included, and
  memory are as with 32-bit indices. The hypre and PETSc backends check the
  global size against their index types.
- Untranslated loops (op_seq.h) computed a dat element's byte offset in
  32 bits and broke past 2 GiB, which a d = 6 matrix reaches at 7.5M blocks:
  NV = 4 at n = 1900 (2.3 GiB of values) segfaulted in the old build and
  solves to 6e-11 against the discrete solution now.
- Ginkgo's Jacobi and ILU skip sorting the matrix, which is sorted: Ginkgo's
  block Jacobi otherwise sorted a full copy, and at the limit ran out of
  memory.

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
- At most 2^31 values of a matrix per rank (above). Going past needs 64-bit
  local indices in the libraries (Ginkgo int64, PETSc --with-64-bit-indices;
  hypre has none on the GPU) and kernel offsets that do not cost registers.
  PETSc's 32-bit build also caps the global size at 2^31 rows.
- Scratch slots are one per (element, block) write to a halo row; deduplicated
  per block they would shrink with heavy halos (a random partition on two
  ranks nearly doubles the values).
- The first commit near the limit takes ~10 s (index arrays built on the host,
  then copied); building them on the device would cut it.
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
`OP_SOLVE_HYPRE_POOL` (MiB; Umpire reserves 4 GiB by default). The libraries'
scripts take `ONLY=<library>`; hypre is built with mixed integers.
`bench_all.sh <tag>` reruns every timing here, `perf_limit.sh` the runs at the
32-bit limit, `jit_regs_implicit.sh` the JIT kernels' registers.
