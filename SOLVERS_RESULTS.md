# Linear solvers in OP2: results

Branch `feature/solvers` in `/home/jk-ext/op2-solvers`, on `feature/rt-redesign`
77ea9858 (op2-loop is untouched; the clone has no remote). Design:
`SOLVERS_DESIGN.md`. Solver library builds, scripts and logs:
`/home/jk-ext/solvers` (`scripts/`, `logs/`).

## What exists

- `op_solve.h`: `op_solve_set_config` (with `OP_SOLVE_CONFIG`, a JSON
  configuration in each library's own terms; design, section 2),
  `op_decl_sparsity`, `op_decl_mat`, `op_arg_mat`, `op_mat_zero`, `op_mat_assemble`,
  `op_decl_solver`, `op_solve`, `op_solver_stats`, and the custom
  preconditioning hooks `op_solver_set_pc_mat` (assembled P),
  `op_solver_set_pc` (par-loop preconditioner) and `op_solver_set_operator`
  (par-loop operator).
- `op_arg_mat(A, i, rowmap, j, colmap, d, "double", OP_INC|OP_READ)` is a
  first-class argument (`OP_ARG_MAT`, its `op_arg.mat` the matrix's
  `op_mat_storage`, `op2/rt/abi/matrix.h`): no op_dat. Plans, colouring,
  chunked staging and the MPI exec-halo rule take a matrix as a target, so
  every strategy assembles. The kernel writes `A[r * d + c] += ...` through an
  `op::rt::kernel::BlockPtr<T, d>` that `mat_block` builds from the storage.
- Storage (design, section 4): up to three regions (a library's local and
  non-local parts, OP2's scratch for blocks of halo rows) that OP2 lends, the
  library owns and OP2 borrows, or OP2 hands over at assembly; a layout entry
  per slot (offset, row and column strides, region and diagonal rule); and a
  form worked out from the layout (contiguous, one region or any region, unit
  column strides, the one diagonal rule) that JIT kernels bake in, so what a
  library's layout does not need compiles out. Untranslated loops (op_seq.h)
  write a blocked buffer that `op_mat_assemble` adds in.
- Libraries and formats, chosen per matrix by the configuration (so a run can
  use several), tested on CPU and the H100:

  | library | formats (default first) | storage | where |
  |---|---|---|---|
  | PETSc 3.26 | aij (host), coo (device); owned aij | lend; copy; borrow | CPU, GPU (owned aij CPU only) |
  | hypre 3.2 (mixed int) | csr, diagonal first | lend | CPU, GPU |
  | Ginkgo 1.11 (patched, below) | csr (strategy cusparse, classical, merge_path), coo, ell, fbcsr; sellp | lend; borrow | CPU, GPU |
  | AmgX 2.5 | bcsr (d = 1, 3, 4; external diagonal on one rank), csr (default past d = 4) | copy | GPU |
  | rocALUTION 4.0 | csr, mcsr, bcsr, coo, ell | borrow | CPU (host build) |
  | mock (SPLISS stand-in) | compact, bell (row or column major; bell interleaved) | borrow, local and non-local apart | CPU, GPU |

  Solvers: CG, GMRES, FGMRES, BiCGStab where the library has them, with its
  Jacobi, ILU and AMG, or a par-loop preconditioner (not AmgX).
- MPI: global numbering by exchanging global ids once, blocks of halo rows
  (from exec-halo elements) to scratch slots, diagonal/off-diagonal split with
  columns compacted to a global column map; the layer's sets and maps get
  empty halos after partitioning (`op_mpi_local_objects`).
- `apps/c/implicit`: backward Euler for u_t = div(K grad u) - C u with NV
  components (`OP2_IMPLICIT_NV`) on an n x n grid stored as a randomly
  renumbered unstructured mesh. Options: the library's preconditioner, a
  par-loop block Jacobi (`-pc bjacobi`) or Neumann series (`-pc neumann`)
  preconditioner reading the matrix in solver memory, a decoupled assembled
  P (`-pmat decoupled`), an assembled or par-loop operator (`-op matfree`),
  OP2's random partitioner in place of ParMETIS (`-partition random`),
  `op_renumber` after partitioning (`-reorder 1`), an airfoil or MG-CFD
  mesh (HDF5) in place of the grid (`-mesh FILE`), `-check TOL`,
  `-timing 1`, `-diags N`.
- Tests: `ctest -L solve` runs the proxy for every variant, library and
  format built: from the mode against the discrete solution, from the bump by
  its true residual, under the Neumann preconditioner for d > 1, on two ranks
  also on a randomly partitioned mesh (as many blocks of a row in the other
  rank's columns as in its own) and from the bump with each rank's nodes and
  edges reordered (Hilbert); each library's other formats at every component
  count. 810 pass on CPU (NV 1, 4, 7; seq, genseq, openmp and their MPI
  forms), 853 on the H100 (NV 1, 4, 6, with cuda and mpi_cuda), and the rest
  of the suite (78/78).

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
values). One rank, driver 615.71.10, CUDA 13.0. Logs:
`logs/perf_formats.log`, `logs/final_ab.log` (2026-10-31).

### Solve by library and format (CG + Jacobi to 1e-8, per iteration)

| library, format | NV = 1 | NV = 4 | NV = 6 |
|---|---|---|---|
| hypre csr | 0.98 ms | 1.42 ms | 1.73 ms |
| PETSc coo | 0.87 | 2.95 | 3.25 |
| Ginkgo csr | 1.31 | 1.74 | 2.15 |
| Ginkgo csr:cusparse | 1.32 | 1.69 | 2.04 |
| Ginkgo csr:classical | 1.27 | 1.76 | 2.00 |
| Ginkgo csr:merge_path | 1.76 | 2.85 | 3.89 |
| Ginkgo coo | 1.31 | 1.91 | 2.37 |
| Ginkgo ell | 1.20 | 1.65 | 2.02 |
| Ginkgo sellp | 1.21 | 1.65 | 2.03 |
| Ginkgo fbcsr | 1.28 (as csr) | 1.51 | 1.95 |
| AmgX bcsr | 0.82 | 1.04 | no 6 x 6 kernels |
| AmgX bcsr:diag | 0.90 | 1.06 | no 6 x 6 kernels |
| AmgX csr | 0.82 | 1.14 | 1.42 |
| mock compact | 0.90 | 0.93 | 1.14 |
| mock compact:column | 0.90 | 0.97 | 1.24 |
| mock bell (any variant) | 0.82 | 0.94-0.95 | 1.13-1.14 |

Iterations (step 5) differ with each library's Jacobi and convergence test:
PETSc and Ginkgo 1147 / 666 / 575, hypre and the mock 1147 / 627 / 539, AmgX
1379 / 747-798 / 642 (its tolerance is relative to the initial residual,
which each step's warm start makes small). Per-iteration times for hypre
and Ginkgo csr match the earlier build's; PETSc's are 5-9% lower at NV > 1.

Block formats pay off in the solve: Ginkgo's fbcsr is the fastest Ginkgo
format at NV = 4 and 6, AmgX's bcsr beats its csr by 8% at NV = 4, and the
mock's block CSR and block ELL (the shapes SPLISS stores) are the fastest
here at NV = 4 and 6, level with AmgX at NV = 1. Ginkgo's merge_path strategy is the slowest throughout.

Device memory in use after the first assembly, NV = 6 (the proxy's own data
included, the same in each): hypre 2397 MiB, Ginkgo csr 2431, coo 2879, ell
2399, sellp 2401, fbcsr 1941, AmgX csr 3811 (its copy of the values and its
own structure), PETSc coo 5229, mock compact 1875, bell 1867.

### Assembly (loops per step)

Each storage form's kernels, against the last build before first-class matrix
arguments (6ebd0b26, values as an op_dat), steps 2-7 at the default strategy
(atomics): offline kernels only (`OP_JIT_LEVEL=0`), and JIT kernels from the
first call (`OP_JIT_MIN_CALLS=1 OP_JIT_SEQ_COMPILE=1`).

| | NV = 1 | NV = 4 offline | NV = 4 JIT | NV = 6 offline | NV = 6 JIT |
|---|---|---|---|---|---|
| Ginkgo csr, before / now | 2.60 / 2.67 ms | 5.81 / 12.27 | 8.32 / 10.14 | 18.84 / 23.71 | 9.14 / 16.07 |
| hypre, before / now | 2.67 / 2.75 | 5.89 / 13.14 | 8.50 / 8.18 | 21.11 / 25.41 | 9.62 / 9.36 |
| PETSc coo, before / now | 2.62 / 2.66 | 4.66 / 16.06 | 9.53 / 16.67 | 23.05 / 27.73 | 7.64 / 27.12 |

- At NV = 1 the contiguous form reads no layout and matches the old build.
  Before the form existed every slot read a 16-byte layout entry and three
  region bases, and NV = 1 took 4.3 ms.
- JIT kernels at NV > 1: hypre matches the old build. Ginkgo's and PETSc's
  kernels are simpler (no rule, contiguous blocks) and their edge kernels
  take 40-56 registers against the old 128, so three times as many threads
  run at once, and their scattered d x d atomics overflow the 50 MB L2 (ncu,
  PETSc NV = 6: L2 hit rate 51% against 91%, 4.5x the DRAM bytes). At n <=
  256 the old and new kernels time the same for all three. This is launch
  tuning (occupancy for scattered atomics), not indexing, and is left for
  later. It is also the shuffled grid's: on a local ordering (below) every
  library assembles in 3.0-3.5 ms at NV = 4.
- Offline kernels take the form at run time and branch on it, the same way
  for every element; at NV = 4 they are 2-3x the old ones. They run until
  the JIT kernel is built (about ten calls).
- Across formats (JIT, step 5): NV = 4 hypre 8.0 ms, Ginkgo csr 10.3, fbcsr
  7.2, ell and sellp 20, mock bell 12.2, bell:column 7.0, interleaved 20.4,
  compact 16.6, compact:column 7.2, AmgX csr 9.9, bcsr 16.6; NV = 6 between
  9.3 (hypre) and 32.5 ms (sellp). The spread is the same occupancy effect
  and the strides of each format; nothing format-specific is tuned.
- The earlier write-up said the default edge loop first ran as forced
  chunk_atomics and later as atomics, "the JIT off too". On NVIDIA
  chunk_atomics is off by default, and forced it is refused here (low
  compression) and falls through to atomics: what changed after about ten
  calls was the atomics kernel, offline to JIT, which the ladder prefers on a
  register prior without timing it (old build, NV = 4: 4.83 ms offline
  against 7.28 ms JIT). The "JIT off" run had used `OP_JIT_ENABLE`, which is
  no longer read. `OP_AUTOTUNE=select` finds the offline kernel faster
  (1.485x) but had not finished exploring after 100 calls.

### Mesh order (op_renumber)

The proxy's grid is shuffled (nodes and edges), and `-mesh` reads real meshes
instead: airfoil (`new_grid_dp.h5`, 0.72M nodes, 1.44M edges with the
boundary ones), MG-CFD's Rotor37 1M (1.14M nodes, 3.36M edges) and 8M (8.35M
nodes, 24.9M edges), each in its file's order, with the same operator on the
mesh's graph from the bump. `-reorder 1` registers the coordinates and calls
`op_renumber`: `OP_REORDER=hilbert`, `rcm`, or `random` (random nodes, the
edges still ordered by their centroids, so not as bad as the grid's shuffle).
MPI+CUDA variants on one H100 rank, JIT kernels from the first call, CG +
Jacobi to 1e-8, step 5. Iterations are the same in every order. Logs:
`logs/perf_reorder.log`, `logs/perf_reorder_sweep.log`.

Grid, n = 1024, NV = 4 (assembly loops per step, solve per iteration):

| | loops: shuffled / Hilbert / RCM | solve: shuffled / Hilbert / RCM |
|---|---|---|
| hypre | 8.29 / 3.35 / 3.47 ms | 1.412 / 1.365 / 1.364 ms |
| Ginkgo csr | 10.06 / 3.39 / 3.48 | 1.620 / 1.509 / 1.518 |
| PETSc coo | 16.57 / 3.43 / 4.29 | 2.913 / 2.876 / 2.863 |
| AmgX bcsr | 16.61 / 3.48 / 4.49 | 1.041 / 0.907 / 0.907 |
| mock bell | 13.01 / 3.03 / 3.12 | 0.942 / 0.803 / 0.801 |

Rotor37 8M, NV = 4 (33.4M unknowns), in the file's order / random /
Hilbert / RCM:

| | loops | solve per iteration |
|---|---|---|
| hypre | 37.2 / 41.8 / 37.4 / 39.2 ms | 12.09 / 14.09 / 12.09 / 12.08 ms |
| Ginkgo csr | 39.0 / 45.3 / 39.7 / 41.1 | 12.74 / 14.51 / 12.72 / 12.75 |
| PETSc coo | 36.3 / 52.8 / 36.4 / 48.4 | 25.56 / 26.59 / 24.64 / 24.65 |
| AmgX bcsr | 36.5 / 53.2 / 36.6 / 47.4 | 8.13 / 10.91 / 7.89 / 8.06 |
| mock bell | 33.0 / 41.8 / 34.5 / 36.5 | 7.05 / 9.57 / 7.10 / 7.05 |
| hypre, NV = 1 | 2.24 / 3.30 / 2.33 / 2.41 | 1.64 / 3.30 / 1.64 / 1.64 |
| Ginkgo, NV = 1 | 2.27 / 3.31 / 2.36 / 2.44 | 1.71 / 3.57 / 1.70 / 1.70 |

- Order matters to both. The shuffled grid's loops take 2.5-4.8x as long as
  Hilbert's, and its solves up to 17% longer (AmgX and the mock, whose block
  formats gain most). On Rotor37 8M random nodes double the scalar solve
  (NV = 1) and add 4-36% at NV = 4 (PETSc least); on airfoil and Rotor37 1M
  up to 23% (PETSc's not at all), as more of the vectors stay in L2.
- The meshes come well ordered: airfoil's and MG-CFD's file orders time as
  Hilbert does within a few percent (Hilbert's PETSc solve is 4-7% faster on
  Rotor37). Real meshes need little reordering for this, but a
  partitioner's output does (the shuffle stands for it).
- RCM solves as fast as Hilbert, but its loops are slower for the formats
  whose blocks are contiguous in slot order (PETSc coo, AmgX bcsr: 30-45% on
  Rotor37 and airfoil, 25-30% on the grid); not looked into further.
- PETSc's and AmgX's loops (contiguous blocks, the leanest kernels) are the
  most sensitive to order: random nodes add 72-81% on airfoil and Rotor37
  1M, against 6-15% for hypre, Ginkgo and the mock. On a local order every
  format assembles at the same speed.

Strategies and chunk compression, Ginkgo csr, NV = 4 (loops per step;
compression of the matrix edge loop, then of the vector edge loop
`apply_edge`):

| | atomics | colouring | chunk_atomics | chunk_colouring | compression |
|---|---|---|---|---|---|
| grid, shuffled | 10.19 ms | 23.35 | refused | 10.27 | 1.00x, 1.00x |
| grid, Hilbert | 3.40 | 21.75 | 8.44 | 4.00 | 1.52x, 3.19x |
| airfoil | 2.35 | 14.29 | refused | 3.06 | 1.33x, 1.98x |
| airfoil, Hilbert | 2.36 | 14.48 | 5.73 | 2.77 | 1.51x, 3.10x |
| Rotor37 1M | 5.13 | 36.76 | refused | 7.62 | 1.33x, 2.00x |
| Rotor37 1M, Hilbert | 5.15 | 36.99 | refused | 7.70 | 1.38x, 2.21x |
| Rotor37 8M | 39.16 | 289.7 | refused | 53.43 | 1.33x, 1.98x |
| Rotor37 8M, Hilbert | 39.60 | 294.3 | refused | 52.20 | 1.37x, 2.16x |

A matrix edge loop's off-diagonal blocks are each one edge's, so only the
diagonal blocks are shared within a chunk: 1.33-1.52x at best, near the
default `OP_CHUNK_ATOMICS_MIN_COMPRESSION` of 1.5, which refuses chunk_atomics
everywhere but the Hilbert-ordered 2D meshes. Where it runs it takes 2.4x
atomics' time; chunk_colouring takes 17-50% longer than atomics on the
ordered meshes, and colouring 6-7x. Atomics (the NVIDIA default) is the
right strategy for matrix assembly on the H100 once the mesh is ordered.
Vector edge loops compress 2-3x.

### AMG (setup every step; measured on the earlier build, whose solve path is unchanged)

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

## Library-owned storage and formats: findings

- Every format of every library runs on every variant and component count
  (above). A library that lays out its own matrix is checked against the
  layer's builders: rocALUTION converts the pattern itself and its index
  arrays must equal the builder's; PETSc's own AIJ arrays are located from
  its row pointers and columns.
- Ginkgo 1.11 fails block Jacobi on Ell past 2^21 rows on a rank: its
  conversion to Csr launches a column reduction with grid y = rows / 32, past
  CUDA's 65535 (`cudaErrorInvalidDevice` reported later by thrust). Upstream
  has fixed it (develop clamps grid y and strides); the build here carries
  that change as `/home/jk-ext/solvers/patches/ginkgo-1.11-col-reduction-grid-y.patch`,
  which `build_cuda.sh` applies. Ell at NV = 4 now solves at n = 1024.
- Sellp slices are whole blocks (the multiple of d nearest Ginkgo's 64 rows),
  since a block's rows must share a slice; 64-row slices refused d = 6 and 7.
- AmgX has block kernels for 1, 3 and 4 components only; 6 x 6 blocks failed
  to launch, and the next run hung the GPU (old driver). `bcsr` is refused
  for other sizes and the default there is `csr`. `bcsr:diag` (the diagonal
  blocks apart) is AmgX's single-rank upload only.
- PETSc's `aij:owned` (OP2 writing PETSc's own arrays) works on the host. On
  the device PETSc moves and refreshes its arrays behind a borrower, and its
  PBJacobi reads host values: experimental, kept out of the GPU tests.
- rocALUTION is a host build here; its GPU backend (HIP) is not built.
- Lending libraries now have one base: a lent region 0 holds the scratch
  slots after the library's values, so hypre, Ginkgo (but Sellp), PETSc's
  aij and coo and AmgX keep "one region" under MPI; the mock, rocALUTION,
  Sellp and `aij:owned` keep their parts apart ("any region") on more than
  one rank.

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

- Mesh order matters more than storage format: shuffled against Hilbert,
  2.5-4.8x on the grid's assembly and up to 17% on its solve, and random
  nodes double Rotor37 8M's scalar solve. Each rank's mesh should be
  renumbered after partitioning (`op_renumber`, Hilbert given coordinates);
  with that, atomics assembles every format alike.
- Assembly into solver memory costs milliseconds beside solves of ~0.1-2 s
  per step here; for Hydra's flux Jacobians the loops are heavier but the
  commit saved is the same bandwidth-bound pass.
- A library's own layout costs the loops nothing extra to reach: the mock
  allocates its local and non-local parts apart, in block CSR or block ELL,
  row or column major, interleaved or not, and the loops write them in place
  through the same layout as OP2's own regions. That is SPLISS's model.
- Writing in place is where the memory goes: PETSc's COO path holds 2.2x the
  device memory of hypre at NV = 6; formats in place hold only their index
  arrays, and block formats (fbcsr, block ELL) the fewest.
- Par-loop preconditioners cost two vector copies per application, nothing
  measurable; the par-loop block Jacobi is 2.2x faster than PETSc's own
  PCPBJACOBI on the GPU.
- hypre's BoomerAMG on the GPU is the fastest AMG here, and the only one that
  handles the coupled 4-component system. Ginkgo's Pgm works on the decoupled
  P (`-pmat decoupled`: 88 iterations where A gives no convergence), the
  intended use of a separate preconditioning matrix.

## Custom preconditioners: the answer

op_par_loops plus `op_arg_mat` are enough. The four mechanisms in the design
(assembled P, par-loop preconditioner callback, `op_arg_mat(OP_READ)` in those
loops, par-loop operator) are implemented and tested on PETSc, hypre, Ginkgo,
rocALUTION and the mock (AmgX takes no callbacks), CPU and GPU, with block Jacobi and a Neumann series as working examples.
Nothing else was needed: the library sees an operator, and the preconditioner
is ordinary OP2 code with access to the assembled matrix in the library's own
storage.

## Limitations and next steps

- Assembly kernels are not tuned (above): JIT kernels for contiguous and
  rule-free layouts lose to the old ones at NV > 1 by occupancy, and offline
  kernels branch on the form. The ladder also swaps a faster offline kernel
  for its JIT kernel untimed (an rt issue, not the solver layer's).
- The layout is 16 bytes per slot on the device, twice the values at NV = 1;
  a contiguous matrix's kernels never read it there and could go without.
- Fortran: the API is C, so a binding is thin, but the Fortran translator
  needs `op_arg_mat` (the f2c path would reuse the C++ templates' BlockPtr).
  Hydra needs this next, with SPLISS behind a backend like the mock's.
- PETSc in place on the device (`aij:owned`) is experimental (above); the
  device default is COO, one copy of the values plus 8-byte maps.
- The par-loop operator is not supported by the hypre backend (ParCSR Krylov
  solvers take a matvec only through hypre's generic interface), and AmgX
  takes no par-loop callbacks.
- Slot maps cost as much as the scalar values: the diagonal blocks of each
  edge could come from the node map, halving them.
- At most 2^31 values of a matrix per rank (above). Going past needs 64-bit
  local indices in the libraries (Ginkgo int64, PETSc --with-64-bit-indices;
  hypre has none on the GPU).
- Scratch slots are one per (element, block) write to a halo row; deduplicated
  per block they would shrink with heavy halos.
- Only square matrices on one set, "double", AoS vectors, full (not partial)
  halo exchanges; CUDA tested (HIP compiles the same code but needs HIP builds
  of the libraries); the legacy C++ targets reject matrix blocks.
- A format whose blocks are not at fixed strides would need a new layout
  rule; none of the formats here needed one.

## Building and running

```
# Solver libraries (already built): /home/jk-ext/solvers/{cpu,cuda},
# PETSc in /home/jk-ext/solvers/src/petsc{,-cuda}/arch-{cpu,cuda}
cmake -S /home/jk-ext/op2-solvers -B build -DOP2_ENABLE_APPS=ON -DOP2_ENABLE_TESTS=ON \
  -DOP2_ENABLE_PETSC=ON -DPETSC_DIR=... -DPETSC_ARCH=... -DOP2_ENABLE_HYPRE=ON \
  -DOP2_ENABLE_GINKGO=ON -DOP2_ENABLE_AMGX=ON -DOP2_ENABLE_ROCALUTION=ON -DOP2_ENABLE_SOLVE_MOCK=ON
OP_SOLVE_CONFIG='{"library": "ginkgo", "matrices": {"A": {"format": "fbcsr"}}}' \
  ./build/apps/c/implicit/implicit4_cuda -n 1024 -steps 10 -init bump -config /home/jk-ext/solvers/configs/bench.json
ctest --test-dir build -L solve
```

`/home/jk-ext/solvers/scripts/` has the exact configurations: `build_cpu.sh`,
`build_cuda.sh`, `build_amgx.sh` (the libraries; `ONLY=<library>`; hypre with
mixed integers, Ginkgo with the patch above), `op2_cpu_build.sh`,
`gpu_configure.sh` (this tree; the GPU one also needs
`CMAKE_CUDA_HOST_COMPILER` and `CUDA_TOOLKIT_ROOT_DIR` for hypre's Umpire/BLT
config), `gpu_solve_tests.sh`, `perf_formats.sh` (the format table; `NVS`,
`FORMATS`, stops at the first device error), `perf_reorder.sh` (mesh order:
`MESHES`, `SWEEP_ONLY`), `perf_limit.sh` (the runs at the 32-bit limit),
`jit_regs_implicit.sh` (the JIT kernels' registers). Meshes:
`~/mgcfd-meshes/rotor37_{1m,8m}/Rotor37_{1M,8M}/grid.1.hdf`, airfoil
`~/op2-loop/build-cpu/test-meshes/new_grid_dp.h5`; reordering needs an MPI
variant (`OP_REORDER`, `mpirun -n 1 ./implicit4_mpi_cuda -mesh FILE -reorder 1
-init bump -dt 0.01`).
Variables: `OP_SOLVE_CONFIG` (JSON, or a file of it, over the application's
configuration), `OP_SOLVE_MEMORY=1` (memory per stage and the matrix's storage
form). The benchmarks' settings, CG and each library's Jacobi, are
`/home/jk-ext/solvers/configs/bench.json`, with hypre's Umpire pool at 64 MiB
(it reserves 4 GiB by default). The figures above were taken with the generic
options these replace; they are the same settings, but for Ginkgo's Jacobi,
which now finds its own blocks (up to 32) where the old option gave it blocks
of d. To time kernels: `OP_JIT_LEVEL=0` for offline
only, `OP_JIT_MIN_CALLS=1 OP_JIT_SEQ_COMPILE=1` for JIT from the first call.
