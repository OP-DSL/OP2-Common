### Headers

Every OP2 header lives here, the library's private ones too, and no header
lives under `src/`.  Two kinds sit side by side:

- the flat headers (`op_lib_core.h`, `op_seq.h`, ...), the long-standing C
  API, included by name (`#include <op_seq.h>`);
- the `op2/` tree, included by its path (`#include <op2/rt/gpu/loop.h>`),
  whose path below `op2/` is its namespace below `op::`: `op2/rt/gpu/loop.h`
  declares into `op::rt::gpu`, and a namespace with one header may be that
  file (`op2/env.h` is `op::env`, `op2/rt/lifecycle.h` is `op::rt`).

This directory installs as `include/op2`, the package's one include
directory, so both spellings work the same in the source tree and once
installed (the `op2/` tree as `include/op2/op2/`).  `op2/rt/` is the loop
runtime, what generated code compiles against and calls to run its loops,
with its sources under `src/rt/`.  The rest of the library includes from it
only `op2/rt/lifecycle.h` (`scripts/check_rt_layering.sh`, the `rt_layering`
test, checks it).

A header is header-only unless its entry names where its definitions are.
`includes:` lists a header's direct includes from this directory.

### The flat headers

#### Core components
```
op_lib_c.h      (includes: op_lib_core.h, op_rt_support.h, op_hdf5.h)
op_lib_core.h   (includes: SafeLong.h)
op_lib_cpp.h    (includes: op_lib_c.h, op_lib_core.h)
op_rt_support.h (includes: op_lib_core.h; op_plan, the legacy targets' plans)
op_profile.h    (includes: extern/json.hpp; the profiler's timing tree and
                 its C API, op_profile_*; from src/externlib/op_profile.cpp)
SafeLong.h      (a long long whose lossy conversions and division by zero
                 throw, with a stack trace; from src/core/SafeLong.cpp)
```

#### Sequential layer
```
op_seq.h   (auto-generated, includes: op_lib_cpp.h)
```

#### CUDA layer
```
op_cuda_rt_support.h (includes: op2/gpu/shims.h, op_lib_core.h,
                      op_rt_support.h)
op_cuda_reduction.h  (includes: op2/gpu/shims.h; kept apart for the Fortran
                      layer)
```

#### MPI support
```
op_lib_mpi.h   (includes: op_lib_core.h, op_mpi_core.h; the MPI library's
                entry points)
op_mpi_core.h  (the halo and partition records the MPI backends share, the
                communicators and reductions)
op_util.h      (allocation, search and file helpers)
```

#### HDF5 I/O
```
op_hdf5.h  (op_decl_*_hdf5, op_get_const_hdf5, op_dump_to_hdf5; from
            src/externlib/op_hdf5.cpp, src/mpi/op_mpi_hdf5.cpp under MPI)
```

#### Fortran library (sources under src/fortran)
```
fortran/op2_for_C_wrappers.h    (includes: op_lib_core.h; the C side of the
                                 Fortran declarations, and the access codes
                                 both sides share)
fortran/op2_for_rt_wrappers.h   (the Fortran runtime's C entry points:
                                 partitioning, map index conversion, plans)
fortran/op2_C_reference.h       (includes: fortran/op2_for_C_wrappers.h,
                                 fortran/op2_reference_macros.h,
                                 op_lib_core.h; the reference library's loop
                                 helpers)
fortran/op2_reference_macros.h  (the argument-list macros the reference
                                 loops, C and Fortran, expand)
```

#### Third party
```
extern/json.hpp      (nlohmann JSON, for op_profile's JSON output)
extern/incbin.h      (binary inclusion, for the JIT's embedded headers)
extern/rapidhash.h   (rapidhash, for call keys and JIT parameter values)
```

### The op2/ tree

#### Core
```
op2/env.h        (op::env: the environment variables' parser: every layer
                  reads its own through it as op_init runs, and op_init warns
                  about any OP_ variable nothing read; from
                  src/core/op_env.cpp)
op2/gpu/shims.h  (gpu* aliases over the CUDA or HIP runtime)
op2/gpu/check.h  (includes: op2/gpu/shims.h; GPU_CHECK, which stops the
                  program on a failed runtime, driver or compiler call)
```

#### MPI (op::mpi, sources under src/mpi)
```
op2/mpi/halo.h              (includes: op_lib_core.h, op_mpi_core.h; the halo
                             lists op_halo_create builds)
op2/mpi/comm.h              (the sparse data exchange (NBX) halo construction
                             uses to find what each rank receives)
op2/mpi/exchange.h          (includes: op_lib_core.h, op_lib_c.h,
                             op_lib_mpi.h; the halo exchange's gathers and
                             scatters, compiled for the host and the device)
op2/mpi/exchange_backend.h  (includes: op2/mpi/exchange.h; the buffers and
                             exchange specifications a backend gives the
                             exchange driver)
```

#### Renumbering (op::renumber, used by src/externlib/op_renumber.cpp)
Graph and geometric orderings with no OP2 dependencies, each producing a
permutation.
```
op2/renumber/rcm.h          (reverse Cuthill-McKee, and the BFS helpers
                             Sloan shares)
op2/renumber/sloan.h        (includes: op2/renumber/rcm.h; Sloan's
                             profile-minimising ordering)
op2/renumber/hilbert_sfc.h  (the Hilbert space-filling curve over 2D or 3D
                             coordinates)
```

### The loop runtime (op2/rt/, sources under src/rt)

```
op2/rt/lifecycle.h  (op::rt: the runtime's start, reports and release,
                     which op_init, op_profile_output and op_exit call;
                     from src/rt/op_rt_lifecycle.cpp, init_cpu from
                     src/rt/cpu/op_cpu_init.cpp, init_gpu from
                     src/rt/gpu/op_gpu_options.cpp)
```

#### par_loop support, backend neutral (op::rt::loop, sources under src/rt/loop)
```
op2/rt/loop/strategy.h  (includes: op_lib_core.h; a loop's strategies, each
                         with its kernels, and the candidates the ladder
                         runs and the tuner times, a strategy with one
                         kernel each, in an order from the backend's
                         estimates; the axes a family's strategies vary
                         along, the families a backend declares, the
                         run-wide variables op_init reads, and the sections
                         an indirect loop runs)
op2/rt/loop/autotune.h  (includes: op2/rt/loop/strategy.h; the autotuner
                         that times every candidate per configuration once
                         its JIT modes have settled)
op2/rt/loop/jit.h       (includes: extern/rapidhash.h, op_lib_core.h,
                         op2/rt/loop/strategy.h; the JIT's options, read by
                         op_init, the parameters a loop's kernels are
                         specialised on and the statistics that move them
                         between baked in and run time, and a strategy's
                         cache of built kernels; a backend supplies the
                         mechanism)
op2/rt/loop/loop.h      (includes: op_lib_core.h, op2/rt/loop/autotune.h,
                         op2/rt/loop/jit.h, op2/rt/loop/strategy.h; the base
                         of every backend's loop: its ladder, tuner and JIT
                         parameters, a call's steps and profile scopes, with
                         hooks for the backend's globals, and the registry
                         of every loop, which op_profile_output and op_exit
                         report on and op_exit releases; the call from
                         src/rt/loop/op_loop_call.cpp, the reports from
                         src/rt/loop/op_loop_report.cpp)
```

#### Planning building blocks, backend neutral (op::rt::plan, sources under src/rt/plan)
Algorithms with no device knowledge and no device limits, which each
backend composes into its own plans (op2/rt/gpu/chunk_plan.h,
op2/rt/cpu/chunk_plan.h).
```
op2/rt/plan/colouring.h       (includes: op_lib_core.h,
                               op2/rt/loop/strategy.h; the set colouring of a
                               call's sections, as op_plan's OP_COLOR2)
op2/rt/plan/chunks.h          (includes: op2/rt/loop/strategy.h; the
                               references a planner follows, chunks within
                               sections, each chunk's targets, coloured
                               batches, the wavefront order, element
                               colours, and the result a backend's plan comes
                               back in)
op2/rt/plan/staging.h         (includes: op2/rt/plan/chunks.h,
                               op2/rt/plan/staging_layout.h; the staged
                               kernels' argument groups, scratch slots and
                               stage words, and the pipeline staging a call's
                               chunks under a backend's lanes and scratch
                               limit)
op2/rt/plan/staging_layout.h  (the staging layout both the host and a staged
                               kernel read; no includes, so NVRTC compiles it
                               too)
op2/rt/plan/options.h         (the chunked strategies' run-wide options, read
                               by op_init)
op2/rt/plan/cache.h           (includes: extern/rapidhash.h,
                               op2/rt/loop/strategy.h, op2/rt/plan/staging.h;
                               the plans strategies share, of any loop: a
                               planner's key over what it plans from, the
                               live plans by key, and each strategy's by
                               configuration, which the tuner has it forget
                               once explored)
op2/rt/plan/greedy_colour.h   (includes: op_lib_core.h; internal to the
                               planners: the greedy colouring they share)
```

#### Translated kernels (op::rt::kernel; C++ cuda, hip and openmp, Fortran's _f2c targets)
```
op2/rt/kernel/prelude.h   (what a translated kernel compiles against: Ptr,
                           Span, the Fortran intrinsics; no includes)
op2/rt/kernel/strategy.h  (includes: op2/rt/kernel/prelude.h,
                           op2/rt/plan/staging_layout.h; the increment hooks
                           and the chunked strategies' device helpers)
```

#### CUDA and HIP backend of the translated kernels (op::rt::gpu, sources under src/rt/gpu)
```
op2/rt/gpu/options.h       (the run-wide options, the JIT mechanism's and the
                            wavefront window, and the device's facts; read by
                            op_init, which declares the GPU families, from
                            src/rt/gpu/op_gpu_options.cpp)
op2/rt/gpu/jit.h           (includes: op2/gpu/check.h, op2/gpu/shims.h,
                            op_lib_core.h, op2/rt/gpu/options.h,
                            op2/rt/loop/jit.h; the JIT mechanism, whose
                            sources include op2/rt/kernel/prelude.h and
                            op2/rt/kernel/strategy.h: offline kernels and
                            their symbols, NVRTC or hipRTC compiles, run-time
                            parameters as wrapper arguments or module
                            symbols, launches)
op2/rt/gpu/jit_embed.h     (includes: extern/incbin.h; embeds the three
                            kernel headers as text for NVRTC; define
                            OP_KERNEL_JIT_TAG first)
op2/rt/gpu/chunk_plan.h    (includes: op2/rt/plan/chunks.h,
                            op2/rt/plan/staging.h; the chunk plans under the
                            device's limits: a block's elements, shared
                            memory, 16-bit stage words, 8-bit colours)
op2/rt/gpu/chunk_encode.h  (includes: op2/gpu/check.h, op2/gpu/shims.h,
                            op2/rt/gpu/chunk_plan.h; a chunk plan uploaded as
                            the kernels read it)
op2/rt/gpu/strategy.h      (includes: op_lib_cpp.h, op2/rt/gpu/chunk_encode.h,
                            op2/rt/gpu/chunk_plan.h, op2/rt/gpu/jit.h,
                            op2/rt/loop/strategy.h, op2/rt/plan/cache.h,
                            op2/rt/plan/colouring.h; the plain, atomics,
                            colouring and chunked strategies, and their axes:
                            block size, unstaged singles, wavefront)
op2/rt/gpu/loop.h          (includes: op2/gpu/check.h, op_cuda_rt_support.h,
                            op_lib_cpp.h, op_profile.h, op2/rt/gpu/jit.h,
                            op2/rt/gpu/strategy.h, op2/rt/loop/loop.h,
                            op2/rt/loop/strategy.h, op2/rt/plan/options.h;
                            the loop the generated code builds, registers its
                            parameters on and invokes)
```

#### OpenMP backend of the translated kernels (op::rt::cpu)
```
op2/rt/cpu/chunk_plan.h  (includes: op2/rt/plan/chunks.h,
                          op2/rt/plan/staging.h; the chunk plans a host
                          thread runs a chunk of on its own)
op2/rt/cpu/strategy.h    (includes: op_lib_core.h, op2/rt/cpu/chunk_plan.h,
                          op2/rt/loop/strategy.h, op2/rt/plan/cache.h,
                          op2/rt/plan/colouring.h; the plain, atomics,
                          colouring and chunked strategies over host
                          threads, the per-thread globals, and the families
                          the CPU libraries' op_init declares)
op2/rt/cpu/loop.h        (includes: op_lib_c.h, op_profile.h,
                          op2/rt/cpu/strategy.h, op2/rt/loop/loop.h,
                          op2/rt/loop/strategy.h, op2/rt/plan/options.h; the
                          loop the generated code builds and invokes)
```
