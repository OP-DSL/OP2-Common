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
and is layered by where it is compiled:

- `op2/rt/abi/` is the contract between the generated code and the
  library: the descriptors a loop describes itself with, the records a
  launch or an OpenMP region is given, and the entry points.  It includes
  nothing else of `op2/rt/`;
- `op2/rt/kernel/` is compiled into op2_kernels, the generated code, and
  never into a library: the skeletons each strategy family runs, filled with
  the generated hooks, the helpers they share, and the calls on what the
  application compiled, its offline kernels' launches and its `__constant__`
  symbols.  It includes only `op2/rt/abi/` and itself;
- the rest of `op2/rt/` is the library's: declarations, templates and
  one-line accessors, defined under `src/rt/`, which never include
  `op2/rt/kernel/`.  The generated code includes from `op2/rt/` only
  `op2/rt/abi/` and `op2/rt/kernel/`.

The rest of the library includes from `op2/rt/` only `op2/rt/lifecycle.h`.
`scripts/check_rt_layering.sh`, the `rt_layering` test, checks each rule.

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

#### The contract with the generated code (op::rt::abi)
What the generated code and the library agree on: plain types, constants and
entry points, with no behaviour behind them.  A loop describes itself to the
library once, as a static descriptor, and runs each call through its
backend's invoke; the library hands each launch or OpenMP region a record.
Includes nothing else of `op2/rt/`; what a kernel reads compiles for the
device, under NVRTC and hipRTC too.
```
op2/rt/abi/descriptor.h      (what any backend's descriptor holds: the
                              arguments a staged kernel groups, and the
                              parameters a loop's kernels may be specialised
                              on, as the JIT takes them, built with constant,
                              scalar_global, global, dat_stride and dat_dim)
op2/rt/abi/staging.h         (the staging layout the host planner writes and
                              a staged kernel reads: stage words, the slot
                              limit, the unstaged word; no includes)
op2/rt/abi/gpu/launch.h      (includes: op2/rt/abi/staging.h; the CUDA and
                              HIP launch records, a range's and a chunk
                              batch's, as X-macro field lists: the library
                              fills a record and pushes its fields, a kernel
                              takes them as its first parameters
                              (OP_ABI_GPU_PARAM); the default block size)
op2/rt/abi/gpu/descriptor.h  (includes: op_lib_core.h,
                              op2/rt/abi/descriptor.h,
                              op2/rt/abi/gpu/launch.h; a loop as the CUDA and
                              HIP code describes it: its families' offline
                              kernels and JIT sources, its parameters with
                              their device side (a __constant__ symbol, its
                              shared hash, by value), the binder of its
                              arguments, the calls on what the application
                              compiled, an offline kernel's resources among
                              them, and the embedded JIT headers; invoke,
                              from src/rt/gpu/op_gpu_loop.cpp)
op2/rt/abi/cpu/launch.h      (includes: op2/rt/abi/staging.h; the OpenMP
                              records, a range's and a chunk batch's, and a
                              thread's own part: its globals and scratch)
op2/rt/abi/cpu/descriptor.h  (includes: op_lib_core.h,
                              op2/rt/abi/cpu/launch.h,
                              op2/rt/abi/descriptor.h; a loop as the OpenMP
                              code describes it: each family's run over a
                              range or over chunks, the host's thread count;
                              invoke, from src/rt/cpu/op_cpu_loop.cpp)
```

#### Compiled into op2_kernels (op::rt::kernel; C++ cuda, hip and openmp, Fortran's _f2c targets)
Never compiled into a library.  The skeletons each strategy family runs,
which the generated code fills with its hooks, the element function first,
so a loop's code inlines into them; the device helpers its kernels share;
and the calls on what the application compiled.  Includes only
`op2/rt/abi/` and `op2/rt/kernel/`, and the JIT compiles the device headers
again from the embedded text.
```
op2/rt/kernel/prelude.h        (what a translated kernel compiles against:
                                Ptr, Span, the Fortran intrinsics, the
                                parameters a JIT kernel may bake in; no
                                includes)
op2/rt/kernel/inc.h            (includes: op2/rt/kernel/prelude.h; the
                                increment hooks, in place or atomic, and
                                opaque)
op2/rt/kernel/staging.h        (includes: op2/rt/abi/staging.h,
                                op2/rt/kernel/inc.h,
                                op2/rt/kernel/prelude.h; a chunk's regions
                                carved, cleared or seeded, and stored or
                                flushed to their targets, on the host and the
                                device)
op2/rt/kernel/gpu/block.h      (includes: op2/rt/kernel/prelude.h; the block's
                                thread count as a skeleton or hook reads it)
op2/rt/kernel/gpu/range.h      (includes: op2/rt/kernel/gpu/block.h,
                                op2/rt/kernel/prelude.h; the plain, atomics
                                and colouring kernels' grid-stride walk of a
                                range, with the element hook)
op2/rt/kernel/gpu/chunk.h      (includes: op2/rt/kernel/gpu/block.h,
                                op2/rt/kernel/gpu/wavefront.h,
                                op2/rt/kernel/inc.h,
                                op2/rt/kernel/prelude.h; chunk_atomics,
                                chunk_colouring and its parallel variant over
                                a batch of chunks, with the stage, element and
                                write hooks)
op2/rt/kernel/gpu/wavefront.h  (includes: op2/rt/kernel/gpu/block.h,
                                op2/rt/kernel/prelude.h; chunk_colouring's
                                wavefront primitives)
op2/rt/kernel/gpu/offline.h    (includes: op2/gpu/check.h, op2/gpu/shims.h,
                                op2/rt/abi/descriptor.h,
                                op2/rt/abi/gpu/descriptor.h; an offline
                                kernel's figures and launches and the device
                                addresses of the application's symbols, made
                                with the runtime that registered them)
op2/rt/kernel/gpu/jit_embed.h  (includes: extern/incbin.h; embeds the device
                                headers as text for the JIT, in include
                                order; define OP_KERNEL_JIT_TAG first)
op2/rt/kernel/cpu/range.h      (includes: op2/rt/abi/cpu/launch.h; the
                                OpenMP region over a range, a contiguous
                                share per thread)
op2/rt/kernel/cpu/chunk.h      (includes: op2/rt/abi/cpu/launch.h; the
                                OpenMP region over batches of chunks, each
                                batch's shared out over the threads)
```

#### A par_loop, backend neutral (op::rt::loop, sources under src/rt/loop)
```
op2/rt/loop/call.h      (includes: op_lib_core.h; one call: its set and
                         arguments, its configuration's key, the sections it
                         runs in, a phase's arguments, and the reduction of
                         its globals across the ranks; from
                         src/rt/loop/op_loop_call.cpp)
op2/rt/loop/strategy.h  (includes: op2/rt/loop/call.h; a strategy, with an
                         offline kernel, a specialised one or both, and its
                         backend's estimate of each: available or why not,
                         and a dimensionless cost the ladder orders them by)
op2/rt/loop/family.h    (the families a backend declares, the axes their
                         strategies vary along, and OP_STRATEGY, read by
                         op_init)
op2/rt/loop/ladder.h    (includes: op2/rt/loop/call.h, op2/rt/loop/family.h,
                         op2/rt/loop/strategy.h; a loop's candidates, a
                         strategy with one of its kernels each, in ladder
                         order, and the one each call runs; from
                         src/rt/loop/op_loop_ladder.cpp)
op2/rt/loop/loop.h      (includes: op_lib_core.h, op2/rt/jit/params.h,
                         op2/rt/loop/call.h, op2/rt/loop/ladder.h,
                         op2/rt/loop/strategy.h, op2/rt/tune/autotune.h; the
                         base of every backend's loop: its ladder, tuner and
                         JIT parameters, a call's steps, phases and profile
                         scopes, with hooks for the backend's globals, and
                         the registry of every loop, which op_profile_output
                         and op_exit report on and op_exit releases; from
                         src/rt/loop/op_loop.cpp, the reports from
                         src/rt/loop/op_loop_report.cpp)
```

#### Specialised kernels, backend neutral (op::rt::jit, sources under src/rt/jit)
The policy, the parameters and the build cache; a backend supplies the
compiler.
```
op2/rt/jit/options.h  (the JIT's policy as OP_JIT_* sets it, read by op_init:
                       whether and when kernels are specialised, what they
                       bake in, and its report)
op2/rt/jit/params.h   (includes: extern/rapidhash.h, op_lib_core.h,
                       op2/rt/abi/descriptor.h, op2/rt/jit/options.h,
                       op2/rt/loop/call.h; the parameters a loop's kernels
                       are specialised on, from its descriptor's rows and the
                       backend's, and the statistics that move each between
                       baked in and given at run time)
op2/rt/jit/cache.h    (includes: op_lib_core.h, op2/rt/jit/options.h,
                       op2/rt/loop/strategy.h; a strategy's built kernels by
                       key, and when each is built)
```

#### Choosing a strategy (op::rt::tune, sources under src/rt/tune)
```
op2/rt/tune/autotune.h    (includes: op2/rt/loop/call.h,
                           op2/rt/loop/ladder.h, op2/rt/loop/strategy.h; the
                           autotuner, which times every candidate per
                           configuration once its JIT modes have settled, and
                           its options, read by op_init)
```

#### Planning building blocks, backend neutral (op::rt::plan, sources under src/rt/plan)
Algorithms with no device knowledge and no device limits, which each
backend composes into its own plans (op2/rt/gpu/chunk_plan.h,
op2/rt/cpu/chunk_plan.h).
```
op2/rt/plan/colouring.h      (includes: op_lib_core.h, op2/rt/loop/call.h;
                              the set colouring of a call's sections, as
                              op_plan's OP_COLOR2)
op2/rt/plan/chunks.h         (includes: op2/rt/loop/call.h; the references a
                              planner follows, chunks within sections, each
                              chunk's targets, coloured batches, the
                              wavefront order, element colours, and the
                              result a backend's plan comes back in)
op2/rt/plan/staging.h        (includes: op2/rt/abi/descriptor.h,
                              op2/rt/abi/staging.h, op2/rt/loop/call.h,
                              op2/rt/plan/chunks.h; the staged kernels'
                              scratch slots and stage words, and the pipeline
                              staging a call's chunks under a backend's lanes
                              and scratch limit)
op2/rt/plan/options.h        (the chunked strategies' run-wide options, read
                              by op_init)
op2/rt/plan/cache.h          (includes: extern/rapidhash.h,
                              op2/rt/abi/descriptor.h, op2/rt/loop/call.h;
                              the plans strategies share, of any loop: a
                              planner's key over what it plans from, the live
                              plans by key, and each strategy's by
                              configuration, which the tuner has it forget
                              once explored)
op2/rt/plan/greedy_colour.h  (includes: op_lib_core.h; internal to the
                              planners: the greedy colouring they share)
```

#### CUDA and HIP backend (op::rt::gpu, sources under src/rt/gpu)
Internal to the library: the generated code reaches it through
`op2/rt/abi/gpu/descriptor.h`.
```
op2/rt/gpu/options.h       (the run-wide options, the JIT mechanism's and the
                            wavefront window, and the device's facts; read by
                            op_init, which declares the GPU families, from
                            src/rt/gpu/op_gpu_options.cpp)
op2/rt/gpu/heuristics.h    (includes: op2/rt/abi/gpu/descriptor.h,
                            op2/rt/loop/strategy.h; the models of what pays
                            on the device, which turn what the backend knows
                            of a strategy's kernels into its estimates, the
                            register rule first; OP_JIT_MIN_REGISTERS, read
                            by op_init; from src/rt/gpu/op_gpu_heuristics.cpp)
op2/rt/gpu/loop.h          (includes: op_lib_core.h, op2/gpu/shims.h,
                            op2/rt/abi/descriptor.h,
                            op2/rt/abi/gpu/descriptor.h,
                            op2/rt/abi/gpu/launch.h, op2/rt/loop/call.h,
                            op2/rt/loop/loop.h; the loop built from a
                            descriptor: its parameters' device symbols, its
                            compiler, a launch's arguments, a record's
                            fields then the loop's, and the call's device
                            globals; from src/rt/gpu/op_gpu_loop.cpp)
op2/rt/gpu/jit.h           (includes: op2/gpu/shims.h,
                            op2/rt/abi/descriptor.h,
                            op2/rt/abi/gpu/descriptor.h, op2/rt/jit/cache.h,
                            op2/rt/jit/params.h; the JIT mechanism: offline
                            kernels and the symbols they read, NVRTC or
                            hipRTC compiles of a family's source after the
                            embedded headers, run-time parameters as wrapper
                            arguments or module symbols, launches; from
                            src/rt/gpu/op_gpu_jit.cpp)
op2/rt/gpu/chunk_plan.h    (includes: op2/rt/plan/chunks.h,
                            op2/rt/plan/staging.h; the chunk plans under the
                            device's limits: a block's elements, shared
                            memory, 16-bit stage words, 8-bit colours; from
                            src/rt/gpu/op_gpu_chunk_plan.cpp)
op2/rt/gpu/chunk_encode.h  (includes: op2/rt/abi/gpu/launch.h,
                            op2/rt/abi/staging.h, op2/rt/gpu/chunk_plan.h; a
                            chunk plan uploaded as the kernels read it, which
                            fills a chunk launch's record; from
                            src/rt/gpu/op_gpu_chunk_encode.cpp)
op2/rt/gpu/strategy.h      (includes: op_lib_core.h,
                            op2/rt/abi/gpu/descriptor.h,
                            op2/rt/abi/gpu/launch.h,
                            op2/rt/gpu/chunk_encode.h,
                            op2/rt/gpu/chunk_plan.h, op2/rt/gpu/jit.h,
                            op2/rt/gpu/loop.h, op2/rt/gpu/options.h,
                            op2/rt/loop/strategy.h, op2/rt/plan/cache.h,
                            op2/rt/plan/colouring.h; the plain, atomics,
                            colouring and chunked strategies, their axes,
                            block size, unstaged singles and wavefront, and
                            their phases' launches; from
                            src/rt/gpu/op_gpu_strategy.cpp)
```

#### OpenMP backend (op::rt::cpu, sources under src/rt/cpu)
Internal to the library: the generated code reaches it through
`op2/rt/abi/cpu/descriptor.h`.
```
op2/rt/cpu/loop.h        (includes: op_lib_core.h,
                          op2/rt/abi/cpu/descriptor.h,
                          op2/rt/abi/cpu/launch.h, op2/rt/loop/call.h,
                          op2/rt/loop/loop.h; the loop built from a
                          descriptor: the call's threads, per-thread globals
                          and chunk scratch, and the records its phases run
                          with; from src/rt/cpu/op_cpu_loop.cpp)
op2/rt/cpu/chunk_plan.h  (includes: op2/rt/plan/chunks.h,
                          op2/rt/plan/staging.h; the chunk plans a host
                          thread runs a chunk of on its own; from
                          src/rt/cpu/op_cpu_chunk_plan.cpp)
op2/rt/cpu/strategy.h    (includes: op_lib_core.h, op2/rt/cpu/chunk_plan.h,
                          op2/rt/cpu/loop.h, op2/rt/loop/strategy.h,
                          op2/rt/plan/cache.h, op2/rt/plan/colouring.h; the
                          plain, atomics, colouring and chunked strategies
                          over host threads, each phase a record its
                          family's run is given, and the families the CPU
                          libraries' op_init declares; from
                          src/rt/cpu/op_cpu_strategy.cpp, the families from
                          src/rt/cpu/op_cpu_init.cpp)
```
