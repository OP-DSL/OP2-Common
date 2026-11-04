### Headers

Every OP2 header lives here, the library's private ones too, and no header
lives under `src/`.  Two kinds sit side by side:

- the flat headers (`op_lib_core.h`, `op_seq.h`, ...), the long-standing C
  API, included by name (`#include <op_seq.h>`);
- the `op2/` tree, included by its path
  (`#include <op2/rt/gpu/executor.h>`), whose path below `op2/` is its
  namespace below `op::`: `op2/rt/gpu/executor.h` declares into
  `op::rt::gpu`, and a namespace with one header may be that file
  (`op2/env.h` is `op::env`, `op2/rt/lifecycle.h` is `op::rt`).

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
op2/rt/context.h    (includes: op2/rt/jit/options.h, op2/rt/loop/family.h,
                     op2/rt/loop/registry.h, op2/rt/plan/options.h,
                     op2/rt/tune/options.h; op::rt: the run's context from
                     op_init to op_exit: its options, the families the
                     backend declared, what the backend keeps for its loops,
                     and the loops made; from src/rt/op_rt_lifecycle.cpp)
op2/rt/hasher.h     (includes: extern/rapidhash.h; a rapidhash over words
                     added one at a time, a block at a time)
op2/rt/lifecycle.h  (op::rt: the runtime's start, which makes its context,
                     its reports and its release, which op_init,
                     op_profile_output and op_exit call;
                     from src/rt/op_rt_lifecycle.cpp, init_cpu from
                     src/rt/cpu/op_cpu_init.cpp, init_gpu from
                     src/rt/gpu/op_gpu_options.cpp)
op2/rt/report.h     (op::rt::report: the text the reports share, counts,
                     CSV quoting, tallies, aligned tables and the CSV files;
                     from src/rt/op_rt_report.cpp)
```

#### The contract with the generated code (op::rt::abi)
What the generated code and the library agree on: plain types, constants and
entry points, with no behaviour behind them.  A loop describes itself to the
library once, as a static descriptor, and runs each call through its
backend's invoke; the library hands each launch or OpenMP region a record.
Includes nothing else of `op2/rt/`; what a kernel reads compiles for the
device, under NVRTC and hipRTC too.
```
op2/rt/abi/descriptor.h      (what any backend's descriptor holds: the loop
                              and its families, which each backend's
                              descriptor extends, the arguments a staged
                              kernel groups, the element types, and the
                              parameters a loop's kernels may be specialised
                              on, as the JIT takes them, built with
                              constant, scalar_global, global, dat_stride,
                              dat_dim and strategy_value)
op2/rt/abi/staging.h         (the 16-bit stage word both backends' staged
                              kernels read a slot as: the slot limit and the
                              unstaged word; no includes)
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
                              shared hash, by value) and the values a
                              strategy fixes, by launch-value slot, the
                              binder of its
                              arguments, the calls on what the application
                              compiled, an offline kernel's resources among
                              them, and the embedded JIT headers; invoke,
                              from src/rt/gpu/op_gpu_executor.cpp)
op2/rt/abi/cpu/launch.h      (includes: op2/rt/abi/staging.h; the OpenMP
                              records, a range's and a chunk batch's, and a
                              thread's own part: its globals and scratch)
op2/rt/abi/cpu/descriptor.h  (includes: op_lib_core.h,
                              op2/rt/abi/cpu/launch.h,
                              op2/rt/abi/descriptor.h; a loop as the OpenMP
                              code describes it: each family's run over a
                              range or over chunks, the host's thread count;
                              invoke, from src/rt/cpu/op_cpu_executor.cpp)
```

#### Compiled into op2_kernels (op::rt::kernel; C++ cuda, hip and openmp, Fortran's _f2c targets)
Never compiled into a library.  The skeletons each strategy family runs,
which the generated code fills with its hooks, the element function first,
so a loop's code inlines into them; the helpers its kernels share; and the
calls on what the application compiled.  Includes only `op2/rt/abi/` and
`op2/rt/kernel/`; what is GPU-only is under `op2/rt/kernel/gpu/`, and the JIT
compiles the device headers again from the embedded text.
```
op2/rt/kernel/prelude.h        (what a translated kernel compiles against:
                                Ptr, Span, the Fortran intrinsics; no
                                includes)
op2/rt/kernel/inc.h            (includes: op2/rt/kernel/prelude.h; the
                                increment hooks, in place or atomic, on the
                                host and the device)
op2/rt/kernel/staging.h        (includes: op2/rt/abi/staging.h,
                                op2/rt/kernel/inc.h,
                                op2/rt/kernel/prelude.h; a chunk's regions
                                carved, cleared or seeded, and stored or
                                flushed to their targets, on the host and the
                                device)
op2/rt/kernel/gpu/device.h     (includes: op2/rt/kernel/prelude.h; what the
                                skeletons and hooks share on the device: the
                                block size as a function a kernel inlines
                                reads it, and opaque)
op2/rt/kernel/gpu/params.h     (includes: op2/rt/kernel/prelude.h; how a
                                kernel has its parameters, baked in by the
                                JIT or given at run time, and small arrays
                                by value)
op2/rt/kernel/gpu/range.h      (includes: op2/rt/kernel/gpu/device.h,
                                op2/rt/kernel/prelude.h; the plain, atomics
                                and colouring kernels' grid-stride walk of a
                                range, with the element hook)
op2/rt/kernel/gpu/chunk.h      (includes: op2/rt/kernel/gpu/device.h,
                                op2/rt/kernel/gpu/wavefront.h,
                                op2/rt/kernel/prelude.h; chunk_atomics,
                                chunk_colouring and its parallel variant over
                                a batch of chunks, with the stage, element and
                                write hooks)
op2/rt/kernel/gpu/wavefront.h  (includes: op2/rt/kernel/gpu/device.h,
                                op2/rt/kernel/prelude.h; chunk_colouring's
                                wavefront primitives)
op2/rt/kernel/gpu/offline.h    (includes: op2/gpu/check.h, op2/gpu/shims.h,
                                op2/rt/abi/descriptor.h,
                                op2/rt/abi/gpu/descriptor.h; an offline
                                kernel's resources and launches and the device
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
op2/rt/kernel/cpu/threads.h    (the host's threads as the generated code's
                                OpenMP sees them, for the library, which is
                                built without OpenMP)
```

#### A par_loop, backend neutral (op::rt::loop, sources under src/rt/loop)
```
op2/rt/loop/call.h      (includes: op_lib_core.h, op2/rt/abi/descriptor.h;
                         one call: its set and arguments, the sections it
                         runs in, which globals it reduces or writes, its
                         configuration's key, a run's arguments, a global's
                         element type and their reduction across the ranks;
                         from src/rt/loop/op_loop_call.cpp)
op2/rt/loop/strategy.h  (includes: op2/rt/loop/call.h; a strategy, with an
                         offline kernel, a specialised one or both, its
                         backend's estimate of each, available or why not
                         and a dimensionless cost the ladder orders them by,
                         and its answer for a call)
op2/rt/loop/family.h    (includes: op2/rt/loop/strategy.h; the families a
                         backend declares, the axes their strategies vary
                         along, a family as a backend offers it with the
                         strategy it makes at each point, and the declared
                         families' settings with OP_STRATEGY, as the run's
                         context holds them; from
                         src/rt/loop/op_loop_family.cpp)
op2/rt/loop/ladder.h    (includes: op2/rt/jit/options.h, op2/rt/loop/call.h,
                         op2/rt/loop/family.h, op2/rt/loop/strategy.h; a
                         loop's candidates, a
                         strategy with one of its kernels each, in ladder
                         order, the families the loop registers as its
                         backend offers them, and the one each call runs;
                         from src/rt/loop/op_loop_ladder.cpp)
op2/rt/loop/loop.h      (includes: op_lib_core.h, op_profile.h,
                         op2/rt/abi/descriptor.h, op2/rt/context.h,
                         op2/rt/jit/params.h, op2/rt/loop/call.h,
                         op2/rt/loop/ladder.h, op2/rt/loop/strategy.h,
                         op2/rt/tune/autotune.h; the loop a descriptor's
                         invoke runs, made in the run's context: its JIT
                         parameters, its backend's executor (what the
                         strategies share, and where the data lives for the
                         halo exchanges and dirty bits), its ladder and
                         tuner, and a call's steps, phases and profile
                         scopes; from src/rt/loop/op_loop.cpp)
op2/rt/loop/registry.h  (includes: op2/rt/abi/descriptor.h; every loop the
                         run made, which the context owns until op_exit, and
                         the reports op_profile_output and op_exit write from
                         them; from src/rt/loop/op_loop_report.cpp)
```

#### Specialised kernels, backend neutral (op::rt::jit, sources under src/rt/jit)
The policy, the parameters, the build cache and the report; a backend
supplies the compiler.
```
op2/rt/jit/options.h  (the JIT's policy as OP_JIT_* sets it, read into the
                       run's context: whether and when kernels are
                       specialised, what they bake in, and its report)
op2/rt/jit/params.h   (includes: extern/rapidhash.h, op_lib_core.h,
                       op2/rt/abi/descriptor.h, op2/rt/hasher.h,
                       op2/rt/jit/options.h, op2/rt/loop/call.h; the
                       parameters a loop's kernels are specialised on, from
                       its descriptor's rows, a dat's stride as the backend
                       lays the dat out, and the statistics that move each
                       between baked in and given at run time)
op2/rt/jit/cache.h    (includes: op_lib_core.h, op2/rt/jit/options.h,
                       op2/rt/loop/strategy.h; a strategy's built kernels by
                       key, and when each is built)
op2/rt/jit/report.h   (includes: op2/rt/jit/params.h,
                       op2/rt/loop/strategy.h; the JIT's summary of every
                       loop and its CSV rows; from
                       src/rt/jit/op_jit_report.cpp)
```

#### Choosing a strategy (op::rt::tune, sources under src/rt/tune)
```
op2/rt/tune/options.h   (the tuner's options as OP_AUTOTUNE* sets them,
                         read into the run's context; from
                         src/rt/tune/op_tune_options.cpp)
op2/rt/tune/autotune.h  (includes: op2/rt/loop/call.h, op2/rt/loop/ladder.h,
                         op2/rt/loop/strategy.h, op2/rt/tune/options.h; the
                         autotuner, which times every candidate per
                         configuration once its JIT modes have settled, and
                         its summary; from
                         src/rt/tune/op_tune_autotune.cpp, the summary from
                         src/rt/tune/op_tune_report.cpp)
```

#### Planning building blocks, backend neutral (op::rt::plan, sources under src/rt/plan)
Algorithms with no device knowledge, no device limits and no encodings,
which each backend composes into its own plans (op2/rt/gpu/chunk_plan.h,
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
                              op2/rt/loop/call.h, op2/rt/plan/chunks.h; the
                              staged kernels' scratch slots, a slot per
                              reference and element, the pipeline staging a
                              call's chunks under a backend's lanes, scratch,
                              slot limit and stride, and what every backend's
                              chunk plan holds: its chunks, batches and
                              staging, the slots taken as the backend's stage
                              words, and its figures)
op2/rt/plan/options.h        (the chunked strategies' run-wide options, read
                              into the run's context)
op2/rt/plan/cache.h          (includes: extern/rapidhash.h,
                              op2/rt/abi/descriptor.h, op2/rt/hasher.h,
                              op2/rt/loop/call.h;
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
op2/rt/gpu/options.h       (the run-wide options, the JIT mechanism's, the
                            wavefront window and launch synchronisation, and
                            the device's facts; read by op_init, which
                            declares the GPU families, from
                            src/rt/gpu/op_gpu_options.cpp)
op2/rt/gpu/layout.h        (includes: op_lib_core.h, op_rt_support.h; the
                            device data layout the kernels index by: the
                            warp pitch, a loop's per-element arrays' stride
                            and a SoA dat's)
op2/rt/gpu/upload.h        (includes: op2/gpu/check.h, op2/gpu/shims.h; a
                            plan's host arrays copied to the device and freed
                            there)
op2/rt/gpu/heuristics.h    (includes: op2/rt/abi/gpu/descriptor.h,
                            op2/rt/loop/strategy.h; the models of what pays
                            on the device, which turn what the backend knows
                            of a strategy's kernels into its estimates, the
                            register rule first; OP_JIT_MIN_REGISTERS, read
                            by op_init; from src/rt/gpu/op_gpu_heuristics.cpp)
op2/rt/gpu/executor.h      (includes: op_lib_core.h, op2/gpu/shims.h,
                            op2/rt/abi/gpu/descriptor.h,
                            op2/rt/abi/gpu/launch.h, op2/rt/gpu/jit.h,
                            op2/rt/gpu/kernel.h, op2/rt/jit/params.h,
                            op2/rt/loop/call.h, op2/rt/loop/ladder.h,
                            op2/rt/loop/loop.h; the executor a loop is made
                            with from a descriptor: its parameters' device
                            symbols, its compiler, a launch's arguments, a
                            record's fields then the loop's, its halo
                            exchanges and dirty bits on the device, and the
                            call's device globals; from
                            src/rt/gpu/op_gpu_executor.cpp)
op2/rt/gpu/kernel.h        (includes: op2/gpu/shims.h,
                            op2/rt/abi/gpu/descriptor.h, op2/rt/jit/params.h;
                            a strategy's kernels, one compiled with the
                            application and one built at run time, each
                            launched the same way, and the device symbols
                            the offline kernels read their parameters from;
                            from src/rt/gpu/op_gpu_kernel.cpp)
op2/rt/gpu/jit.h           (includes: op2/rt/abi/gpu/descriptor.h,
                            op2/rt/gpu/kernel.h, op2/rt/jit/params.h; the
                            JIT mechanism: NVRTC or hipRTC compiles of a
                            family's source after the embedded headers, its
                            run-time parameters as wrapper arguments or
                            module symbols; from src/rt/gpu/op_gpu_jit.cpp)
op2/rt/gpu/chunk_plan.h    (includes: op2/rt/abi/staging.h,
                            op2/rt/plan/chunks.h, op2/rt/plan/staging.h; the
                            chunk plans under the device's limits: a block's
                            elements, shared memory, slots as 16-bit stage
                            words, 8-bit colours; from
                            src/rt/gpu/op_gpu_chunk_plan.cpp)
op2/rt/gpu/chunk_encode.h  (includes: op2/rt/abi/gpu/launch.h,
                            op2/rt/gpu/chunk_plan.h; a chunk plan uploaded
                            as the kernels read it, which fills a chunk
                            launch's record and runs a wavefront call; from
                            src/rt/gpu/op_gpu_chunk_encode.cpp)
op2/rt/gpu/strategy.h      (includes: op_lib_core.h,
                            op2/rt/abi/gpu/descriptor.h,
                            op2/rt/abi/gpu/launch.h,
                            op2/rt/gpu/chunk_encode.h,
                            op2/rt/gpu/chunk_plan.h, op2/rt/gpu/executor.h,
                            op2/rt/gpu/jit.h, op2/rt/gpu/kernel.h,
                            op2/rt/gpu/layout.h, op2/rt/gpu/options.h,
                            op2/rt/jit/cache.h, op2/rt/loop/strategy.h,
                            op2/rt/plan/cache.h, op2/rt/plan/colouring.h;
                            the plain, atomics, colouring and chunked
                            strategies, their axes, block size, unstaged
                            singles and wavefront, their phases' launches,
                            and the table of the families the backend
                            offers; from src/rt/gpu/op_gpu_strategy.cpp)
```

#### OpenMP backend (op::rt::cpu, sources under src/rt/cpu)
Internal to the library: the generated code reaches it through
`op2/rt/abi/cpu/descriptor.h`.
```
op2/rt/cpu/executor.h    (includes: op_lib_core.h,
                          op2/rt/abi/cpu/descriptor.h,
                          op2/rt/abi/cpu/launch.h, op2/rt/jit/params.h,
                          op2/rt/loop/call.h, op2/rt/loop/ladder.h,
                          op2/rt/loop/loop.h; the executor a loop is made
                          with from a descriptor: the call's threads, their
                          per-thread globals and chunk scratch, and the
                          records its phases run with; from
                          src/rt/cpu/op_cpu_executor.cpp)
op2/rt/cpu/chunk_plan.h  (includes: op2/rt/abi/staging.h,
                          op2/rt/plan/chunks.h, op2/rt/plan/staging.h; the
                          chunk plans a host thread runs a chunk of on its
                          own, slots as 16-bit stage words; from
                          src/rt/cpu/op_cpu_chunk_plan.cpp)
op2/rt/cpu/strategy.h    (includes: op_lib_core.h, op2/rt/cpu/chunk_plan.h,
                          op2/rt/cpu/executor.h, op2/rt/loop/strategy.h,
                          op2/rt/plan/cache.h, op2/rt/plan/colouring.h; the
                          plain, atomics, colouring and chunked strategies
                          over host threads, each phase a record its
                          family's run is given, and the table of the
                          families the backend offers, which the CPU
                          libraries' op_init declares; from
                          src/rt/cpu/op_cpu_strategy.cpp, the declaration
                          from src/rt/cpu/op_cpu_init.cpp)
```
