### Headers

Every OP2 header lives here, the library's private ones too: a source under
`src/` includes them by their path from this directory
(`#include <loop/strategy.h>`), and no header lives under `src/`.  A header is
header-only unless its entry names where its definitions are.  `includes:`
lists a header's direct includes from this directory.

#### Core components
```
op_env.h        (the environment variables' parser: every layer reads its
                 own through it as op_init runs, and op_init warns about
                 any OP_ variable nothing read; from src/core)
op_lib_c.h      (includes: op_lib_core.h, op_rt_support.h, op_hdf5.h)
op_lib_core.h   (includes: SafeLong.h)
op_lib_cpp.h    (includes: op_lib_c.h, op_lib_core.h)
op_rt_support.h (includes: op_lib_core.h)
op_profile.h    (includes: extern/json.hpp; the profiler's timing tree and
                 its C API, op_profile_*; from src/externlib/op_profile.cpp)
SafeLong.h      (a long long whose lossy conversions and division by zero
                 throw, with a stack trace; from src/core/SafeLong.cpp)
```

#### par_loop support, backend neutral (sources under src/loop)
```
loop/strategy.h        (includes: op_lib_core.h; a loop's strategies as
                        objects the ladder asks to prepare and run a call,
                        each choosing its kernel kind from its backend's
                        estimates, the axes a family's variants vary along,
                        the families a backend declares, the run-wide
                        variables op_init reads, the sections an indirect
                        loop runs, and the teardown registry)
loop/autotune.h        (includes: loop/strategy.h; the autotuner that times
                        every candidate strategy per configuration)
loop/jit.h             (includes: extern/rapidhash.h, loop/strategy.h,
                        op_lib_core.h; the JIT's options, read by op_init,
                        the parameters a loop's kernels are specialised on
                        and the statistics that move them between baked in
                        and run time, and a strategy's cache of built
                        kernels; a backend supplies the mechanism)
loop/loop.h            (includes: loop/autotune.h, loop/jit.h,
                        loop/strategy.h, op_lib_core.h; the base of every
                        backend's loop: its ladder, tuner and JIT
                        parameters, and the steps of a call every backend
                        shares)
```

#### Planning building blocks, backend neutral (sources under src/plan)
Algorithms with no device knowledge and no device limits, which each
backend composes into its own plans (gpu/chunk_plan.h, cpu/chunk_plan.h).
```
plan/colouring.h       (includes: loop/strategy.h, op_lib_core.h; the set
                        colouring of a call's sections, as op_plan's
                        OP_COLOR2)
plan/chunks.h          (includes: loop/strategy.h; the references a planner
                        follows, chunks within sections, each chunk's
                        targets, coloured batches, the wavefront order,
                        element colours, and the result a backend's plan
                        comes back in)
plan/staging.h         (includes: plan/chunks.h, plan/staging_layout.h; the
                        staged kernels' argument groups, scratch slots and
                        stage words, and the pipeline staging a call's
                        chunks under a backend's lanes and scratch limit)
plan/staging_layout.h  (the staging layout both the host and a staged kernel
                        read; no includes, so NVRTC compiles it too)
plan/options.h         (includes: loop/strategy.h; the chunked strategies'
                        run-wide options, read by op_init, and their chunk
                        size axis)
plan/greedy_colour.h   (includes: op_lib_core.h; internal to the planners:
                        the greedy colouring they share)
```

#### Translated kernels (C++ cuda, hip and openmp; Fortran's _f2c targets)
```
kernel/prelude.h       (what a translated kernel compiles against: Ptr, Span,
                        the Fortran intrinsics; no includes)
kernel/strategy.h      (includes: kernel/prelude.h, plan/staging_layout.h; the
                        increment hooks and the chunked strategies' device
                        helpers)
kernel/jit_embed.h     (includes: extern/incbin.h; embeds the three kernel
                        headers as text for NVRTC; define OP_KERNEL_JIT_TAG
                        first)
```

#### CUDA and HIP backend of the translated kernels
```
gpu/shims.h            (gpu* aliases over the CUDA or HIP runtime)
gpu/check.h            (includes: gpu/shims.h; GPU_CHECK, which stops the
                        program on a failed runtime, driver or compiler call)
gpu/options.h          (the run-wide options, the JIT mechanism's and the
                        wavefront window, and the device's facts; read by
                        op_init, which declares the GPU families, from
                        src/gpu)
gpu/jit.h              (includes: gpu/check.h, gpu/options.h, gpu/shims.h,
                        kernel/prelude.h, kernel/strategy.h, loop/jit.h,
                        op_lib_core.h; the JIT mechanism: offline kernels and
                        their symbols, NVRTC or hipRTC compiles, run-time
                        parameters as wrapper arguments or module symbols,
                        launches)
gpu/chunk_plan.h       (includes: plan/chunks.h, plan/staging.h; the chunk
                        plans under the device's limits: block multiples,
                        shared memory, 16-bit stage words, 8-bit colours)
gpu/chunk_encode.h     (includes: gpu/check.h, gpu/chunk_plan.h, gpu/shims.h;
                        a chunk plan uploaded as the kernels read it)
gpu/strategy.h         (includes: gpu/chunk_encode.h, gpu/chunk_plan.h,
                        gpu/jit.h, loop/strategy.h, op_lib_cpp.h,
                        plan/colouring.h, plan/options.h; the plain, atomics,
                        colouring and chunked strategies, and their axes:
                        block size, unstaged singles, wavefront)
gpu/loop.h             (includes: gpu/check.h, gpu/jit.h, gpu/strategy.h,
                        loop/loop.h, loop/strategy.h, op_cuda_rt_support.h,
                        op_lib_cpp.h, op_profile.h, plan/options.h; the loop
                        the generated code builds, registers its parameters
                        on and invokes)
```

#### OpenMP backend of the translated kernels
```
cpu/chunk_plan.h       (includes: plan/chunks.h, plan/staging.h; the chunk
                        plans a host thread runs a chunk of on its own)
cpu/strategy.h         (includes: cpu/chunk_plan.h, loop/strategy.h,
                        op_lib_core.h, plan/colouring.h, plan/options.h; the
                        plain, atomics, colouring and chunked strategies over
                        host threads, the per-thread globals, and the
                        families the CPU libraries' op_init declares)
cpu/loop.h             (includes: cpu/strategy.h, loop/loop.h,
                        loop/strategy.h, op_lib_c.h, op_profile.h,
                        plan/options.h; the loop the generated code builds
                        and invokes)
```

#### Sequential layer
```
op_seq.h   (auto-generated, includes: op_lib_cpp.h)
```

#### CUDA layer
```
op_cuda_rt_support.h (includes: gpu/shims.h, op_lib_core.h, op_rt_support.h)
op_cuda_reduction.h  (includes: gpu/shims.h; kept apart for the Fortran
                      layer)
```

#### MPI support
```
op_lib_mpi.h               (includes: op_lib_core.h, op_mpi_core.h; the MPI
                            library's entry points)
op_mpi_core.h              (the halo and partition records the MPI backends
                            share, the communicators and reductions)
op_mpi_halo.h              (includes: op_lib_core.h, op_mpi_core.h; the halo
                            lists op_halo_create builds)
op_mpi_comm.h              (the sparse data exchange (NBX) halo construction
                            uses to find what each rank receives)
op_mpi_unified_exchanges.h (includes: op_lib_core.h, op_lib_c.h, op_lib_mpi.h;
                            the halo exchange's gathers and scatters, compiled
                            for the host and the device)
op_mpi_unified_backend.h   (includes: op_mpi_unified_exchanges.h; the buffers
                            and exchange specifications a backend gives the
                            exchange driver)
op_util.h                  (allocation, search and file helpers)
```

#### Renumbering (used by src/externlib/op_renumber.cpp)
Graph and geometric orderings with no OP2 dependencies, each producing a
permutation.
```
renumber/rcm.h         (reverse Cuthill-McKee, and the BFS helpers Sloan
                        shares)
renumber/sloan.h       (includes: renumber/rcm.h; Sloan's profile-minimising
                        ordering)
renumber/hilbert_sfc.h (the Hilbert space-filling curve over 2D or 3D
                        coordinates)
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
