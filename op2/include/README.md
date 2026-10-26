### Headers

#### Core components
```
op_env.h        (the environment variables' parser: every layer reads its
                 own through it as op_init runs, and op_init warns about
                 any OP_ variable nothing read; from src/core)
op_lib_c.h      (includes: op_lib_core.h)
op_lib_core.h
op_lib_cpp.h    (includes: op_lib_core.h, op_lib_c.h)
op_rt_support.h (includes: op_lib_core.h)
```

#### par_loop support, backend neutral (sources under src/loop)
```
loop/strategy.h        (includes: op_lib_core.h; a loop's strategies as
                        objects the ladder asks to prepare and run a
                        configuration, the axes a family's variants vary
                        along, the families a backend declares, the
                        run-wide variables op_init reads, the sections an
                        indirect loop runs, and the teardown registry)
loop/autotune.h        (includes: loop/strategy.h; the autotuner that times
                        every candidate strategy per configuration)
loop/jit.h             (includes: loop/strategy.h; the JIT's policy, read by
                        op_init, the parameters a loop's kernels are
                        specialised on and the statistics that move them
                        between baked in and run time, and a strategy's
                        cache of built kernels; a backend supplies the
                        mechanism)
loop/select.h          (includes: loop/strategy.h; the selection: the
                        kernel kind each strategy runs, decided from the
                        backends' estimates; the ladder and the tuner ask
                        here)
loop/loop.h            (includes: loop/autotune.h, loop/jit.h; the base of
                        every backend's loop: its ladder, tuner and JIT
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
                        staged kernels' scratch slots and stage words, and
                        the pipeline staging a call's chunks under a
                        backend's lanes and scratch budget)
plan/staging_layout.h  (the staging layout both the host and a staged kernel
                        read; no includes, so NVRTC compiles it too)
plan/options.h         (includes: loop/strategy.h; the chunked strategies'
                        run-wide options, read by op_init, and their chunk
                        size axis)
plan/greedy_colour.h   (internal: the greedy colouring the planners share)
```

#### Translated kernels (Fortran to C, and the C++ JIT targets)
```
kernel/prelude.h       (what a translated kernel compiles against: Ptr, Span,
                        the Fortran intrinsics; no includes)
kernel/strategy.h      (includes: kernel/prelude.h, plan/staging_layout.h; the
                        increment hooks and the chunked strategies' device
                        helpers)
kernel/jit_embed.h     (embeds the three kernel headers as text for NVRTC;
                        define OP_KERNEL_JIT_TAG first)
```

#### CUDA and HIP backend of the translated kernels
```
gpu/shims.h            (gpu* aliases over the CUDA or HIP runtime)
gpu/config.h           (the run-wide configuration: the JIT mechanism's
                        settings, device, wavefront window; read by op_init,
                        which declares the GPU families, from src/gpu)
gpu/jit.h              (includes: gpu/config.h, gpu/shims.h, loop/jit.h; the
                        JIT mechanism: offline kernels and their symbols,
                        NVRTC or hipRTC compiles, run-time parameters as
                        wrapper arguments or module symbols, launches)
gpu/chunk_plan.h       (includes: plan/chunks.h, plan/staging.h; the chunk
                        plans under the device's limits: block multiples,
                        shared memory, 16-bit stage words, 8-bit colours)
gpu/chunk_encode.h     (includes: gpu/chunk_plan.h, gpu/shims.h; a chunk
                        plan uploaded as the kernels read it)
gpu/strategy.h         (includes: gpu/jit.h, gpu/chunk_encode.h; the plain,
                        atomics, colouring and chunked strategies, and their
                        axes: block size, unstaged singles, wavefront)
gpu/loop.h             (includes: gpu/strategy.h, loop/loop.h,
                        op_cuda_rt_support.h; the loop the generated code
                        builds, registers its parameters on and invokes)
```

#### OpenMP backend of the translated kernels
```
cpu/chunk_plan.h       (includes: plan/chunks.h, plan/staging.h; the chunk
                        plans a host thread runs a chunk of on its own)
cpu/strategy.h         (includes: cpu/chunk_plan.h, plan/colouring.h; the
                        plain, atomics, colouring and chunked strategies over
                        host threads, the per-thread globals, and the
                        families the CPU libraries' op_init declares)
cpu/loop.h             (includes: cpu/strategy.h, loop/loop.h; the loop the
                        generated code builds and invokes)
```

#### Sequential layer
```
op_seq.h   (auto-generated, includes: op_lib_core.h)
```

#### CUDA layer
```
op_cuda_rt_support.h (includes: op_lib_core.h, op_rt_support.h)
op_cuda_reduction.h  (has to be kept separate becuase of FORTRAN layer)
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

#### HDF5 I/O
```
op_hdf5.h  (op_decl_*_hdf5, op_get_const_hdf5, op_dump_to_hdf5; from
            src/externlib/op_hdf5.cpp, src/mpi/op_mpi_hdf5.cpp under MPI)
```
