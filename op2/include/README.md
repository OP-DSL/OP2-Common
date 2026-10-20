### Headers

#### Core components
```
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
                        along, the run-wide variables op_init reads, the
                        sections an indirect loop runs, and the teardown
                        registry)
loop/autotune.h        (includes: loop/strategy.h; the autotuner that times
                        every candidate strategy per configuration)
```

#### Planning building blocks, backend neutral (sources under src/plan)
Algorithms with no device knowledge and no device limits, which each
backend composes into its own plans (gpu/chunk_plan.h, cpu/chunk_plan.h).
```
plan/colouring.h       (includes: op_lib_core.h; the set colouring, as
                        op_plan's OP_COLOR2)
plan/chunks.h          (includes: loop/strategy.h; the references a planner
                        follows, chunks within sections, each chunk's
                        targets, coloured batches, the wavefront order and
                        element colours)
plan/staging.h         (includes: plan/chunks.h, plan/staging_layout.h; the
                        staged kernels' scratch slots and stage words)
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
                        define OP_F2C_JIT_TAG first)
```

#### CUDA and HIP backend of the translated kernels
```
gpu/shims.h            (gpu* aliases over the CUDA or HIP runtime)
gpu/config.h           (the run-wide configuration: JIT policy, device,
                        wavefront window; read by op_init, from src/gpu)
gpu/jit.h              (includes: gpu/config.h, gpu/shims.h; the JIT
                        service: parameters, compiles and every JIT policy)
gpu/chunk_plan.h       (includes: plan/chunks.h, plan/staging.h; the chunk
                        plans under the device's limits: block multiples,
                        shared memory, 16-bit stage words, 8-bit colours)
gpu/chunk_encode.h     (includes: gpu/chunk_plan.h, gpu/shims.h; a chunk
                        plan uploaded as the kernels read it)
gpu/strategy.h         (includes: gpu/jit.h, gpu/chunk_encode.h; the plain,
                        atomics, colouring and chunked strategies, and their
                        axes: block size, unstaged singles, wavefront)
gpu/loop.h             (includes: gpu/strategy.h; the loop the generated
                        code builds and invokes)
```

#### OpenMP backend of the translated kernels
```
cpu/chunk_plan.h       (includes: plan/chunks.h, plan/staging.h; the chunk
                        plans a host thread runs a chunk of on its own)
cpu/strategy.h         (includes: cpu/chunk_plan.h, plan/colouring.h; the
                        plain, atomics, colouring and chunked strategies over
                        host threads, and the per-thread globals)
cpu/loop.h             (includes: cpu/strategy.h, loop/autotune.h; the loop
                        the generated code builds and invokes)
```

#### Sequential layer
```
op_seq.h   (auto-generated, includes: op_lib_core.h)
```

#### OpenMP layer
```
op_openmp_rt_support.h (includes: op_rt_support.h)
```


#### CUDA layer
```
op_cuda_rt_support.h (includes: op_lib_core.h, op_rt_support.h)
op_cuda_reduction.h  (has to be kept separate becuase of FORTRAN layer)
```


#### MPI support
```
op_hdf5.h            (obsolete?)
op_lib_mpi.h         (include op_lib_core.h, op_rt_support.h, op_mpi_core.h, op_hdf5.h)
op_mpi_core.h
op_mpi_hdf5.h
op_mpi_seq.h         (includes: op_lib_core.h, op_rt_support.h)
op_util.h
```
