### Headers

#### Core components
```
op_lib_c.h      (includes: op_lib_core.h)
op_lib_core.h
op_lib_cpp.h    (includes: op_lib_core.h, op_lib_c.h)
op_rt_support.h (includes: op_lib_core.h)
```


#### Device memory
```
op_device.h     (op::Device: malloc, free, copy, memset; the host until a
                 backend registers its own, as cutilDeviceInit does)
```

#### par_loop support, backend neutral (sources under src/loop)
```
loop/strategy.h        (includes: op_lib_core.h; a loop's strategies as
                        objects the ladder asks to prepare and run a
                        configuration, the generic settings, the sections an
                        indirect loop runs, and the teardown registry)
loop/autotune.h        (includes: loop/strategy.h; the autotuner that times
                        every candidate strategy per configuration)
loop/set_colouring.h   (includes: op_lib_core.h; the set colouring planner,
                        as op_plan's OP_COLOR2)
loop/chunk_partition.h (includes: loop/chunk_layout.h, loop/strategy.h; the
                        chunk partition planner the chunked strategies build
                        on, and its options)
loop/chunk_layout.h    (the partition layout both the planner and a staged
                        kernel read; no includes, so NVRTC compiles it too)
loop/chunk_internal.h  (internal: the partition's resolved input and the
                        builders its two kinds share)
loop/greedy_colour.h   (internal: the greedy colouring both planners use)
```

#### Translated kernels (Fortran to C, and the C++ JIT targets)
```
kernel/prelude.h       (what a translated kernel compiles against: Ptr, Span,
                        the Fortran intrinsics; no includes)
kernel/strategy.h      (includes: kernel/prelude.h, loop/chunk_layout.h; the
                        increment hooks and the chunked strategies' device
                        helpers)
kernel/jit_embed.h     (embeds the three kernel headers as text for NVRTC;
                        define OP_F2C_JIT_TAG first)
```

#### CUDA and HIP backend of the translated kernels
```
gpu/shims.h            (gpu* aliases over the CUDA or HIP runtime)
gpu/config.h           (the run-wide configuration: JIT policy, block sizes,
                        device, chunk kernel variants; read by op_init, from
                        src/gpu)
gpu/jit.h              (includes: gpu/config.h, gpu/shims.h; the JIT
                        service: parameters, compiles and every JIT policy)
gpu/chunk_encode.h     (includes: loop/chunk_partition.h, op_device.h; a
                        chunk partition uploaded as the kernels read it)
gpu/strategy.h         (includes: gpu/jit.h, gpu/chunk_encode.h; the plain,
                        atomics, colouring and chunked strategies)
gpu/loop.h             (includes: gpu/strategy.h; the loop the generated
                        code builds and invokes)
```

#### OpenMP backend of the translated kernels
```
cpu/strategy.h         (includes: loop/chunk_partition.h, loop/set_colouring.h;
                        the plain, atomics, colouring and chunked strategies
                        over host threads, and the per-thread globals)
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
