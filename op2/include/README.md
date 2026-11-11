### Headers

Every OP2 header lives here, the library's private ones too, and no header
lives under `src/`.  Two kinds sit side by side:

- the flat headers (`op_lib_core.h`, `op_seq.h`, ...), the long-standing C
  API, included by name (`#include <op_seq.h>`);
- the `op2/` tree, included by its path (`#include <op2/env.h>`), whose
  path below `op2/` is its namespace below `op::`; a namespace with one
  header may be that file (`op2/env.h` is `op::env`).

This directory installs as `include/op2`, the package's one include
directory, so both spellings work the same in the source tree and once
installed (the `op2/` tree as `include/op2/op2/`).

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

#### OpenMP layer
```
op_openmp_rt_support.h (includes: op_rt_support.h; op_plan_get for the
                        OpenMP targets)
```

#### CUDA layer
```
op_gpu_shims.h       (gpu* aliases over the CUDA or HIP runtime)
op_cuda_rt_support.h (includes: op_gpu_shims.h, op_lib_core.h,
                      op_rt_support.h)
op_cuda_reduction.h  (includes: op_gpu_shims.h; kept apart for the Fortran
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

#### Fortran-to-C kernels (c_seq, c_cuda, c_hip)
```
op_f2c_prelude.h  (op::f2c: the arrays and intrinsics the translated kernels
                   compile against)
op_f2c_helpers.h  (includes: extern/rapidhash.h, op_profile.h,
                   op_gpu_shims.h; the c_cuda and c_hip JIT)
```

#### Third party
```
extern/json.hpp      (nlohmann JSON, for op_profile's JSON output)
extern/incbin.h      (binary inclusion, for the JIT's embedded headers)
extern/rapidhash.h   (rapidhash, for JIT parameter values)
```

### The op2/ tree

#### Core
```
op2/env.h        (op::env: the environment variables' parser: every layer
                  reads its own through it as op_init runs, and op_init warns
                  about any OP_ variable nothing read; from
                  src/core/op_env.cpp)
```
