### Headers

#### Core components
```
op_lib_c.h      (includes: op_lib_core.h)
op_lib_core.h
op_lib_cpp.h    (includes: op_lib_core.h, op_lib_c.h)
op_rt_support.h (includes: op_lib_core.h)
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
