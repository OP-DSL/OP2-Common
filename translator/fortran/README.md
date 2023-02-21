### Fortran Code Generators
This directory contains the OP2 code generators written in Python targeting the Fortran API. The parallelisations and optimisations supported by each generator are as follows:

##### MPI+SEQ
 * `op2_gen_mpiseq.py`: Generate host stubs for MPI+SEQ.
 * `op2_gen_mpiseq3.py`: Generate host stubs for MPI+SEQ -- optimised by removing the overhead due to Fortran C to F pointer setups.
 * `op2_gen_mpivec.py`: Generate host stubs for MPI+SEQ with intel vectorization optimisations.

##### OpenMP
 * `op2_gen_openmp3.py`: Optimised by removing the overhead due to Fortran C to F pointer setups.
 * `op2_gen_openmp2.py`: Version without staging.
 * `op2_gen_openmp.py`: Original version - one that most OP2 papers refer to.

##### CUDA
 * `op2_gen_cuda.py`
 * `op2_gen_cuda_permute.py`: Permute does a different coloring (permute execution within blocks by color).
 * `op2_gen_cudaINC.py`: Stages increment data only in shared memory.
 * `op2_gen_cuda_old.py`: Code generator targettign Fermi GPUs.

##### If hydra:
 * `op2_gen_cuda_hydra()`: Includes several Hydra specific features.

#### Invoking the Code Generator
Uncomment the parallelization you want to code generate in `op2_fortran.py`. For example for CUDA code generation do:
```
#op2_gen_openmp(str(sys.argv[init_ctr]), date, consts, kernels, hydra)
op2_gen_cuda(str(sys.argv[1]), date, consts, kernels, hydra)
```

Make `./op2_fortran.py` executable
```
chmod a+x ./op2_fortran.py
```

Invoke the code generator by supplying the files that contain op_* API calls. Thus for example for Airfoil do the following.
```
./op2_fortran.py airfoil.F90
```
