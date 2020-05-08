### OP2 Code Generators

This directory contains the OP2 code generators written in python targeting the C/C++ API. There are two folders, one for standard code generation (aot), and one for Just-In-Time Compilation enabled (jit).

The parallelisations and optimisations supported by each generator are as follows:

##### aot
* op2_gen_seq.py
* op2_gen_openmp.py - Initial OpenMP code generator
* op2_gen_openmp_simple.py - Simplified and Optimized OpenMP code generator
* op2_gen_cuda.py - Optimized for Fermi GPUs
* op2_gen_cuda_simple - Optimized for Kepler GPUs
* op2_gen_cuda_simple_hyb.py - generates openmp code as well as cuda code into the same file. Both CPUs and GPUs will then be used to do computations as a hybrid application.

##### jit
* op2_gen_seq_jit.py - Sequential JIT compiled application
* op2_gen_cuda_jit.py - JIT compiled application for NVidia CUDA - optimised for Kepler GPUs

### Invoking the Code Generator

Uncomment the parallelization you want to code generate in ops.py. For example for AOT compiled CUDA code generation do:

```
#op2_gen_seq(str(sys.argv[1]), date, consts, kernels)
#op2_gen_openmp(str(sys.argv[1]), date, consts, kernels) # Initial OpenMP code generator
op2_gen_cuda(str(sys.argv[1]), date, consts, kernels,sets) # Optimized for Fermi GPUs
```

Make ./op2.py executable

`chmod a+x ./op2.py`

Invoke the code generator by supplying the files that contain op_* API calls. Thus for example for Airfoil do the following.

```
./op2.py airfoil.cpp
```

To generate code that will JIT compiled when executed, also pass the parameter 'JIT'.

```
./op2.py airfoil.cpp JIT
```
This will overwrite the same folders used for aot code generation.
