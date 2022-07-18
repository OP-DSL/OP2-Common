## Makefiles
This directory contains the common GNU Make infrastructure used by both the OP2 and example app builds. The entry point `common.mk` serves to set up compiler executables and library paths using `compilers/`, `profiles/` and `dependencies/`, while `{c, f}_app.mk` specify a complete set of build rules for the apps written in their respective languages. For OP2, the build rules are instead located in `op2/Makefile`.

### `common.mk`: The entry point
This is included at the start of both the primary OP2 makefile in `op2/Makefile`, and before the `{c, f}_app.mk` in the app `Makefile`s. There is a selection of partially optional input variables used to specify the compilers, flags and libraries:
 * `OP2_{C, C_CUDA, F}_COMPILER`: These specify the basenames of the makefiles in their respective `compilers/{c, c_cuda, fortran}` directories that are automatically included to setup the compiler executables, flags and feature flags. If one of these is left undefined then the corresponding library and app functionality will not be built. At the very minimum an `OP2_C_COMPILER` is required.
 * `OP2_COMPILER`: Equivalent to `OP2_C_COMPILER=X OP2_F_COMPILER=X OP2_C_CUDA_COMPILER=nvhpc`.
 * `OP2_PROFILE`: A potential alternative (although not necessarily) to specifiying `OP2_COMPILER` or the `OP2_{C, C_CUDA, F}_COMPILER` variables, this specifies the basename of the profile under `profiles/` that is included to help setup compilation in a particular build environment. More information [below](#profiles-profiles).
 * (optional) `OP2_BUILD_DIR`: The directory under which OP2's `lib/` `mod/` and `obj/` folders and contents will be built. This defaults to `OP2-Common/op2` which should be acceptable for most cases.

In addition, most variables defined as a result of the inclusion of `common.mk` and the respective compiler/profile makefiles can be overriden on input, for example to use `mpic++` instead of `mpicxx`: `make MPICXX=mpic++ ...`.

### `profiles/`: Profiles
These are intended for cluster and/or custom compiler setup and are specified by setting `OP2_PROFILE` as detailed above. They are included at the start of `common.mk`, and thus can be used to override any Make variables that are initally assigned using `?=`. Variables set here have the same effect as those set on the command-line. Hard-coded paths to resources that are not part of the common infrastructure on a particular cluster should not be set here.

### `compilers.mk` and `compilers/`: Compiler setup
The `compilers.mk` makefile first includes the specific compiler makefiles in `compilers/` as dictated by the values of `OP2_COMPILER` and `OP2_{C, C_CUDA, F}_COMPILER`. Following this it then verifies that the compiler executables set by these makefiles are present, and then sets the `HAVE_{C, C_CUDA, F, MPI_C, MPI_F}` variables which are used to calculate the buildable set of OP2 libraries and app variants.

The `compilers/c` directory contains definitions for the C and C++ compilers, the `compilers/c_cuda` directory contains definitions for CUDA compilers that can ingest `.cu` files and the `compilers/fortran` directory contains definitions for Fortran compilers that can ingest `.F90` and optionally `.CUF`. Each makefile in these directories is expected to set specific set of variables.

#### `compilers/c`: The C and C++ compilers
Basic compiler setup:
 * `CC`: The plain C compiler executable name (e.g. `gcc`).
 * `CXX`: The C++ compiler executable name (e.g. `g++`).
 * `CFLAGS`: The flags to be used for plain C compilation. These are expected to adhere to whether `DEBUG` is defined.
 * `CXXFLAGS`: The flags to be used for C++ compilation. As with `CFLAGS` these should adhere to `DEBUG`.
 * `CXXLINK`: Any additional libraries required for linking objects produced by `CXX` (e.g. `-lstdc++`).

Available OpenMP features:
 * `OMP_CXXFLAGS`: The flags to enable compilation of OpenMP pragmas (e.g. `-fopenmp`).
 * `CPP_HAS_OMP`: Set to `true` if the C and C++ compilers support OpenMP.
 * `OMP_OFFLOAD_CXXFLAGS`: The flags to enable the compilation of OpenMP 4.0 offload pragmas (e.g. `-foffload=nvptx-none`).
 * `CPP_HAS_OMP_OFFLOAD`: Set to `true` if the C and C++ compilers support OpenMP 4.0 offload.

#### `compilers/c_cuda`: The C/C++ CUDA compilers
Basic compiler setup:
 * `NVCC`: The CUDA compiler executable (e.g. `nvcc`).
 * `NVCCFLAGS`: The flags used for CUDA compilation. These are expected to adhere to `DEBUG` and also enable target-specific code generation for each of the numerical architectures in the `CUDA_GEN` list.

#### `compilers/fortran`: The Fortran compilers
Basic compiler setup:
 * `FC`: The Fortran compiler executable. (e.g. `gfortran`)
 * `FFLAGS`: The flags to be used for Fortran compilation. These are expected to adhere to whether `DEBUG` is defined.
 * `F_MOD_OUT_OPT`: The flag to specify the `.mod` output directory (e.g. `-J`). This will be concatenated with the directory without a separating space.
 * `F_HAS_PARALLEL_BUILDS`: Set to `true` if running two instances of the compiler on the same source files at the same time is supported. This is likely to be the case unless the compiler emits object files in the same directory as the source files with no way to control this behaviour (`nvfortran`...).

Available OpenMP features:
 * `OMP_FFLAGS`: The flags to enable compilation of OpenMP pragmas (e.g. `-fopenmp`).
 * `F_HAS_OMP`: Set to `true` if the Fortran compiler supports OpenMP.
 * `OMP_OFFLOAD_FFLAGS`: The flags to enable the compilation of OpenMP 4.0 offload pragmas (e.g. `-foffload=nvptx-none`).
 * `F_HAS_OMP_OFFLOAD`: Set to `true` if the Fortran compiler supports OpenMP 4.0 offload.

Available CUDA features:
 * `CUDA_FFLAGS`: The flags to enable compilation of CUDA Fortran (`.CUF`) sources. These should enable target-specific code generation for each of the numerical architectures in the `CUDA_GEN` list.
 * `F_HAS_CUDA`: Set to `true` if the Fortran compiler supports CUDA.

### `dependencies/`: The dependencies
This directory contains makefiles corresponding to each library dependency of OP2. These all follow a specific structure, and test for the presence of the dependency by compiling the small test executables in `dependencies/tests`. The structure is as follows:
 1. The `X_INC_PATH` and `X_LIB_PATH` variables are set using the `X_INSTALL_PATH` input variable.
 2. The test executable is compiled with both the `X_INC_PATH` and `X_LIB_PATH` set but *without* the expected flags needed to link to the library (for example `-lparmetis -lmetis`). This is done to detect if the dependency is automatically linked by a compiler wrapper or similar.
 3. If the previous test fails the `X_LINK` variable is then set to the expected flags needed to link the library, and the test executable is recompiled.
 4. If either of the tests succeeded, then the `HAVE_X` variable is set to `true`, and the `X_INC` and `X_LIB` variables are set in accordance.

### `{c, f}_app.mk`: Generic app rules
These are generic makefiles that define a complete set of build rules for the common source structures found in the apps. The variants that will be built depend on the available compilers and compiler features. All of the possible variants currently are:
 * `seq`: Basic sequential variant that simply links with the OP2 libraries without code translation.
 * `genseq`: Translated sequential variant.
 * `vec`: A sequential variant with OpenMP SIMD vectorisation pragmas.
 * `openmp`: A multi-threaded variant using OpenMP pragmas.
 * `cuda`: A multi-threaded variant using NVIDIA CUDA.
 * `mpi_<variant>`: Many-core MPI variants of all other variants.

A few input variables must be defined before the inclusion of the makefiles. For the `c_app.mk` makefile these are:
 * `APP_NAME`: The name that will be used in the output executables and objects.
 * `APP_ENTRY`: The main `.cpp` source file which will be run through the translators.
 * `APP_ENTRY_MPI`: The main `.cpp` source file with MPI support that will be run through the translators.
 * (optional) `OP2_LIBS_WITH_HDF5`: Provides the HDF5 libraries along with the OP2 libraries in the `OP2_LIB_*` helpers. Set to `true` when building an app that uses HDF5.

For the `f_app.mk` makefile the following are available:
 * `APP_NAME`: The name that will be used in the output executables and objects as well as the name (without `.F90`) extension of the main app source file.
 * (optional) `OP2_LIBS_WITH_HDF5`: Provides the HDF5 libraries along with the OP2 libraries in the `OP2_LIB_*` helpers. Set to `true` when building an app that uses HDF5.

The inclusion of `common.mk` is required prior to the inclusion of the appropriate app makefile. Additional build rules may be added using GNU Make's prerequisite merging functionality. See [airfoil_hdf5/dp/Makefile](../apps/c/airfoil/airfoil_hdf5/dp/Makefile) for an example.
