## Makefiles
This directory contains the common GNU Make infrastructure used by both the OP2 and example app builds. The entry point `common.mk` serves to set up compiler executables and library paths using the `compilers/` and `profiles/`, while `{c, f}_app.mk` specify a complete set of build rules for the apps written in their respective languages. For OP2, the build rules are instead located in `op2/Makefile`.

### `common.mk`: The entry point
This is included at the start of both the primary OP2 makefile in `op2/Makefile`, and before the `{c, f}_app.mk` in the app `Makefile`s. There is a selection of partially optional input variables used to specify the compilers, flags and libraries:
 * `OP2_{C, C_CUDA, F}_COMPILER`: These specify the basenames of the makefiles in their respective `compilers/{c, c_cuda, fortran}` directories that are automatically included to setup the compiler executables, flags and feature flags. If one of these is left undefined then the corresponding library and app functionality will not be built. At the very minimum an `OP2_C_COMPILER` is required.
 * `OP2_COMPILER`: Equivalent to `OP2_C_COMPILER=X OP2_F_COMPILER=X OP2_C_CUDA_COMPILER=nvhpc`.
 * `OP2_PROFILE`: A potential alternative (although not necessarily) to specifiying `OP2_COMPILER` or the `OP2_{C, C_CUDA, F}_COMPILER` variables, this specifies the basename of the profile under `profiles/` that is split and included as the prelude and epilogue. More information [below](#profiles-profiles).
 * (optional) `OP2_BUILD_DIR`: The directory under which OP2's `lib/` `mod/` and `obj/` folders and contents will be built. This defaults to `OP2-Common/op2` which should be acceptable for most cases.
 * (optional) `CUDA_INSTALL_PATH`: The location of the CUDA installation directory containing `include/` and `lib[64]/{libculibos.a, libcudart_static.a}`. Required if `OP2_C_CUDA_COMPILER` is specified and `CXX` does not inject the CUDA include and library paths (likely all but `nvhpc`).
 * `PTSCOTCH_INSTALL_PATH`: The location of a [PT-Scotch](https://www.labri.fr/perso/pelegrin/scotch/) installation directory. It is recommended to build PT-Scotch with the same compiler as OP2 to avoid link issues.
 * `PARMETIS_INSTALL_PATH`: The location of a [ParMETIS](http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview) installation directory. As with PT-Scotch, build with the same compiler as OP2.
 * (optional) `HDF5_{PAR, SEQ}_INSTALL_PATH`: The locations of [HDF5](https://www.hdfgroup.org/solutions/hdf5/) installations built with and without `--enable-parallel` respectively. Either or both may be specified, and HDF5 support will be available for the corresponding OP2 libraries and app variants.
 * (optional) `HDF5_INSTALL_PATH`: A convenience variable that may be used as an alternative to the `HDF5_{PAR, SEQ}_INSTALL_PATH` variables. This detects the variant of the HDF5 installation by inspecting `include/H5pubconf.h` and sets the appropriate `PAR` or `SEQ` installion path variable.

In addition, most variables defined as a result of the inclusion of `common.mk` and the respective compiler/profile makefiles can be overriden on input, for example to use `mpic++` instead of `mpicxx`: `make MPICXX=mpic++ ...`.

#### `compilers/`: Compiler setup
These are included as per the `OP2[_{C, C_CUDA, F}]_COMPILER` variables as specified above. The `compilers/c` directory contains definitions for the C and C++ compilers, the `compilers/c_cuda` directory contains definitions for CUDA compilers that can ingest `.cu` files and the `compilers/fortran` directory contains definitions for Fortran compilers that can ingest `.F90` and optionally `.CUF`. Each makefile in these directories is expected to set specific set of variables.

##### `compilers/c`: The C and C++ compilers
Basic compiler setup:
 * `CC`: The plain C compiler executable name (e.g. `gcc`).
 * `CXX`: The C++ compiler executable name (e.g. `g++`).
 * `CFLAGS`: The flags to be used for plain C compilation. These are expected to adhere to whether `DEBUG` is defined.
 * `CXXFLAGS`: The flags to be used for C++ compilation. As with `CFLAGS` these should adhere to `DEBUG`.
 * `CXXLINK`: Any additional libraries required for linking objects produced by `CXX` (e.g. `-lstdc++`).

Available OpenMP features:
 * `OMP_CPPFLAGS`: The flags to enable compilation of OpenMP pragmas (e.g. `-fopenmp`).
 * `CPP_HAS_OMP`: Set to `true` if the C and C++ compilers support OpenMP.
 * `OMP_OFFLOAD_CPPFLAGS`: The flags to enable the compilation of OpenMP 4.0 offload pragmas (e.g. `-foffload=nvptx-none`).
 * `CPP_HAS_OMP_OFFLOAD`: Set to `true` if the C and C++ compilers support OpenMP 4.0 offload.

##### `compilers/c_cuda`: The C/C++ CUDA compilers
Basic compiler setup:
 * `NVCC`: The CUDA compiler executable (e.g. `nvcc`).
 * `NVCCFLAGS`: The flags used for CUDA compilation. These are expected to adhere to `DEBUG` and also enable target-specific code generation if `NV_ARCH={Fermi, Kepler, ..., Volta}` is specified.

##### `compilers/fortran`: The Fortran compilers
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
 * `CUDA_FFLAGS`: The flags to enable compilation of CUDA Fortran (`.CUF`) sources. These should enable target-specific code generation if `NV_ARCH={Fermi, Kepler, ..., Volta}` is specified.
 * `F_HAS_CUDA`: Set to `true` if the Fortran compiler supports CUDA.

#### `profiles/`: Profiles
These are intended for cluster and/or custom compiler setup and are specified by setting `OP2_PROFILE` as detailed above. These are split into a prelude and epilogue section as specified by the `#! PRE` and `#! POST` comments. The prelude is included after the definition of `SHELL`, `MAKEFILES_DIR`, and `ROOT_DIR` but before any other configuration. Variables set here have the same effect as those set on the command-line. Hard-coded paths to resources that are not part of the common infrastructure on a particular cluster should not be set here.

The epilogue section is included at the end of `common.mk` and can override or append to any of the variables set during the compiler and library configuration. This may be useful as a means to extend compiler flags to target a particular hardware setup.

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
