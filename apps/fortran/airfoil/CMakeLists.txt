# airfoil Fortran - three build flavours, matching the legacy Makefile.
# The three flavours share sources; DPLAIN / DARG_PTRS / DHDF5 select the
# behaviour at compile time.  The consts module is passed to the translator
# via --consts-module so it can resolve `use airfoil_constants`.

set(_airfoil_srcs
    airfoil_constants.F90
    airfoil_kernels.F90
    airfoil.F90)

set(_airfoil_translator_args
    --consts-module "${CMAKE_CURRENT_SOURCE_DIR}/airfoil_constants.F90")

op2_add_app_variants(
    NAME     airfoil_plain
    LANGUAGE fortran
    SOURCES  ${_airfoil_srcs}
    TRANSLATOR_ONLY_ARGS ${_airfoil_translator_args}
    COMPILE_DEFINITIONS  PLAIN
    EXCLUDE_VARIANTS
        mpi_seq mpi_genseq mpi_openmp mpi_cuda mpi_c_cuda mpi_c_hip)

op2_add_app_variants(
    NAME     airfoil_arg_ptrs
    LANGUAGE fortran
    SOURCES  ${_airfoil_srcs}
    TRANSLATOR_ONLY_ARGS ${_airfoil_translator_args}
    COMPILE_DEFINITIONS  ARG_PTRS
    EXCLUDE_VARIANTS
        mpi_seq mpi_genseq mpi_openmp mpi_cuda mpi_c_cuda mpi_c_hip)

op2_add_app_variants(
    NAME     airfoil_hdf5
    LANGUAGE fortran
    SOURCES  ${_airfoil_srcs}
    TRANSLATOR_ONLY_ARGS ${_airfoil_translator_args}
    COMPILE_DEFINITIONS  HDF5
    WITH_HDF5)
