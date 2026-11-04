# airfoil (C++) - non-MPI and MPI variants, same as the legacy Makefile.  No
# legacy GPU variants: update's op_arg_info is rejected there.

op2_add_app_variants(
    NAME     airfoil
    LANGUAGE cpp
    SOURCES  airfoil.cpp
    EXCLUDE_VARIANTS mpi_* cuda_legacy hip_legacy)

op2_add_app_variants(
    NAME     airfoil_par
    LANGUAGE cpp
    SOURCES  airfoil_mpi.cpp
    VARIANTS mpi_*
    EXCLUDE_VARIANTS mpi_cuda_legacy mpi_hip_legacy)
