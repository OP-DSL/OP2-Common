# airfoil (C++) - non-MPI and MPI variants, same as the legacy Makefile.

op2_add_app_variants(
    NAME     airfoil
    LANGUAGE cpp
    SOURCES  airfoil.cpp
    EXCLUDE_VARIANTS
        mpi_seq mpi_genseq mpi_openmp mpi_cuda mpi_hip mpi_c_cuda mpi_c_hip)

op2_add_app_variants(
    NAME     airfoil_par
    LANGUAGE cpp
    SOURCES  airfoil_mpi.cpp
    VARIANTS mpi_seq mpi_genseq mpi_openmp mpi_cuda mpi_hip mpi_c_cuda mpi_c_hip)
