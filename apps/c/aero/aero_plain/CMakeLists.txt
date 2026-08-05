# aero (C++) - plain (non-HDF5).  Two calls mirror the legacy Makefile:
# one for non-MPI variants (aero.cpp), one for MPI variants (aero_mpi.cpp).

op2_add_app_variants(
    NAME     aero
    LANGUAGE cpp
    SOURCES  aero.cpp
    EXCLUDE_VARIANTS
        mpi_seq mpi_genseq mpi_openmp mpi_cuda mpi_hip mpi_c_cuda mpi_c_hip)

op2_add_app_variants(
    NAME     aero_par
    LANGUAGE cpp
    SOURCES  aero_mpi.cpp
    VARIANTS mpi_seq mpi_genseq mpi_openmp mpi_cuda mpi_hip mpi_c_cuda mpi_c_hip)
