# min - two independent single-source apps, both non-MPI.

op2_add_app_variants(
    NAME     min_direct
    LANGUAGE cpp
    SOURCES  min_direct.cpp
    EXCLUDE_VARIANTS
        mpi_seq mpi_genseq mpi_openmp mpi_cuda mpi_hip mpi_c_cuda mpi_c_hip)

op2_add_app_variants(
    NAME     min_indirect
    LANGUAGE cpp
    SOURCES  min_indirect.cpp
    EXCLUDE_VARIANTS
        mpi_seq mpi_genseq mpi_openmp mpi_cuda mpi_hip mpi_c_cuda mpi_c_hip)
