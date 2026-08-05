op2_add_app_variants(
    NAME     reduction
    LANGUAGE cpp
    SOURCES  reduction.cpp
    EXCLUDE_VARIANTS
        mpi_seq mpi_genseq mpi_openmp mpi_cuda mpi_hip mpi_c_cuda mpi_c_hip)

op2_add_app_variants(
    NAME     reduction_par
    LANGUAGE cpp
    SOURCES  reduction_mpi.cpp
    VARIANTS mpi_seq mpi_genseq mpi_openmp mpi_cuda mpi_hip mpi_c_cuda mpi_c_hip)
