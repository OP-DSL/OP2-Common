op2_add_app_variants(
    NAME     jac
    LANGUAGE cpp
    SOURCES  jac.cpp
    EXCLUDE_VARIANTS
        mpi_seq mpi_genseq mpi_openmp mpi_cuda mpi_hip mpi_c_cuda mpi_c_hip)

op2_add_app_variants(
    NAME     jac_par
    LANGUAGE cpp
    SOURCES  jac_mpi.cpp
    VARIANTS mpi_seq mpi_genseq mpi_openmp mpi_cuda mpi_hip mpi_c_cuda mpi_c_hip)
