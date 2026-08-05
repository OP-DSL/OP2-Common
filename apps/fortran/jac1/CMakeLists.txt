# Fortran jac1 - non-MPI variants.

op2_add_app_variants(
    NAME     jac_f
    LANGUAGE fortran
    SOURCES  jac.F90
    EXCLUDE_VARIANTS
        mpi_seq mpi_genseq mpi_openmp mpi_cuda mpi_c_cuda mpi_c_hip)
