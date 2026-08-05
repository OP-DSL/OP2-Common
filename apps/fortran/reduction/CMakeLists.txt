# Fortran reduction - plain + HDF5 flavour.

op2_add_app_variants(
    NAME     reduction_f
    LANGUAGE fortran
    SOURCES  reduction.F90
    EXCLUDE_VARIANTS
        mpi_seq mpi_genseq mpi_openmp mpi_cuda mpi_c_cuda mpi_c_hip)

op2_add_app_variants(
    NAME     reduction_hdf5
    LANGUAGE fortran
    SOURCES  reduction.F90
    COMPILE_DEFINITIONS  HDF5
    WITH_HDF5)
