# Fortran reduction - plain + HDF5 flavour.

op2_add_app_variants(
    NAME     reduction_f
    LANGUAGE fortran
    SOURCES  reduction.F90
    EXCLUDE_VARIANTS mpi_*)

op2_add_app_variants(
    NAME     reduction_hdf5
    LANGUAGE fortran
    SOURCES  reduction.F90
    COMPILE_DEFINITIONS  HDF5
    WITH_HDF5)
