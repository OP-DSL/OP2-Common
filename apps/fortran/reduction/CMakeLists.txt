# Fortran reduction - plain + HDF5 flavour.

op2_add_app_variants(
    NAME     reduction_f
    LANGUAGE fortran
    SOURCES  reduction.F90
    EXCLUDE_VARIANTS mpi_*)

# The HDF5 flavour uses the Fortran HDF5 declarations module, built only when
# HDF5 is found.
if(HDF5_FOUND)
    op2_add_app_variants(
        NAME     reduction_hdf5
        LANGUAGE fortran
        SOURCES  reduction.F90
        COMPILE_DEFINITIONS  HDF5
        WITH_HDF5)
endif()
