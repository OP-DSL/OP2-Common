# op_mpi_put_data is only meaningful under MPI, so only the MPI variant
# families are built - matching this directory's Makefile. Unlike the other
# Fortran tests the source does `use mpi` unconditionally, not under
# USE_MPI, so without MPI Fortran the mpi_* exclusion leaves this directory
# with nothing to build, which is the wanted outcome. See the MPI Fortran
# search in ../CMakeLists.txt.
op2_add_functional_tests(
    LANGUAGE fortran
    SOURCES  put_data_kernels.F90 put_data_tests.F90
    MPI_ONLY
    EXCLUDE_VARIANTS ${OP2_TEST_MPI_FORTRAN_EXCLUDE}
    LINK_LIBRARIES   ${OP2_TEST_MPI_FORTRAN_LINK})
