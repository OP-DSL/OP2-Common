# The test source does `use mpi` under USE_MPI - see the MPI Fortran
# search in ../CMakeLists.txt.
op2_add_functional_tests(
    LANGUAGE fortran
    SOURCES  gbl_kernels.F90 gbl_tests.F90
    EXCLUDE_VARIANTS ${OP2_TEST_MPI_FORTRAN_EXCLUDE}
    LINK_LIBRARIES   ${OP2_TEST_MPI_FORTRAN_LINK})
