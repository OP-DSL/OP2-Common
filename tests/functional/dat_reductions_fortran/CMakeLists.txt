# The test source does `use mpi` under USE_MPI - see the MPI Fortran
# search in ../CMakeLists.txt.
op2_add_functional_tests(
    LANGUAGE fortran
    SOURCES  reduc_kernels.F90 reduc_tests.F90
    EXCLUDE_VARIANTS ${OP2_TEST_MPI_FORTRAN_EXCLUDE}
    LINK_LIBRARIES   ${OP2_TEST_MPI_FORTRAN_LINK})
