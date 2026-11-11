# The test source does `use mpi` under USE_MPI - see the MPI Fortran
# search in ../CMakeLists.txt.  Serial for its run-time dimensions, SoA for
# the strides.
op2_add_functional_tests(
    LANGUAGE fortran
    SOURCES  stride_kernels.F90 stride_tests.F90
    FAMILIES serial soa
    EXCLUDE_VARIANTS ${OP2_TEST_MPI_FORTRAN_EXCLUDE}
    LINK_LIBRARIES   ${OP2_TEST_MPI_FORTRAN_LINK})
