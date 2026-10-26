# The test source does `use mpi` under USE_MPI - see the MPI Fortran
# search in ../CMakeLists.txt.
op2_add_functional_tests(
    LANGUAGE fortran
    SOURCES    rw_kernels.F90 rw_tests.F90
    FAMILIES   serial mpi soa soa_mpi
    EXCLUDE_VARIANTS ${OP2_TEST_MPI_FORTRAN_EXCLUDE}
    LINK_LIBRARIES   ${OP2_TEST_MPI_FORTRAN_LINK}
    STRATEGIES chunk_atomics atomics chunk_colouring colouring)
