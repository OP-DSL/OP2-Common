# The test source does `use mpi` under USE_MPI - see the MPI Fortran
# search in ../CMakeLists.txt.  The OpenMP variants are excluded: the Fortran
# openmp target misrenders the runtime-dimensioned dats of rw_dims and
# inc_dims.
op2_add_functional_tests(
    LANGUAGE fortran
    SOURCES  rw_kernels.F90 rw_tests.F90
    EXCLUDE_VARIANTS openmp mpi_openmp ${OP2_TEST_MPI_FORTRAN_EXCLUDE}
    LINK_LIBRARIES   ${OP2_TEST_MPI_FORTRAN_LINK}
    STRATEGIES       chunk_atomics atomics chunk_colouring colouring)
