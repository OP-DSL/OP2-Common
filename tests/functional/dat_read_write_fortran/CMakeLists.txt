# The test source does `use mpi` under USE_MPI - see the MPI Fortran
# search in ../CMakeLists.txt.  The OpenMP variants are excluded: their SIMD
# path loses updates when lanes of one vector share an indirect RW target.
op2_add_functional_tests(
    LANGUAGE fortran
    SOURCES  rw_kernels.F90 rw_tests.F90
    EXCLUDE_VARIANTS openmp mpi_openmp ${OP2_TEST_MPI_FORTRAN_EXCLUDE}
    LINK_LIBRARIES   ${OP2_TEST_MPI_FORTRAN_LINK}
    STRATEGIES       hier_atomics atomics hier_colouring colouring)
