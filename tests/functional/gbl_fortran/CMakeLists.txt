# The test source does `use mpi` under USE_MPI - see the MPI Fortran
# search in ../CMakeLists.txt.
op2_add_functional_tests(
    LANGUAGE fortran
    SOURCES  gbl_kernels.F90 gbl_tests.F90
    FAMILIES serial mpi
    EXCLUDE_VARIANTS ${OP2_TEST_MPI_FORTRAN_EXCLUDE}
    LINK_LIBRARIES   ${OP2_TEST_MPI_FORTRAN_LINK})

# Again with every global OP_INC incremented atomically in place, on the
# GPU targets that take the option.
op2_add_functional_tests(
    LANGUAGE fortran
    NAME     gbl_inc_atomic_fortran
    SOURCES  gbl_kernels.F90 gbl_tests.F90
    FAMILIES serial mpi
    TRANSLATOR_ARGS -c "{\".*\": {\"gbl_inc_atomic\": true}}"
    EXCLUDE_VARIANTS *seq *openmp_legacy *seq_f2c *openmp_f2c ${OP2_TEST_MPI_FORTRAN_EXCLUDE}
    LINK_LIBRARIES   ${OP2_TEST_MPI_FORTRAN_LINK})
