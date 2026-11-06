# The test source does `use mpi` under USE_MPI - see the MPI Fortran
# search in ../CMakeLists.txt.
op2_add_functional_tests(
    LANGUAGE fortran
    SOURCES  gbl_kernels.F90 gbl_tests.F90
    FAMILIES serial mpi
    EXCLUDE_VARIANTS ${OP2_TEST_MPI_FORTRAN_EXCLUDE}
    LINK_LIBRARIES   ${OP2_TEST_MPI_FORTRAN_LINK})

# Again through the regex program translator, as Hydra builds.
op2_add_functional_tests(
    LANGUAGE fortran
    NAME     gbl_regex_fortran
    SOURCES  gbl_kernels.F90 gbl_tests.F90
    FAMILIES serial soa
    TRANSLATOR_ARGS --regex-program-translator)
