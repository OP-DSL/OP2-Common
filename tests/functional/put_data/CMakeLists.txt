# op_mpi_put_data is declared in op_lib_mpi.h and provided by the MPI library,
# so only the MPI variant families are built - matching this directory's
# Makefile.
op2_add_functional_tests(
    LANGUAGE cpp
    SOURCES  put_data_tests.cpp
    MPI_ONLY)
