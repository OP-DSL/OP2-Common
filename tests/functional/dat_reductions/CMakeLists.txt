# Only the unchunked strategies run forced here; dat_increments and
# dat_read_write force the chunked ones.
op2_add_functional_tests(
    LANGUAGE cpp
    SOURCES    reduc_tests.cpp
    FAMILIES   serial mpi soa soa_mpi
    STRATEGIES atomics colouring)
