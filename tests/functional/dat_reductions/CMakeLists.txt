# The chunked strategies pass over two of these loops by design (an optional
# increment called inactive, and spoke_count's single hub, which leaves its
# plan too little to compress), so only the unchunked ones run forced;
# dat_increments and dat_read_write force the chunked ones.
op2_add_functional_tests(
    LANGUAGE cpp
    SOURCES    reduc_tests.cpp
    FAMILIES   serial mpi soa soa_mpi
    STRATEGIES atomics colouring)
