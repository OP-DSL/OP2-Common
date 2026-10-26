# One indirect RW loop over dim-1 dats: SoA lays them out the same, and
# atomics never runs a loop with an RW argument.
op2_add_functional_tests(
    LANGUAGE cpp
    SOURCES    rw_tests.cpp
    FAMILIES   serial mpi
    STRATEGIES chunk_colouring colouring)
