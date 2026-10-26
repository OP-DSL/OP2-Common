op2_add_functional_tests(
    LANGUAGE cpp
    SOURCES  rw_tests.cpp
    STRATEGIES chunk_atomics atomics chunk_colouring colouring)
