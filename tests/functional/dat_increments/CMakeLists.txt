# Indirect increments in every form the C++ translator accepts, of mixed types
# and dimensions, with aliased and optional arguments, checked element by
# element against a host reference.  Serial only: the reference indexes the
# global numbering.
op2_add_functional_tests(
    LANGUAGE cpp
    SOURCES    dat_increments.cpp
    FAMILIES   serial soa
    STRATEGIES chunk_atomics atomics chunk_colouring colouring)
