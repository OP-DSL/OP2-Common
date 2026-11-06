# Indirect increments in every form the C++ translator accepts, of mixed types
# and dimensions, with aliased and optional arguments, checked element by
# element against a host reference.  Serial only: the reference indexes the
# global numbering.  c_cuda and c_hip cannot translate every form.
op2_add_functional_tests(
    LANGUAGE cpp
    SOURCES    dat_increments.cpp
    FAMILIES   serial soa
    EXCLUDE_VARIANTS c_cuda c_hip
    STRATEGIES chunk_atomics atomics chunk_colouring colouring)
