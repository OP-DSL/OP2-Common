# Indirect increments of mixed types and dimensions, with aliased, optional
# and runtime-dimensioned arguments, checked element by element against a
# host reference.  Serial only: the reference indexes the global numbering.
op2_add_functional_tests(
    LANGUAGE fortran
    SOURCES    dat_increments.F90
    FAMILIES   serial soa
    STRATEGIES chunk_atomics atomics chunk_colouring colouring)
