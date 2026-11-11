# The Fortran translator corpus, against its kernels called directly.
op2_add_functional_tests(
    LANGUAGE fortran
    SOURCES  corpus_kernels.F90 corpus_tests.F90
    FAMILIES serial soa)
