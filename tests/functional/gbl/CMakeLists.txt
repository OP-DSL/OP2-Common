op2_add_functional_tests(
    LANGUAGE cpp
    SOURCES  gbl_tests.cpp
    FAMILIES serial mpi)

# Again with every global OP_INC incremented atomically in place, on the
# GPU targets that take the option.
op2_add_functional_tests(
    LANGUAGE cpp
    NAME     gbl_inc_atomic
    SOURCES  gbl_tests.cpp
    FAMILIES serial mpi
    TRANSLATOR_ARGS -c "{\".*\": {\"gbl_inc_atomic\": true}}"
    EXCLUDE_VARIANTS *seq *openmp *_legacy)
