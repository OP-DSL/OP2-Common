# The C++ translator corpus, against its kernels called directly.  The
# legacy targets reject its run-time map index and dims.
op2_add_functional_tests(
    LANGUAGE cpp
    SOURCES  corpus.cpp
    FAMILIES serial soa
    EXCLUDE_VARIANTS *_legacy)
