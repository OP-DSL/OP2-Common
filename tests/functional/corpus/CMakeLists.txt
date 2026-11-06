# The C++ translator corpus, against its kernels called directly.  Only seq
# and openmp translate its run-time map index and dims.
op2_add_functional_tests(
    LANGUAGE cpp
    SOURCES  corpus.cpp
    FAMILIES serial
    EXCLUDE_VARIANTS openmp_legacy cuda hip c_cuda c_hip)
