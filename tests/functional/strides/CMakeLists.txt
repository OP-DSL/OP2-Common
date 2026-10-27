# Strides exist only under SoA, which only the GPU variants lay out.
op2_add_functional_tests(
    LANGUAGE cpp
    SOURCES  stride_tests.cpp
    FAMILIES soa)

# A helper shared by two sets' dats and a local array.  The legacy targets
# insert one stride into its text, which fits only one of them.
op2_add_functional_tests(
    LANGUAGE cpp
    NAME     strides_shared_helper
    SOURCES  shared_helper_tests.cpp
    FAMILIES soa
    EXCLUDE_VARIANTS *_legacy)
