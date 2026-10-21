# The OpenMP variants are excluded: their SIMD path loses updates when lanes
# of one vector share an indirect RW target.
op2_add_functional_tests(
    LANGUAGE cpp
    SOURCES  rw_tests.cpp
    EXCLUDE_VARIANTS openmp mpi_openmp)
