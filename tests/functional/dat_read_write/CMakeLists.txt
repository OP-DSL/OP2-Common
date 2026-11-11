# One indirect RW loop over dim-1 dats: SoA lays them out the same.  c_cuda
# and c_hip miscount an indirect RW.
op2_add_functional_tests(
    LANGUAGE cpp
    SOURCES    rw_tests.cpp
    FAMILIES   serial mpi
    EXCLUDE_VARIANTS *c_cuda *c_hip)
