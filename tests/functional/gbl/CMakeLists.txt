op2_add_functional_tests(
    LANGUAGE cpp
    SOURCES  gbl_tests.cpp
    FAMILIES serial mpi)

# A two-component maximum with an info argument, serial only: info is not
# reduced across ranks, cuda and hip reject it, and c_cuda and c_hip return
# the wrong one.
op2_add_functional_tests(
    LANGUAGE cpp
    NAME     gbl_info
    SOURCES  info_tests.cpp
    EXCLUDE_VARIANTS cuda hip c_cuda c_hip)
