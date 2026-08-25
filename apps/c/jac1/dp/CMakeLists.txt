op2_add_app_variants(
    NAME     jac
    LANGUAGE cpp
    SOURCES  jac.cpp
    EXCLUDE_VARIANTS mpi_*)

op2_add_app_variants(
    NAME     jac_par
    LANGUAGE cpp
    SOURCES  jac_mpi.cpp
    VARIANTS mpi_*)
