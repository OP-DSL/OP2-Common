op2_add_app_variants(
    NAME     jac2
    LANGUAGE cpp
    SOURCES  jac.cpp
    EXCLUDE_VARIANTS mpi_*)

op2_add_app_variants(
    NAME     jac2_par
    LANGUAGE cpp
    SOURCES  jac_mpi.cpp
    VARIANTS mpi_*)
