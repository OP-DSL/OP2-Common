op2_add_app_variants(
    NAME     reduction
    LANGUAGE cpp
    SOURCES  reduction.cpp
    EXCLUDE_VARIANTS mpi_*)

op2_add_app_variants(
    NAME     reduction_par
    LANGUAGE cpp
    SOURCES  reduction_mpi.cpp
    VARIANTS mpi_*)
