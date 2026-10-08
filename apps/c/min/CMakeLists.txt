# min - two independent single-source apps, both non-MPI.

op2_add_app_variants(
    NAME     min_direct
    LANGUAGE cpp
    SOURCES  min_direct.cpp
    EXCLUDE_VARIANTS mpi_*)

op2_add_app_variants(
    NAME     min_indirect
    LANGUAGE cpp
    SOURCES  min_indirect.cpp
    EXCLUDE_VARIANTS mpi_*)
