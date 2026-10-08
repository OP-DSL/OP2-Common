# longint variant is MPI-only.

op2_add_app_variants(
    NAME     jac_longint_par
    LANGUAGE cpp
    SOURCES  jac_mpi.cpp
    VARIANTS mpi_*)
