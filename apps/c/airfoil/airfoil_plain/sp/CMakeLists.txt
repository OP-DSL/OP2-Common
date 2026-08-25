# airfoil (C++, single-precision) - same shape as dp variant, name-prefixed
# to avoid target-name collisions with the dp counterpart.

op2_add_app_variants(
    NAME     airfoil_sp
    LANGUAGE cpp
    SOURCES  airfoil.cpp
    EXCLUDE_VARIANTS mpi_*)

op2_add_app_variants(
    NAME     airfoil_sp_par
    LANGUAGE cpp
    SOURCES  airfoil_mpi.cpp
    VARIANTS mpi_*)
