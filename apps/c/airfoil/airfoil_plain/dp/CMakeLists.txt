# airfoil (C++) - non-MPI and MPI variants, same as the legacy Makefile.

op2_add_app_variants(
    NAME     airfoil
    LANGUAGE cpp
    SOURCES  airfoil.cpp
    EXCLUDE_VARIANTS mpi_*)

op2_add_app_variants(
    NAME     airfoil_par
    LANGUAGE cpp
    SOURCES  airfoil_mpi.cpp
    VARIANTS mpi_*)
