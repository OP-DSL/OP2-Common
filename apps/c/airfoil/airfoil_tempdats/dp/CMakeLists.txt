# airfoil (C++, tempdats variant) - same shape as airfoil_plain, prefixed
# target names.

op2_add_app_variants(
    NAME     airfoil_tempdats
    LANGUAGE cpp
    SOURCES  airfoil.cpp
    EXCLUDE_VARIANTS mpi_*)

op2_add_app_variants(
    NAME     airfoil_tempdats_par
    LANGUAGE cpp
    SOURCES  airfoil_mpi.cpp
    VARIANTS mpi_*)
