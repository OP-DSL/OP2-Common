# Fortran jac1_long - MPI-only, and jac1_mpi.F90 does `use mpi` itself, so the
# MPI Fortran bindings are this app's dependency rather than OP2's.

find_package(MPI COMPONENTS Fortran QUIET)
if(NOT MPI_Fortran_FOUND)
    message(VERBOSE "OP2 app jac1_long_par: MPI Fortran bindings not found")
    return()
endif()

op2_add_app_variants(
    NAME     jac1_long_par
    LANGUAGE fortran
    SOURCES  jac1_mpi.F90
    VARIANTS mpi_seq mpi_genseq mpi_openmp mpi_cuda mpi_c_cuda mpi_c_hip
    TARGETS_VAR jac1_long_targets)

foreach(_t IN LISTS jac1_long_targets)
    target_link_libraries(${_t} PRIVATE MPI::MPI_Fortran)
endforeach()
