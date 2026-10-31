# The implicit proxy, with one and four components per node, when the solver
# layer is built; the legacy targets take no matrix blocks.
if(_op2_solve_srcs)
    foreach(_nv 1 4)
        op2_add_app_variants(
            NAME     implicit${_nv}
            LANGUAGE cpp
            SOURCES  implicit.cpp
            COMPILE_DEFINITIONS NV=${_nv}
            EXCLUDE_VARIANTS *_legacy mpi_*_legacy
            TARGETS_VAR _targets)
        foreach(_t IN LISTS _targets)
            target_link_libraries(${_t} PRIVATE MPI::MPI_CXX)
        endforeach()
    endforeach()
endif()
