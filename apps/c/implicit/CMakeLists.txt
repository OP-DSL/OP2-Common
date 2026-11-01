# The implicit proxy, with one and four components per node, when the solver
# layer is built; the legacy targets take no matrix blocks.
set(OP2_IMPLICIT_NV "1;4" CACHE STRING "Components per node of the implicit proxy's builds")

if(_op2_solve_srcs)
    foreach(_nv IN LISTS OP2_IMPLICIT_NV)
        op2_add_app_variants(
            NAME     implicit${_nv}
            LANGUAGE cpp
            SOURCES  implicit.cpp
            COMPILE_DEFINITIONS NV=${_nv}
            EXCLUDE_VARIANTS *_legacy mpi_*_legacy
            TARGETS_VAR _targets)
        # The mesh reader (-mesh) is no part of the translation.
        foreach(_t IN LISTS _targets)
            target_link_libraries(${_t} PRIVATE MPI::MPI_CXX)
            target_sources(${_t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/implicit_mesh.cpp)
            if(HDF5_FOUND)
                target_compile_definitions(${_t} PRIVATE IMPLICIT_HDF5)
                target_link_libraries(${_t} PRIVATE ${_op2_hdf5_target})
            endif()
        endforeach()
    endforeach()
endif()
