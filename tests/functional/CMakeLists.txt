include("${CMAKE_CURRENT_SOURCE_DIR}/OP2FunctionalTestSupport.cmake")

add_subdirectory(const)
add_subdirectory(dat_reductions)
add_subdirectory(gbl)
add_subdirectory(strides)
add_subdirectory(idx)
add_subdirectory(put_data)

if(CMAKE_Fortran_COMPILER)
    # Five Fortran test sources do `use mpi` - the tests' own dependency, found
    # here rather than by the OP2 build. Four guard it with #ifdef USE_MPI, so
    # without MPI Fortran those directories build their serial families only;
    # put_data_fortran does not, having no serial families to fall back on, so
    # the same exclusion drops that directory entirely.
    find_package(MPI COMPONENTS Fortran QUIET)
    if(MPI_Fortran_FOUND)
        set(OP2_TEST_MPI_FORTRAN_LINK    MPI::MPI_Fortran)
        set(OP2_TEST_MPI_FORTRAN_EXCLUDE "")
    else()
        set(OP2_TEST_MPI_FORTRAN_LINK    "")
        set(OP2_TEST_MPI_FORTRAN_EXCLUDE mpi_*)
    endif()

    add_subdirectory(const_fortran)
    add_subdirectory(dat_reductions_fortran)
    add_subdirectory(gbl_fortran)
    add_subdirectory(strides_fortran)
    add_subdirectory(idx_fortran)
    add_subdirectory(put_data_fortran)
endif()
