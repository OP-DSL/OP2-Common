include("${CMAKE_CURRENT_SOURCE_DIR}/OP2FunctionalTestSupport.cmake")

add_subdirectory(const)
add_subdirectory(dat_reductions)
add_subdirectory(gbl)
add_subdirectory(strides)
add_subdirectory(idx)

if(CMAKE_Fortran_COMPILER)
    # Four Fortran test sources do `use mpi` under #ifdef USE_MPI - the tests'
    # own dependency, found here rather than by the OP2 build. Without it
    # those directories build their serial families only.
    find_package(MPI COMPONENTS Fortran QUIET)
    if(MPI_Fortran_FOUND)
        set(OP2_TEST_MPI_FORTRAN_LINK    MPI::MPI_Fortran)
        set(OP2_TEST_MPI_FORTRAN_EXCLUDE "")
    else()
        set(OP2_TEST_MPI_FORTRAN_LINK    "")
        set(OP2_TEST_MPI_FORTRAN_EXCLUDE
            mpi_seq mpi_genseq mpi_openmp mpi_cuda mpi_hip mpi_c_cuda mpi_c_hip)
    endif()

    add_subdirectory(const_fortran)
    add_subdirectory(dat_reductions_fortran)
    add_subdirectory(gbl_fortran)
    add_subdirectory(strides_fortran)
    add_subdirectory(idx_fortran)
endif()
