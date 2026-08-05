# Tutorial series. Steps 1-6 are single-source binaries linked against
# progressively more OP2 backends; step7/final use the full translator
# pipeline. "original" is a standalone demo with no OP2 dependency, kept
# out of the CMake build.
foreach(_step step1 step2 step3 step4 step5)
    add_executable(airfoil_${_step} ${_step}/airfoil_${_step}.cpp)
    target_link_libraries(airfoil_${_step} PRIVATE op2::op2_seq)
endforeach()

if(MPI_CXX_FOUND AND HDF5_FOUND AND OP2_HDF5_IS_PARALLEL)
    add_executable(airfoil_step6 step6/airfoil_step6.cpp)
    target_link_libraries(airfoil_step6 PRIVATE
        op2::op2_mpi op2::op2_hdf5 MPI::MPI_CXX)
endif()

add_subdirectory(step7)
add_subdirectory(final)
