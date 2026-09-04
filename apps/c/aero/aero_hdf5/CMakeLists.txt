# aero (C++) - HDF5-based, name-prefixed to avoid collision with aero_plain.
# Single source, all buildable variants get HDF5 via WITH_HDF5.  Plus the
# write_hdf5 companion tool (MPI + parallel HDF5).

op2_add_app_variants(
    NAME     aero_hdf5
    LANGUAGE cpp
    SOURCES  aero.cpp
    WITH_HDF5)

# These tools call the raw HDF5 API themselves, so they name HDF5 directly:
# op2::* links it PRIVATE and so passes on no HDF5 include dir.
if(MPI_CXX_FOUND AND HDF5_FOUND AND OP2_HDF5_IS_PARALLEL)
    add_executable(aero_hdf5_write_hdf5 write_hdf5.cpp)
    target_link_libraries(aero_hdf5_write_hdf5 PRIVATE
        op2::op2_mpi MPI::MPI_CXX ${_op2_hdf5_link_targets})
endif()
