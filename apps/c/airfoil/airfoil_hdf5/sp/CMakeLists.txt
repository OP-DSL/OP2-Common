# airfoil (C++, HDF5, sp) - same structure as dp, prefixed target names.

op2_add_app_variants(
    NAME     airfoil_hdf5_sp
    LANGUAGE cpp
    SOURCES  airfoil.cpp
    WITH_HDF5)

# These tools call the raw HDF5 API themselves, so they name HDF5 directly:
# op2::* links it PRIVATE and so passes on no HDF5 include dir.
if(HDF5_FOUND)
    add_executable(airfoil_hdf5_sp_convert_mesh convert_mesh.cpp)
    target_link_libraries(airfoil_hdf5_sp_convert_mesh PRIVATE
        op2::op2_seq op2::op2_hdf5 ${_op2_hdf5_link_targets})
endif()

if(MPI_CXX_FOUND AND HDF5_FOUND AND OP2_HDF5_IS_PARALLEL)
    add_executable(airfoil_hdf5_sp_convert_mesh_mpi convert_mesh_mpi.cpp)
    target_link_libraries(airfoil_hdf5_sp_convert_mesh_mpi PRIVATE
        op2::op2_mpi MPI::MPI_CXX ${_op2_hdf5_link_targets})
endif()
