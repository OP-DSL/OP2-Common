add_subdirectory(aero_plain)
if(HDF5_FOUND)
    add_subdirectory(aero_hdf5)
endif()
