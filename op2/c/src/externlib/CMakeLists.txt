# HDF5 library
include_directories(${HDF5_INCLUDE_DIRS} ${MPI_INCLUDE_PATH})
add_definitions(${HDF5_DEFINITIONS})
add_library(op2_hdf5 ${COMMON_SRC} ${UTIL_SRC} op_hdf5.c)
target_link_libraries(op2_hdf5 ${HDF5_LIBRARIES})

# Add target to the build-tree export set
export(TARGETS op2_hdf5 APPEND
  FILE "${PROJECT_BINARY_DIR}/OP2LibraryDepends.cmake")

# Install
install(TARGETS op2_hdf5
  EXPORT OP2LibraryDepends
  LIBRARY DESTINATION ${INSTALLATION_LIB_DIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${INSTALLATION_LIB_DIR} COMPONENT Development
)
