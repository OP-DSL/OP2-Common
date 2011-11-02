#------------------------------------------------------------------------------
# HDF5 library
include_directories(${HDF5_INCLUDE_DIRS} ${MPI_INCLUDE_PATH})
add_definitions(${HDF5_DEFINITIONS})
add_library(op2_hdf5 ${COMMON_SRC} ${UTIL_SRC} op_hdf5.c)
target_link_libraries(op2_hdf5 ${HDF5_LIBRARIES})

#------------------------------------------------------------------------------
# Install

install(TARGETS op2_hdf5
  RUNTIME DESTINATION ${OP2_LIB_DIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${OP2_LIB_DIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${OP2_LIB_DIR} COMPONENT Development
)
