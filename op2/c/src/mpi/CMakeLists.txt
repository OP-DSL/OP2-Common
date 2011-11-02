#------------------------------------------------------------------------------
# MPI library
include_directories(${HDF5_INCLUDE_DIRS} ${PARMETIS_INCLUDE_DIRS} ${SCOTCH_INCLUDE_DIRS} ${MPI_INCLUDE_PATH})
add_library(op2_mpi ${COMMON_SRC} ${RT_SRC} ${UTIL_SRC} op_mpi_decl.c op_mpi_part_core.c op_mpi_core.c op_mpi_hdf5.c)
target_link_libraries(op2_mpi ${HDF5_LIBRARIES} ${PARMETIS_LIBRARIES} ${SCOTCH_LIBARIES})

#------------------------------------------------------------------------------
# Install

install(TARGETS op2_mpi
  RUNTIME DESTINATION ${OP2_LIB_DIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${OP2_LIB_DIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${OP2_LIB_DIR} COMPONENT Development
)
