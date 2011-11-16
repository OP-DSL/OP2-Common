# MPI library
add_definitions(${OP2_MPI_DEFINITIONS})
include_directories(${OP2_MPI_INCLUDE_DIRS})
add_library(op2_mpi ${COMMON_SRC} ${RT_SRC} ${UTIL_SRC} op_mpi_decl.c op_mpi_part_core.c op_mpi_core.c op_mpi_hdf5.c)
target_link_libraries(op2_mpi ${HDF5_LIBRARIES} ${PARMETIS_LIBRARIES} ${SCOTCH_LIBRARIES} ${MPI_LIBRARIES})

# Add all targets to the build-tree export set
export(TARGETS op2_mpi
  FILE "${PROJECT_BINARY_DIR}/OP2LibraryDepends.cmake")

# Install
install(TARGETS op2_mpi
  EXPORT OP2LibraryDepends
  LIBRARY DESTINATION ${INSTALLATION_LIB_DIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${INSTALLATION_LIB_DIR} COMPONENT Development
)
