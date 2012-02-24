set(MPI_SRC op_mpi_part_core.c op_mpi_core.c)
if(OP2_WITH_HDF5)
  set(MPI_SRC ${MPI_SRC} op_mpi_hdf5.c)
endif()

# MPI library
add_definitions(${OP2_MPI_DEFINITIONS})
include_directories(${OP2_MPI_INCLUDE_DIRS})
add_library(op2_mpi ${COMMON_SRC} ${RT_SRC} ${UTIL_SRC} ${MPI_SRC} op_mpi_decl.c)
target_link_libraries(op2_mpi ${OP2_MPI_LIBRARIES})

# Add target to the build-tree export set
export(TARGETS op2_mpi APPEND
  FILE "${PROJECT_BINARY_DIR}/${OP2_TARGETS_EXPORT_SET}.cmake")

# Install
install(TARGETS op2_mpi
  EXPORT ${OP2_TARGETS_EXPORT_SET}
  LIBRARY DESTINATION ${INSTALLATION_LIB_DIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${INSTALLATION_LIB_DIR} COMPONENT Development
)

# MPI-CUDA library
if(OP2_WITH_CUDA AND CUDA_FOUND)
  include_directories(${CUDA_INCLUDE_DIRS})

  cuda_add_library(op2_mpi_cuda ${COMMON_SRC} ${RT_SRC} ${UTIL_SRC} ${MPI_SRC}
    op_mpi_cuda_decl.c op_mpi_cuda_kernels.cu op_mpi_cuda_rt_support.c)
  target_link_libraries(op2_mpi_cuda ${OP2_MPI_LIBRARIES})

  # Add target to the build-tree export set
  export(TARGETS op2_mpi_cuda APPEND
    FILE "${PROJECT_BINARY_DIR}/${OP2_TARGETS_EXPORT_SET}.cmake")

  # Install
  install(TARGETS op2_mpi_cuda
    EXPORT ${OP2_TARGETS_EXPORT_SET}
    LIBRARY DESTINATION ${INSTALLATION_LIB_DIR} COMPONENT RuntimeLibraries
    ARCHIVE DESTINATION ${INSTALLATION_LIB_DIR} COMPONENT Development
    )
endif()
