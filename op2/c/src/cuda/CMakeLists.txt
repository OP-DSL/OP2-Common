# CUDA library
include_directories(${CUDA_INCLUDE_DIRS})
add_library(op2_cuda ${COMMON_SRC} ${RT_SRC} op_cuda_decl.c op_cuda_rt_support.c)

# Add target to the build-tree export set
export(TARGETS op2_cuda APPEND
  FILE "${PROJECT_BINARY_DIR}/${OP2_TARGETS_EXPORT_SET}.cmake")

# Install
install(TARGETS op2_cuda
  EXPORT ${OP2_TARGETS_EXPORT_SET}
  LIBRARY DESTINATION ${INSTALLATION_LIB_DIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${INSTALLATION_LIB_DIR} COMPONENT Development
)
