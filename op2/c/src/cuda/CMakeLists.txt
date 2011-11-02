#------------------------------------------------------------------------------
# CUDA library
include_directories(${CUDA_INCLUDE_DIRS})
add_library(op2_cuda ${COMMON_SRC} ${RT_SRC} op_cuda_decl.c op_cuda_rt_support.c)

#------------------------------------------------------------------------------
# Install

install(TARGETS op2_cuda
  RUNTIME DESTINATION ${OP2_LIB_DIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${OP2_LIB_DIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${OP2_LIB_DIR} COMPONENT Development
)
