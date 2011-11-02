#------------------------------------------------------------------------------
# OpenMP library
add_library(op2_openmp ${COMMON_SRC} ${RT_SRC} op_openmp_decl.c)

#------------------------------------------------------------------------------
# Install

install(TARGETS op2_openmp
  RUNTIME DESTINATION ${OP2_LIB_DIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${OP2_LIB_DIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${OP2_LIB_DIR} COMPONENT Development
)
