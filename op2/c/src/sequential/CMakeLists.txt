#------------------------------------------------------------------------------
# Sequential library
add_library(op2_seq ${COMMON_SRC} ${RT_SRC} op_seq.c)

#------------------------------------------------------------------------------
# Install

install(TARGETS op2_seq
  RUNTIME DESTINATION ${OP2_LIB_DIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${OP2_LIB_DIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${OP2_LIB_DIR} COMPONENT Development
)
