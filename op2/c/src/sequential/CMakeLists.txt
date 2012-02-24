# Sequential library
add_library(op2_seq ${COMMON_SRC} ${RT_SRC} op_seq.c)

# Add target to the build-tree export set
export(TARGETS op2_seq APPEND
  FILE "${PROJECT_BINARY_DIR}/${OP2_TARGETS_EXPORT_SET}.cmake")

# Install
install(TARGETS op2_seq
  EXPORT ${OP2_TARGETS_EXPORT_SET}
  LIBRARY DESTINATION ${INSTALLATION_LIB_DIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${INSTALLATION_LIB_DIR} COMPONENT Development
)
