# Sequential library
add_library(op2_seq ${COMMON_SRC} ${RT_SRC} op_seq.c)

# Add all targets to the build-tree export set
export(TARGETS op2_seq
  FILE "${PROJECT_BINARY_DIR}/OP2LibraryDepends.cmake")

# Install
install(TARGETS op2_seq
  EXPORT OP2LibraryDepends
  LIBRARY DESTINATION ${INSTALLATION_LIB_DIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${INSTALLATION_LIB_DIR} COMPONENT Development
)
