# OpenMP library
add_library(op2_openmp ${COMMON_SRC} ${RT_SRC} op_openmp_decl.c)

# Add all targets to the build-tree export set
export(TARGETS op2_openmp
  FILE "${PROJECT_BINARY_DIR}/OP2LibraryDepends.cmake")

# Install
install(TARGETS op2_openmp
  EXPORT OP2LibraryDepends
  LIBRARY DESTINATION ${INSTALLATION_LIB_DIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${INSTALLATION_LIB_DIR} COMPONENT Development
)
