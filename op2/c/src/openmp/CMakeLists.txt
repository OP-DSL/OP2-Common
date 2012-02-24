# OpenMP library
add_library(op2_openmp ${COMMON_SRC} ${RT_SRC} op_openmp_decl.c)
set(CMAKE_C_FLAGS "${OpenMP_C_FLAGS} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

# Add target to the build-tree export set
export(TARGETS op2_openmp APPEND
  FILE "${PROJECT_BINARY_DIR}/${OP2_TARGETS_EXPORT_SET}.cmake")

# Install
install(TARGETS op2_openmp
  EXPORT ${OP2_TARGETS_EXPORT_SET}
  LIBRARY DESTINATION ${INSTALLATION_LIB_DIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${INSTALLATION_LIB_DIR} COMPONENT Development
)
