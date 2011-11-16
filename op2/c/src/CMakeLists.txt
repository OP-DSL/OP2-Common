#------------------------------------------------------------------------------
# Common source files

# Core library sources
set(COMMON_SRC "${OP2_SOURCE_DIR}/src/core/op_lib_core.c")
# Utility sources
set(UTIL_SRC "${OP2_SOURCE_DIR}/src/externlib/op_util.c")
# Runtime support sources
set(RT_SRC "${OP2_SOURCE_DIR}/src/core/op_rt_support.c")

#------------------------------------------------------------------------------
# Check for packages

message(STATUS "Checking for optional packages...")

if(OP2_WITH_HDF5)
  find_package(HDF5)
  if(HDF5_FOUND)
    message(STATUS "HDF5 found")
  endif()
endif()

if(OP2_WITH_CUDA)
  find_package(CUDA)
  if(CUDA_FOUND)
    message(STATUS "CUDA found")
  endif()
endif()

if(OP2_WITH_MPI)
  find_package(MPI)
  if(MPI_FOUND)
    message(STATUS "MPI found")
  endif()
  find_package(ParMETIS)
  if(PARMETIS_FOUND)
    message(STATUS "ParMETIS found")
  endif()
  find_package(ScotchPT)
  if(SCOTCH_FOUND)
    message(STATUS "PTScotch found")
  endif()
endif()

#------------------------------------------------------------------------------
# Configure targets

message(STATUS "Configuring targets...")

if(OP2_WITH_SEQ)
  message(STATUS "Configuring OP2 sequential library")
  add_subdirectory(sequential)
endif()

if(OP2_WITH_OPENMP)
  message(STATUS "Configuring OP2 OpenMP library")
  add_subdirectory(openmp)
endif()

if(MPI_FOUND AND HDF5_FOUND)
  if(OP2_WITH_HDF5)
    message(STATUS "Configuring OP2 HDF5 library")
    add_subdirectory(externlib)
  endif()
  if(OP2_WITH_MPI AND PARMETIS_FOUND AND SCOTCH_FOUND)
    message(STATUS "Configuring OP2 MPI library")
    set(OP2_MPI_DEFINITIONS -DPTSCOTCH -DPARMETIS)
    set(OP2_MPI_INCLUDE_DIRS ${HDF5_INCLUDE_DIRS} ${PARMETIS_INCLUDE_DIRS}
      ${SCOTCH_INCLUDE_DIRS} ${MPI_INCLUDE_PATH})
    add_subdirectory(mpi)
  endif()
endif()

if(OP2_WITH_CUDA AND CUDA_FOUND)
  message(STATUS "Configuring OP2 CUDA library")
  add_subdirectory(cuda)
endif()

#------------------------------------------------------------------------------
# Export configuration

# Add all targets to the build-tree export set
export(TARGETS ${OP2_TARGETS_BUILT}
  FILE "${PROJECT_BINARY_DIR}/OP2LibraryDepends.cmake")

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE OP2)

# Create the OP2Config.cmake and OP2ConfigVersion files
configure_file(${OP2_CMAKE_DIR}/OP2Config.cmake.in
  "${PROJECT_BINARY_DIR}/OP2Config.cmake" @ONLY)
install(FILES
  "${PROJECT_BINARY_DIR}/OP2Config.cmake"
  DESTINATION "${INSTALLATION_CMAKE_DIR}")

# Install the export set for use with the install-tree
install(EXPORT OP2LibraryDepends DESTINATION
  "${INSTALLATION_CMAKE_DIR}"
  COMPONENT dev)
