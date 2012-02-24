#------------------------------------------------------------------------------
# Check for packages

message(STATUS "Checking for optional packages...")

set(OP2_MPI_INCLUDE_DIRS)
set(OP2_MPI_DEFINITIONS)
set(OP2_MPI_LIBRARIES)

if(OP2_WITH_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    message(STATUS "OpenMP found")
  else()
    set(OP2_WITH_OPENMP OFF)
  endif()
endif()

if(OP2_WITH_HDF5)
  if(NOT BUILD_SHARED_LIBS)
    set(HDF5_USE_STATIC_LIBRARIES)
  endif()
  find_package(HDF5)
  if(HDF5_FOUND)
    message(STATUS "HDF5 found")
    set(OP2_MPI_INCLUDE_DIRS ${HDF5_INCLUDE_DIRS} ${OP2_MPI_INCLUDE_DIRS})
    set(OP2_MPI_LIBRARIES ${OP2_MPI_LIBRARIES} ${HDF5_LIBRARIES})
  else()
    set(OP2_WITH_HDF5 OFF)
  endif()
endif()

if(OP2_WITH_CUDA)
  find_package(CUDA)
  if(CUDA_FOUND)
    message(STATUS "CUDA found")
    # Do not propagate C/CXX flags to nvcc
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    # These need to be separated by semicolon
    set(OP2_NVCC_FLAGS -O3;-arch=sm_20;-Xptxas;-dlcm=ca;-Xptxas=-v;-use_fast_math
      CACHE STRING "Flags used by NVCC during all build types.")
    set(OP2_NVCC_DEVELOPER_FLAGS -g;-G;-Xcompiler=-O2;-Xcompiler=-g;-Xcompiler=-Wall;-Xcompiler=-pipe
      CACHE STRING "Flags used by NVCC during developer builds.")
    set(CUDA_NVCC_FLAGS ${OP2_NVCC_FLAGS})
    set(CUDA_NVCC_FLAGS_DEVELOPER ${OP2_NVCC_DEVELOPER_FLAGS})
  else()
    set(OP2_WITH_CUDA OFF)
  endif()
endif()

if(OP2_WITH_MPI)
  find_package(MPI)
  if(MPI_FOUND)
    message(STATUS "MPI found")
    set(OP2_MPI_INCLUDE_DIRS ${MPI_INCLUDE_PATH} ${OP2_MPI_INCLUDE_DIRS})
    set(OP2_MPI_LIBRARIES ${MPI_LIBRARIES} ${OP2_MPI_LIBRARIES})
    if(CMAKE_VERBOSE_CONFIGURE)
      set(SCOTCH_DEBUG TRUE)
    endif()
    # Parallel graph partitioner: Try PTScotch first, fall back to ParMETIS
    if(OP2_WITH_PTSCOTCH)
      find_package(ScotchPT)
      if(SCOTCH_FOUND)
        message(STATUS "PTScotch found")
        set(OP2_MPI_DEFINITIONS "${OP2_MPI_DEFINITIONS} -DPTSCOTCH")
        set(OP2_MPI_INCLUDE_DIRS ${SCOTCH_INCLUDE_DIRS} ${OP2_MPI_INCLUDE_DIRS})
        set(OP2_MPI_LIBRARIES ${OP2_MPI_LIBRARIES} ${SCOTCH_LIBRARIES})
      else()
        set(OP2_WITH_PTSCOTCH OFF)
      endif()
    endif()
    if(OP2_WITH_PARMETIS AND NOT OP2_WITH_PTSCOTCH)
      find_package(ParMETIS)
      if(PARMETIS_FOUND)
        message(STATUS "ParMETIS found")
        set(OP2_MPI_DEFINITIONS "${OP2_MPI_DEFINITIONS} -DPARMETIS")
        set(OP2_MPI_INCLUDE_DIRS ${PARMETIS_INCLUDE_DIRS} ${OP2_MPI_INCLUDE_DIRS})
        set(OP2_MPI_LIBRARIES ${OP2_MPI_LIBRARIES} ${PARMETIS_LIBRARIES})
      else()
        set(OP2_WITH_PARMETIS OFF)
      endif()
    endif()
  else()
    set(OP2_WITH_MPI OFF)
  endif()
endif()

#------------------------------------------------------------------------------
# Common source files

# Core library sources
set(COMMON_SRC "${OP2_SOURCE_DIR}/src/core/op_lib_core.c")
# Utility sources
set(UTIL_SRC "${OP2_SOURCE_DIR}/src/externlib/op_util.c")
# Runtime support sources
set(RT_SRC "${OP2_SOURCE_DIR}/src/core/op_rt_support.c")

#------------------------------------------------------------------------------
# Configuration

set(OP2_TARGETS_EXPORT_SET OP2LibraryDepends)
# Reset build-tree export set
export(TARGETS FILE "${PROJECT_BINARY_DIR}/${OP2_TARGETS_EXPORT_SET}.cmake")

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

if(OP2_WITH_HDF5)
  message(STATUS "Configuring OP2 HDF5 library")
  add_subdirectory(externlib)
endif()

if(OP2_WITH_MPI)
  message(STATUS "Configuring OP2 MPI library")
  add_subdirectory(mpi)
endif()

if(OP2_WITH_CUDA)
  message(STATUS "Configuring OP2 CUDA library")
  add_subdirectory(cuda)
endif()

#------------------------------------------------------------------------------
# Export configuration to build tree

# Create the build tree OP2Config.cmake files
set(OP2_CONF_DIR ${PROJECT_BINARY_DIR})
configure_file(${OP2_CMAKE_DIR}/OP2Config.cmake.in
  "${PROJECT_BINARY_DIR}/OP2Config.cmake" @ONLY)

# Store the current build directory in the CMake user package registry.
# This helps dependent projects find and use a package from the current
# project's build tree without help from the user
export(PACKAGE OP2)

#------------------------------------------------------------------------------
# Export configuration to install tree

# Create the install tree OP2Config.cmake files
set(OP2_CONF_DIR ${INSTALLATION_CMAKE_DIR})
configure_file(${OP2_CMAKE_DIR}/OP2Config.cmake.in
  "${PROJECT_BINARY_DIR}/install/OP2Config.cmake" @ONLY)

# Install the configuration and export set for use with the install-tree
install(FILES "${PROJECT_BINARY_DIR}/install/OP2Config.cmake"
  DESTINATION "${INSTALLATION_CMAKE_DIR}")
install(EXPORT ${OP2_TARGETS_EXPORT_SET} DESTINATION "${INSTALLATION_CMAKE_DIR}"
  COMPONENT dev)
