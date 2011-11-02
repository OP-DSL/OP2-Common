# Top level CMakeLists.txt file for OP2

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

#------------------------------------------------------------------------------
# Set project name and version number

project(OP2)

# Set location of our FindFoo.cmake modules
set(OP2_CMAKE_DIR "${OP2_SOURCE_DIR}/../../cmake" CACHE INTERNAL "")
set(CMAKE_MODULE_PATH "${OP2_CMAKE_DIR}/modules")

#------------------------------------------------------------------------------
# General configuration

# Set verbose output while testing CMake
set(CMAKE_VERBOSE_MAKEFILE TRUE)

# Set CMake options, see `cmake --help-policy CMP000x`
if (COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
  cmake_policy(SET CMP0004 OLD)
endif()

# Set location of our FindFoo.cmake modules
#set(OP2_CMAKE_DIR "${OP2_SOURCE_DIR}/cmake" CACHE INTERNAL "")
#set(CMAKE_MODULE_PATH "${OP2_CMAKE_DIR}/modules")

#------------------------------------------------------------------------------
# Configurable options for how we want to build

option(OP2_WITH_CUDA "Build OP2 with NVIDIA CUDA support." OFF)
option(BUILD_SHARED_LIBS "Build OP2 with shared libraries." ON)

#------------------------------------------------------------------------------
# Compiler flags

# Default build type (can be overridden by user)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
    "Choose the type of build, options are: Debug Developer MinSizeRel Release RelWithDebInfo." FORCE)
endif()

# Check for some C compiler flags
include(CheckCCompilerFlag)

# C99 is required
CHECK_C_COMPILER_FLAG(-std=c99 HAVE_C99)
if (HAVE_C99)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
else()
  message(FATAL_ERROR "C99 support for the C compiler is required.")
endif()

CHECK_C_COMPILER_FLAG(-pipe HAVE_PIPE)
if (HAVE_PIPE)
  set(OP2_C_DEVELOPER_FLAGS "-pipe ${OP2_C_DEVELOPER_FLAGS}")
endif()

CHECK_C_COMPILER_FLAG("-Wall -Werror -pedantic" HAVE_PEDANTIC)
if (HAVE_PEDANTIC)
  set(OP2_C_DEVELOPER_FLAGS "-Wall -Werror -pedantic ${OP2_C_DEVELOPER_FLAGS}")
endif()

CHECK_C_COMPILER_FLAG(-g HAVE_DEBUG)
if (HAVE_DEBUG)
  set(OP2_C_DEVELOPER_FLAGS "-g ${OP2_C_DEVELOPER_FLAGS}")
endif()

CHECK_C_COMPILER_FLAG(-O2 HAVE_O2_OPTIMISATION)
if (HAVE_O2_OPTIMISATION)
  set(OP2_C_DEVELOPER_FLAGS "-O2 ${OP2_C_DEVELOPER_FLAGS}")
endif()

# Set 'Developer' build type flags
set(CMAKE_C_FLAGS_DEVELOPER ${OP2_C_DEVELOPER_FLAGS} CACHE STRING
  "Flags used by the compiler during development.")

# Check for some CXX compiler flags
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-pipe HAVE_PIPE)
if (HAVE_PIPE)
  set(OP2_CXX_DEVELOPER_FLAGS "-pipe ${OP2_CXX_DEVELOPER_FLAGS}")
endif()

CHECK_CXX_COMPILER_FLAG("-Wall -Werror" HAVE_PEDANTIC)
if (HAVE_PEDANTIC)
  set(OP2_CXX_DEVELOPER_FLAGS "-Wall -Werror ${OP2_CXX_DEVELOPER_FLAGS}")
endif()

CHECK_CXX_COMPILER_FLAG(-std=c++98 HAVE_STD)
if (HAVE_STD)
  set(OP2_CXX_DEVELOPER_FLAGS "-std=c++98 ${OP2_CXX_DEVELOPER_FLAGS}")
endif()

CHECK_CXX_COMPILER_FLAG(-g HAVE_DEBUG)
if (HAVE_DEBUG)
  set(OP2_CXX_DEVELOPER_FLAGS "-g ${OP2_CXX_DEVELOPER_FLAGS}")
endif()

CHECK_CXX_COMPILER_FLAG(-O2 HAVE_O2_OPTIMISATION)
if (HAVE_O2_OPTIMISATION)
  set(OP2_CXX_DEVELOPER_FLAGS "-O2 ${OP2_CXX_DEVELOPER_FLAGS}")
endif()

# Set 'Developer' build type flags
set(CMAKE_CXX_FLAGS_DEVELOPER ${OP2_CXX_DEVELOPER_FLAGS} CACHE STRING
  "Flags used by the compiler during development.")

#------------------------------------------------------------------------------
# Run tests to find required packages

# Check for PETSc
#find_package(PETSc REQUIRED)

#------------------------------------------------------------------------------
# Run tests to find optional packages

# Check for CUDA
if (OP2_WITH_CUDA)
  find_package(CUDA)
endif()

#------------------------------------------------------------------------------
# Installation directories
set(OP2_BIN_DIR "bin" CACHE PATH "Binary installation directory.")
set(OP2_LIB_DIR "lib" CACHE PATH "Library installation directory.")
set(OP2_INCLUDE_DIR "include" CACHE PATH "C/C++ header installation directory.")
set(OP2_APPS_DIR "apps" CACHE PATH "Apps installation directory.")

#------------------------------------------------------------------------------
# Install header files

file(GLOB HEADERS include/*.h)
install(FILES ${HEADERS} DESTINATION ${OP2_INCLUDE_DIR} COMPONENT Development)

#------------------------------------------------------------------------------
# Set include directories

# Add compiler include directories
include_directories(${OP2_SOURCE_DIR}/include)

#------------------------------------------------------------------------------
# OP2 source directories

add_subdirectory(src)

#------------------------------------------------------------------------------
# OP2 examples

#set(OP2_EXAMPLES
  #airfoil
  #jac1
  #jac2
  #laplace1d
#)

# Add tests but do not add to default target
#foreach(EXAMPLE ${OP2_EXAMPLES})
  #add_subdirectory(examples/C/${EXAMPLE})
  #set(EXAMPLE_TARGETS ${EXAMPLE}_seq ${EXAMPLE_TARGETS})
#endforeach()

#ADD_CUSTOM_TARGET( examples DEPENDS ${EXAMPLE_TARGETS} COMMENT "Build examples" )
