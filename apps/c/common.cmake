# This should only be loaded once
if(DEFINED __APPS_COMMON_INCLUDED)
  return()
endif()
set(__APPS_COMMON_INCLUDED TRUE)

# Load OP2 configuration (Set OP2_DIR to the directory containing OP2Config.cmake)
find_package(OP2 REQUIRED PATHS
  ${OP2-APPS_SOURCE_DIR}/../../op2/c/build
  ${PROJECT_SOURCE_DIR}/../../../op2/c/build
  ${CMAKE_INSTALL_PREFIX}/lib/op2)

# Default installation directory to bin
if (NOT OP2_APPS_DIR)
  set(OP2_APPS_DIR ${CMAKE_INSTALL_PREFIX}/bin)
endif()

option(OP2_BUILD_SP "Build a single precision versions of the OP2 applications."  ON)
option(OP2_BUILD_DP "Build a double precision versions of the OP2 applications."  ON)

set(OP2_BUILD_VARIANTS)
if(OP2_BUILD_SP)
  list(APPEND OP2_BUILD_VARIANTS sp)
endif()
if(OP2_BUILD_DP)
  list(APPEND OP2_BUILD_VARIANTS dp)
endif()

if(BUILD_SHARED_LIBS)
  option(USE_INSTALL_RPATH "Set rpath for installed applications." ON)
  if(USE_INSTALL_RPATH)
    # Append directories in the linker search path of the imported libraries to
    # the rpath. This makes the installed apps also find libraries in the OP2
    # build tree without having to set LD_LIBRARY_PATH.
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  endif()
endif()

# Import compiler flags for all build types
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OP2_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OP2_C_FLAGS}")

# Default build type (can be overridden by user)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
    "Choose the type of build, options are: Debug MinSizeRel Release RelWithDebInfo." FORCE)
endif()
string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)

# FIXME: at the moment we can't build with -Werror flags due to some warnings
# not easily fixable
if(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE})
  STRING(REPLACE "-Werror" "" CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}})
endif()
if(CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE})
  STRING(REPLACE "-Werror" "" CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE} ${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE}})
endif()

include_directories(${OP2_INCLUDE_DIRS})
add_definitions(${OP2_USER_DEFINITIONS})

include(OP2Application)
