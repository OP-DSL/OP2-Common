# Load OP2 configuration (Set OP2_DIR to the directory containing OP2Config.cmake)
find_package(OP2 REQUIRED)

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

# Default installation directory to bin
if (NOT OP2_APPS_DIR)
  set(OP2_APPS_DIR ${CMAKE_INSTALL_PREFIX}/bin)
endif()

include_directories(${OP2_INCLUDE_DIRS})

