# Load OP2 configuration (Set OP2_DIR to the directory containing OP2Config.cmake)
find_package(OP2 REQUIRED PATHS
  ${OP2-APPS_SOURCE_DIR}/../../op2/c/build
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

# Generate the airfoil input grid using the NACA grid generator
set(AIRFOIL_MESH_FILE ${CMAKE_BINARY_DIR}/new_grid.dat CACHE FILEPATH
  "Manually select an input grid to override automatic grid generation")
if(EXISTS ${AIRFOIL_MESH_FILE})
  set(AIRFOIL_GENERATE_MESH_INIT OFF)
else()
  message(STATUS "No existing input mesh specified, enable automatic grid generation.")
  set(AIRFOIL_GENERATE_MESH_INIT ON)
  set(AIRFOIL_MESH_FILE ${CMAKE_BINARY_DIR}/new_grid.dat)
endif()
option(AIRFOIL_GENERATE_MESH "Generate input meshes during the build process."
  ${AIRFOIL_GENERATE_MESH_INIT})
install(FILES ${AIRFOIL_MESH_FILE} DESTINATION ${OP2_APPS_DIR} COMPONENT RuntimeInputFiles OPTIONAL)
# Skip if the grid target already exists (i.e. this has already been run)
if(AIRFOIL_GENERATE_MESH AND NOT TARGET grid)

  find_file(AIRFOIL_MESH_GENERATOR naca0012.m PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../mesh_generators)

  # Use Octave if available, otherwise fall back to MATLAB
  find_program(OCTAVE_EXECUTABLE octave hints ${OCTAVE_DIR} ENV OCTAVE_DIR)
  if(NOT OCTAVE_EXECUTABLE)
    find_program(MATLAB_EXECUTABLE matlab hints ${MATLAB_DIR} ENV MATLAB_DIR)
    if(NOT MATLAB_EXECUTABLE)
      message(STATUS "Could not find Octave or MATLAB. Set OCTAVE_DIR and/or MATLAB_DIR to the folder(s) containing the executable(s).

Automatic generation of the input mesh is skipped. Generate it manually by running:
  ${AIRFOIL_MESH_GENERATOR}
or set AIRFOIL_MESH_FILE to manually specify an input mesh")
    else()
      message(STATUS "Generating input mesh with MATLAB")
      set(GENERATE_MESH_CMD "${MATLAB_EXECUTABLE} -nodisplay -nojvm -nodesktop -nosplash -r 'naca0012 old;exit'")
    endif()
  else()
    message(STATUS "Generating input mesh with Octave")
    set(GENERATE_MESH_CMD ${OCTAVE_EXECUTABLE} --eval "naca0012")
  endif()

  if(GENERATE_MESH_CMD)
    # Custom command to generate the grid
    add_custom_command(OUTPUT ${AIRFOIL_MESH_FILE}
      COMMAND ${CMAKE_COMMAND} -E copy ${AIRFOIL_MESH_GENERATOR} .
      COMMAND ${GENERATE_MESH_CMD}
      MAIN_DEPENDENCY ${AIRFOIL_MESH_GENERATOR}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Generating the input grid ${AIRFOIL_MESH_FILE}...")
    # We need a (global) custom target since output of a custom command can
    # be used as a dependency only within the same file. This also makes
    # sure the grid is only generated once if called from multiple files.
    add_custom_target(grid DEPENDS ${AIRFOIL_MESH_FILE})
  endif()

endif(AIRFOIL_GENERATE_MESH AND NOT TARGET grid)
