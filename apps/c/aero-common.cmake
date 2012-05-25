# This should only be loaded once
if(DEFINED __AERO_COMMON_INCLUDED)
  return()
endif()
set(__AERO_COMMON_INCLUDED TRUE)

# Generate the AERO input grid using the NACA grid generator
set(AERO_MESH_FILE ${CMAKE_BINARY_DIR}/FE_grid.dat CACHE FILEPATH
  "Manually select an input grid to override automatic grid generation")
if(EXISTS ${AERO_MESH_FILE})
  set(AERO_GENERATE_MESH_INIT OFF)
else()
  message(STATUS "No existing input mesh specified, enable automatic grid generation.")
  set(AERO_GENERATE_MESH_INIT ON)
  set(AERO_MESH_FILE ${CMAKE_BINARY_DIR}/FE_grid.dat)
endif()
option(AERO_GENERATE_MESH "Generate input meshes during the build process."
  ${AERO_GENERATE_MESH_INIT})
install(FILES ${AERO_MESH_FILE} DESTINATION ${OP2_APPS_DIR} COMPONENT RuntimeInputFiles OPTIONAL)

if(AERO_GENERATE_MESH)

  find_file(AERO_MESH_GENERATOR naca_fem.m PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../mesh_generators
    ${CMAKE_CURRENT_SOURCE_DIR}/../mesh_generators)

  # Use Octave if available, otherwise fall back to MATLAB
  find_program(OCTAVE_EXECUTABLE octave hints ${OCTAVE_DIR} ENV OCTAVE_DIR)
  if(NOT OCTAVE_EXECUTABLE)
    find_program(MATLAB_EXECUTABLE matlab hints ${MATLAB_DIR} ENV MATLAB_DIR)
    if(NOT MATLAB_EXECUTABLE)
      message(STATUS "Could not find Octave or MATLAB. Set OCTAVE_DIR and/or MATLAB_DIR to the folder(s) containing the executable(s).")
      message(STATUS "  Automatic generation of the input mesh is skipped. Generate it manually by running:")
      message(STATUS "    ${AERO_MESH_GENERATOR}")
      message(STATUS "  or set AERO_MESH_FILE to manually specify an input mesh")
    else()
      message(STATUS "Generating input mesh with MATLAB")
      set(GENERATE_MESH_CMD "${MATLAB_EXECUTABLE} -nodisplay -nojvm -nodesktop -nosplash -r 'naca_fem;exit'")
    endif()
  else()
    message(STATUS "Generating input mesh with Octave")
    set(GENERATE_MESH_CMD ${OCTAVE_EXECUTABLE} --eval "naca_fem")
  endif()

  if(GENERATE_MESH_CMD)
    # Custom command to generate the grid
    add_custom_command(OUTPUT ${AERO_MESH_FILE}
      COMMAND ${CMAKE_COMMAND} -E copy ${AERO_MESH_GENERATOR} .
      COMMAND ${GENERATE_MESH_CMD}
      MAIN_DEPENDENCY ${AERO_MESH_GENERATOR}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Generating the input grid ${AERO_MESH_FILE}...")
    # We need a (global) custom target since output of a custom command can
    # be used as a dependency only within the same file. This also makes
    # sure the grid is only generated once if called from multiple files.
    add_custom_target(grid DEPENDS ${AERO_MESH_FILE})
  endif()

endif(AERO_GENERATE_MESH)
