# This should only be loaded once
if(DEFINED __GENERATE_MESH_INCLUDED)
  return()
endif()
set(__GENERATE_MESH_INCLUDED TRUE)

function(generate_mesh APP MESH_FILE MESH_GENERATOR)
  # Generate the input mesh using the NACA grid generator
  set(${APP}_MESH_FILE ${CMAKE_BINARY_DIR}/${MESH_FILE} CACHE FILEPATH
    "Manually select an input grid to override automatic grid generation")
  if(EXISTS ${${APP}_MESH_FILE})
    set(${APP}_GENERATE_MESH_INIT OFF)
  else()
    message(STATUS "No existing input mesh specified, enable automatic grid generation.")
    set(${APP}_GENERATE_MESH_INIT ON)
    set(${APP}_MESH_FILE ${CMAKE_BINARY_DIR}/${MESH_FILE})
  endif()
  option(${APP}_GENERATE_MESH "Generate input meshes during the build process."
    ${${APP}_GENERATE_MESH_INIT})
  install(FILES ${${APP}_MESH_FILE} DESTINATION ${OP2_APPS_DIR} COMPONENT RuntimeInputFiles OPTIONAL)

  if(${APP}_GENERATE_MESH)

    find_file(${APP}_MESH_GENERATOR ${MESH_GENERATOR}.m PATHS
      ${CMAKE_CURRENT_SOURCE_DIR}/../../mesh_generators
      ${CMAKE_CURRENT_SOURCE_DIR}/../mesh_generators)

    # Use Octave if available, otherwise fall back to MATLAB
    find_program(OCTAVE_EXECUTABLE octave hints ${OCTAVE_DIR} ENV OCTAVE_DIR)
    if(NOT OCTAVE_EXECUTABLE)
      find_program(MATLAB_EXECUTABLE matlab hints ${MATLAB_DIR} ENV MATLAB_DIR)
      if(NOT MATLAB_EXECUTABLE)
        message(STATUS "Could not find Octave or MATLAB. Set OCTAVE_DIR and/or MATLAB_DIR to the folder(s) containing the executable(s).")
        message(STATUS "  Automatic generation of the input mesh is skipped. Generate it manually by running:")
        message(STATUS "    ${${APP}_MESH_GENERATOR}")
        message(STATUS "  or set ${APP}_MESH_FILE to manually specify an input mesh")
      else()
        message(STATUS "Generating input mesh with MATLAB")
        set(GENERATE_MESH_CMD ${MATLAB_EXECUTABLE} -nodisplay -nojvm -nodesktop -nosplash -r '${MESH_GENERATOR};exit')
      endif()
    else()
      message(STATUS "Generating input mesh with Octave")
      set(GENERATE_MESH_CMD ${OCTAVE_EXECUTABLE} --eval "${MESH_GENERATOR}")
    endif()

    if(GENERATE_MESH_CMD)
      # Custom command to generate the grid
      add_custom_command(OUTPUT ${${APP}_MESH_FILE}
        COMMAND ${CMAKE_COMMAND} -E copy ${${APP}_MESH_GENERATOR} .
        COMMAND ${GENERATE_MESH_CMD}
        MAIN_DEPENDENCY ${${APP}_MESH_GENERATOR}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating the input grid ${${APP}_MESH_FILE}...")
      # We need a (global) custom target since output of a custom command can
      # be used as a dependency only within the same file. This also makes
      # sure the grid is only generated once if called from multiple files.
      add_custom_target(${APP}_grid DEPENDS ${${APP}_MESH_FILE})
    endif()

  endif(${APP}_GENERATE_MESH)
endfunction()
