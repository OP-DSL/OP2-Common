# This should only be loaded once
if(DEFINED __OP2_APPLICATION_INCLUDED)
  return()
endif()
set(__OP2_APPLICATION_INCLUDED TRUE)

include(CMakeParseArguments)

function(op2_application APP)
  # Parse arguments
  cmake_parse_arguments(${APP} "" "" "LIBS;SOURCES" ${ARGN})

  # Preparation
  foreach (LIB ${${APP}_LIBS})
    # Skip if the required library hasn't been build
    if (NOT TARGET ${LIB})
      message(STATUS "Library ${LIB} not available, skipping application ${APP}")
      return()
    endif()

    if (${LIB} STREQUAL op2_cuda)
      find_package(CUDA)
      if (NOT CUDA_FOUND)
        return()
      endif()
      set(CUDA_ENABLED TRUE)
    elseif (${LIB} STREQUAL op2_openmp)
      set(OPENMP_ENABLED TRUE)
    elseif (${LIB} STREQUAL op2_mpi)
      add_definitions(${OP2_MPI_DEFINITIONS})
      include_directories(${OP2_MPI_INCLUDE_DIRS})
    endif()

    # Otherwise add to library set
    set(LIBS ${LIBS} ${LIB})
  endforeach()
  message(STATUS "Configuring application ${APP} linking against libraries: ${${APP}_LIBS}")

  # Add the executable
  if (CUDA_ENABLED)
    cuda_add_executable(${APP} ${${APP}_SOURCES})
  else()
    add_executable(${APP} ${${APP}_SOURCES})
  endif()

  if (OPENMP_ENABLED)
    set_target_properties(${APP} PROPERTIES COMPILE_FLAGS
      "${OpenMP_CXX_FLAGS}" LINK_FLAGS "${OpenMP_CXX_FLAGS}")
  endif()

  add_dependencies(${APP} grid)
  target_link_libraries(${APP} ${LIBS})
  install(TARGETS ${APP} RUNTIME DESTINATION ${OP2_APPS_DIR} COMPONENT RuntimeExecutables)
endfunction()
