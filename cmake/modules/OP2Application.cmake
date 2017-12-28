# This should only be loaded once
if(DEFINED __OP2_APPLICATION_INCLUDED)
  return()
endif()
set(__OP2_APPLICATION_INCLUDED TRUE)

include(CMakeParseArguments)

function(op2_application APP)
  # Parse arguments
  cmake_parse_arguments(${APP} "" "" "DEPENDS;LIBS;SOURCES;BIN" ${ARGN})

  # Preparation
  foreach (LIB ${${APP}_LIBS})
    # Skip if the required library hasn't been build
    if (NOT TARGET ${LIB})
      message(STATUS "Library ${LIB} not available, skipping application ${APP}")
      return()
    endif()

    if (${LIB} MATCHES cuda)
      find_package(CUDA)
      if (NOT CUDA_FOUND)
        return()
      endif()
      set(CUDA_ENABLED TRUE)
    endif()
    if (${LIB} MATCHES openmp)
      set(OPENMP_ENABLED TRUE)
    endif()
    if (${LIB} MATCHES mpi)
      add_definitions(${OP2_MPI_DEFINITIONS})
      include_directories(${OP2_MPI_INCLUDE_DIRS})
    endif()
    if (${LIB} MATCHES hdf5)
      add_definitions(${OP2_HDF5_DEFINITIONS})
      include_directories(${OP2_HDF5_INCLUDE_DIRS})
    endif()

    # Otherwise add to library set
    set(LIBS ${LIBS} ${LIB})
  endforeach()
  message(STATUS "Configuring application ${APP} linking against libraries: ${${APP}_LIBS}")

  # Add the executable
  if (${${APP}_BIN} MATCHES library)
    if (CUDA_ENABLED)
      cuda_add_library(${APP} SHARED ${${APP}_SOURCES})
    else()
      add_library(${APP} SHARED ${${APP}_SOURCES})
    endif()
  elseif (${${APP}_BIN} MATCHES static)
    if (CUDA_ENABLED)
      cuda_add_library(${APP} STATIC ${${APP}_SOURCES})
    else()
      add_library(${APP} STATIC ${${APP}_SOURCES})
    endif()
  else()
    if (CUDA_ENABLED)
      cuda_add_executable(${APP} ${${APP}_SOURCES})
    else()
      add_executable(${APP} ${${APP}_SOURCES})
    endif()
  endif()

  if (OPENMP_ENABLED)
    set_target_properties(${APP} PROPERTIES COMPILE_FLAGS
      "${OpenMP_CXX_FLAGS}" LINK_FLAGS "${OpenMP_CXX_FLAGS}")
  endif()

  if (${APP}_DEPENDS)
    add_dependencies(${APP} ${${APP}_DEPENDS})
  endif()
  target_link_libraries(${APP} ${LIBS})
  install(TARGETS ${APP} RUNTIME DESTINATION ${OP2_APPS_DIR} LIBRARY DESTINATION ${OP2_APPS_DIR} ARCHIVE DESTINATION ${OP2_APPS_DIR} COMPONENT RuntimeExecutables)
endfunction()
