# This should only be loaded once
if(DEFINED __OP2_APPLICATION_INCLUDED)
  return()
endif()
set(__OP2_APPLICATION_INCLUDED TRUE)

macro(op2_application APP BACKEND)
  # Preparation
  if (${BACKEND} STREQUAL SEQUENTIAL)
    set(LIB op2_seq)
  elseif (${BACKEND} STREQUAL CUDA)
    set(LIB op2_cuda)
  elseif (${BACKEND} STREQUAL OPENMP)
    set(LIB op2_openmp)
  elseif (${BACKEND} STREQUAL MPI)
    set(LIB op2_mpi)
    add_definitions(${OP2_MPI_DEFINITIONS})
    include_directories(${OP2_MPI_INCLUDE_DIRS})
  elseif (${BACKEND} STREQUAL HDF5)
    set(LIB op2_hdf5)
  else()
    message(WARNING "Invalid backend ${BACKEND} for application ${APP}")
  endif()

  # Skip if the required library hasn't been build
  if (NOT TARGET ${LIB})
    return()
  endif()
  message(STATUS "Configuring application ${APP} for the ${BACKEND} backend")

  # Add the executable
  if (${BACKEND} STREQUAL CUDA)
    find_package(CUDA)
    if (NOT CUDA_FOUND)
      return()
    endif()
    set(LIB op2_cuda)
    cuda_add_executable(${APP} ${ARGN})
  else()
    add_executable(${APP} ${ARGN})
  endif()

  if (${BACKEND} STREQUAL OPENMP)
    set_target_properties(airfoil_${VARIANT}_openmp PROPERTIES COMPILE_FLAGS
      "${OpenMP_CXX_FLAGS}" LINK_FLAGS "${OpenMP_CXX_FLAGS}")
  endif()

  add_dependencies(${APP} grid)
  target_link_libraries(${APP} ${LIB})
  install(TARGETS ${APP} RUNTIME DESTINATION ${OP2_APPS_DIR} COMPONENT RuntimeExecutables)
endmacro()
