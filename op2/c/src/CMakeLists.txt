# Core library sources
set(COMMON_SRC "${OP2_SOURCE_DIR}/src/core/op_lib_core.c")
# Utility sources
set(UTIL_SRC "${OP2_SOURCE_DIR}/src/externlib/op_util.c")
# Runtime support sources
set(RT_SRC "${OP2_SOURCE_DIR}/src/core/op_rt_support.c")

#------------------------------------------------------------------------------
# Check for packages
find_package(MPI)
find_package(HDF5)
find_package(CUDA)
find_package(ParMETIS)
find_package(ScotchPT)

add_subdirectory(sequential)
add_subdirectory(openmp)
if(MPI_FOUND)
  if(HDF5_FOUND)
    add_subdirectory(externlib)
    if(PARMETIS_FOUND AND SCOTCH_FOUND)
      add_definitions(-DPTSCOTCH -DPARMETIS)
      add_subdirectory(mpi)
    endif()
  endif()
endif()
if(CUDA_FOUND)
  add_subdirectory(cuda)
endif()
