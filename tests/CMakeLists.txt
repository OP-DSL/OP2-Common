add_subdirectory(functional)

if(CMAKE_CUDA_COMPILER)
    add_subdirectory(cuda)
endif()
