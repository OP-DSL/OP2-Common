add_subdirectory(unit)
add_subdirectory(functional)

if(CMAKE_CUDA_COMPILER)
    add_subdirectory(cuda)
endif()

if(OP2_ENABLE_APP_TESTS)
    if(NOT OP2_ENABLE_APPS)
        message(FATAL_ERROR "OP2_ENABLE_APP_TESTS needs OP2_ENABLE_APPS")
    endif()
    add_subdirectory(apps)
endif()
