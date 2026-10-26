if(NOT TARGET op2_cuda OR NOT TARGET CUDA::nvrtc OR NOT TARGET CUDA::cuda_driver)
    return()
endif()

# NVRTC dlopen()s libnvrtc-builtins on its first compile, and that lookup never
# consults the executable's DT_RUNPATH.  Name it as a direct dependency so the
# loader resolves it at startup, exactly as op2_add_app_variants does for the
# cuda and cuda_f2c variants.  See the comment there for the full reasoning.
function(op2_test_link_nvrtc_builtins target)
    if(TARGET CUDA::nvrtc_builtins AND _op2_linker_has_push_state)
        target_link_libraries(${target} PRIVATE
            "-Wl,--push-state,--no-as-needed"
            CUDA::nvrtc_builtins
            "-Wl,--pop-state")
    endif()
endfunction()

# Runtime tests of the GPU backend, from test kernels: the chunked strategy
# offline (the opt-in to more shared memory differs) and with every kernel
# compiled on its first call, and a compile that fails.
foreach(_test chunk_strategy jit_build_failure)
    add_executable(${_test} ${_test}.cu)
    target_link_libraries(${_test} PRIVATE op2_cuda CUDA::nvrtc CUDA::cuda_driver)
    target_compile_definitions(${_test} PRIVATE OP2_CUDA)
    op2_test_link_nvrtc_builtins(${_test})

    add_test(NAME ${_test}_jit COMMAND ${_test})
    set_tests_properties(${_test}_jit PROPERTIES
        ENVIRONMENT "OP_JIT_LEVEL=2;OP_JIT_FORCE=1;OP_JIT_SEQ_COMPILE=1"
        LABELS "cuda;unit")
endforeach()

add_test(NAME chunk_strategy_offline COMMAND chunk_strategy)
set_tests_properties(chunk_strategy_offline PROPERTIES
    ENVIRONMENT "OP_JIT_LEVEL=0"
    LABELS "cuda;unit")

# The device plan's uploads, read back from the device.
add_executable(chunk_encode chunk_encode.cu)
target_link_libraries(chunk_encode PRIVATE op2_cuda)
target_compile_definitions(chunk_encode PRIVATE OP2_CUDA)
add_test(NAME chunk_encode COMMAND chunk_encode)
set_tests_properties(chunk_encode PROPERTIES LABELS "cuda;unit")

if(TARGET op2_mpi_cuda AND MPIEXEC_EXECUTABLE)
    add_executable(chunk_plan_mpi_teardown chunk_plan_mpi_teardown.cu)
    target_link_libraries(chunk_plan_mpi_teardown PRIVATE op2_mpi_cuda)
    target_compile_definitions(chunk_plan_mpi_teardown PRIVATE OP2_CUDA)

    add_test(NAME chunk_plan_mpi_teardown
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                $<TARGET_FILE:chunk_plan_mpi_teardown>)
    set_tests_properties(chunk_plan_mpi_teardown PROPERTIES
        LABELS "cuda;mpi;unit")
endif()
