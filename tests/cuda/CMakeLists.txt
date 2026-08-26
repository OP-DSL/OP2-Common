if(NOT TARGET op2_cuda OR NOT TARGET CUDA::nvrtc OR NOT TARGET CUDA::cuda_driver)
    return()
endif()

# NVRTC dlopen()s libnvrtc-builtins on its first compile, and that lookup never
# consults the executable's DT_RUNPATH.  Name it as a direct dependency so the
# loader resolves it at startup, exactly as op2_add_app_variants does for the
# c_cuda variants.  See the comment there for the full reasoning.
function(op2_test_link_nvrtc_builtins target)
    if(TARGET CUDA::nvrtc_builtins AND _op2_linker_has_push_state)
        target_link_libraries(${target} PRIVATE
            "-Wl,--push-state,--no-as-needed"
            CUDA::nvrtc_builtins
            "-Wl,--pop-state")
    endif()
endfunction()

add_executable(f2c_dynamic_smem_launch f2c_dynamic_smem_launch.cu)
target_link_libraries(f2c_dynamic_smem_launch PRIVATE
    op2_cuda
    CUDA::nvrtc
    CUDA::cuda_driver)
target_compile_definitions(f2c_dynamic_smem_launch PRIVATE OP2_CUDA)
op2_test_link_nvrtc_builtins(f2c_dynamic_smem_launch)

add_test(NAME f2c_dynamic_smem_launch_offline
    COMMAND f2c_dynamic_smem_launch)
set_tests_properties(f2c_dynamic_smem_launch_offline PROPERTIES
    ENVIRONMENT "OP_JIT_ENABLE=0"
    LABELS "cuda;unit")

add_test(NAME f2c_dynamic_smem_launch_jit
    COMMAND f2c_dynamic_smem_launch)
set_tests_properties(f2c_dynamic_smem_launch_jit PROPERTIES
    ENVIRONMENT
        "OP_JIT_ENABLE=1;OP_JIT_FORCE=1;OP_JIT_SEQ_COMPILE=1;OP_EXPECT_JIT=1"
    LABELS "cuda;unit")
