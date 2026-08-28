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

add_executable(f2c_hier_smem_cache f2c_hier_smem_cache.cu)
target_link_libraries(f2c_hier_smem_cache PRIVATE
    op2_cuda
    CUDA::nvrtc
    CUDA::cuda_driver)
target_compile_definitions(f2c_hier_smem_cache PRIVATE OP2_CUDA)
op2_test_link_nvrtc_builtins(f2c_hier_smem_cache)

add_test(NAME f2c_hier_smem_cache_offline COMMAND f2c_hier_smem_cache)
set_tests_properties(f2c_hier_smem_cache_offline PROPERTIES
    ENVIRONMENT "OP_JIT_ENABLE=0;OP_HIER_SMEM_ATOMICS=auto"
    LABELS "cuda;unit")

add_test(NAME f2c_hier_smem_cache_jit COMMAND f2c_hier_smem_cache)
set_tests_properties(f2c_hier_smem_cache_jit PROPERTIES
    ENVIRONMENT
        "OP_JIT_ENABLE=1;OP_JIT_FORCE=1;OP_JIT_SEQ_COMPILE=1;OP_EXPECT_JIT=1;OP_HIER_SMEM_ATOMICS=auto"
    LABELS "cuda;unit")

if(TARGET op2_mpi_cuda AND MPIEXEC_EXECUTABLE)
    add_executable(hier_smem_cache_mpi_teardown
        hier_smem_cache_mpi_teardown.cu)
    target_link_libraries(hier_smem_cache_mpi_teardown PRIVATE op2_mpi_cuda)
    target_compile_definitions(hier_smem_cache_mpi_teardown PRIVATE OP2_CUDA)

    add_test(NAME hier_smem_cache_mpi_teardown
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                $<TARGET_FILE:hier_smem_cache_mpi_teardown>)
    set_tests_properties(hier_smem_cache_mpi_teardown PROPERTIES
        LABELS "cuda;mpi;unit")
endif()
