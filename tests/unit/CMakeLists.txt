# The loop runtime's layering: what each layer includes, where device headers
# appear, and which files read variables.
add_test(NAME rt_layering COMMAND "${PROJECT_SOURCE_DIR}/scripts/check_rt_layering.sh")
set_tests_properties(rt_layering PROPERTIES LABELS "unit")

if(NOT TARGET op2_seq)
    return()
endif()

# Host-only tests of the environment parser and the par_loop support: the
# planning building blocks and the backends' plans over them, the strategy
# ladder, the autotuner and the JIT's policy, parameters and build cache, and
# the translated kernels' intrinsics, against the sequential library.
foreach(_test env_test set_colouring_test plan_chunks_test plan_cache_test gpu_chunk_plan_test
              loop_ladder_test loop_autotune_test loop_jit_test kernel_prelude_test)
    add_executable(${_test} ${_test}.cpp)
    target_link_libraries(${_test} PRIVATE op2_seq)
    add_test(NAME ${_test} COMMAND ${_test})
    set_tests_properties(${_test} PROPERTIES LABELS "unit")
endforeach()

# The backends' chunk plans, host code their libraries build.
target_sources(plan_chunks_test PRIVATE "${PROJECT_SOURCE_DIR}/op2/src/rt/cpu/op_cpu_chunk_plan.cpp")
target_sources(gpu_chunk_plan_test PRIVATE "${PROJECT_SOURCE_DIR}/op2/src/rt/gpu/op_gpu_chunk_plan.cpp")

# The OpenMP strategies, as the generated code runs them.
if(TARGET op2_openmp)
    add_executable(cpu_strategy_test cpu_strategy_test.cpp)
    target_link_libraries(cpu_strategy_test PRIVATE op2_openmp)
    add_test(NAME cpu_strategy_test COMMAND cpu_strategy_test)
    set_tests_properties(cpu_strategy_test PROPERTIES LABELS "unit")

    # A dat argument of the wrong dim or element type stops the run.
    add_executable(dat_check_test dat_check_test.cpp)
    target_link_libraries(dat_check_test PRIVATE op2_openmp)
    add_test(NAME dat_check_dim COMMAND dat_check_test dim)
    add_test(NAME dat_check_type COMMAND dat_check_test type)
    set(_passes "error: argument 0 of check passes dat q, of dim 2 and 8-byte elements, as")
    set_tests_properties(dat_check_dim PROPERTIES LABELS "unit"
        PASS_REGULAR_EXPRESSION "${_passes} dim 1 of double")
    set_tests_properties(dat_check_type PROPERTIES LABELS "unit"
        PASS_REGULAR_EXPRESSION "${_passes} dim 2 of float")
endif()
