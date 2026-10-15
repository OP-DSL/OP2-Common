if(NOT TARGET op2_seq)
    return()
endif()

# Host-only tests of the par_loop support: the planning building blocks and
# the backends' plans over them, the strategy ladder and the autotuner,
# against the sequential library.
foreach(_test set_colouring_test plan_chunks_test gpu_chunk_plan_test chunk_encode_test
              loop_ladder_test loop_autotune_test)
    add_executable(${_test} ${_test}.cpp)
    target_link_libraries(${_test} PRIVATE op2_seq)
    add_test(NAME ${_test} COMMAND ${_test})
    set_tests_properties(${_test} PROPERTIES LABELS "unit")
endforeach()

# The OpenMP strategies, header-only over the same library.
if(TARGET OpenMP::OpenMP_CXX)
    add_executable(cpu_strategy_test cpu_strategy_test.cpp)
    target_link_libraries(cpu_strategy_test PRIVATE op2_seq OpenMP::OpenMP_CXX)
    add_test(NAME cpu_strategy_test COMMAND cpu_strategy_test)
    set_tests_properties(cpu_strategy_test PROPERTIES LABELS "unit")
endif()
