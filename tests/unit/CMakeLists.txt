if(NOT TARGET op2_seq)
    return()
endif()

add_executable(hier_smem_plan_test hier_smem_plan_test.cpp)
target_link_libraries(hier_smem_plan_test PRIVATE op2_seq)

add_test(NAME hier_smem_plan_test COMMAND hier_smem_plan_test)
set_tests_properties(hier_smem_plan_test PROPERTIES LABELS "unit")
