if(NOT TARGET op2_seq)
    return()
endif()

add_executable(hier_plan_test hier_plan_test.cpp)
target_link_libraries(hier_plan_test PRIVATE op2_seq)

add_test(NAME hier_plan_test COMMAND hier_plan_test)
set_tests_properties(hier_plan_test PROPERTIES LABELS "unit")
