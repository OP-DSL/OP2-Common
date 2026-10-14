if(NOT TARGET op2_seq)
    return()
endif()

# Host-only tests of the par_loop support against the sequential library.
foreach(_test loop_ladder_test)
    add_executable(${_test} ${_test}.cpp)
    target_link_libraries(${_test} PRIVATE op2_seq)
    add_test(NAME ${_test} COMMAND ${_test})
    set_tests_properties(${_test} PROPERTIES LABELS "unit")
endforeach()
