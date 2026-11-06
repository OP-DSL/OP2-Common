op2_add_functional_tests(
    LANGUAGE cpp
    SOURCES  const_tests.cpp)

op2_add_functional_tests(
    LANGUAGE cpp
    NAME     const_files
    SOURCES  const_files.cpp const_files_step.cpp)
