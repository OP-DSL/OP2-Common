message(STATUS "Setting Intel C compiler options")
include(Compiler/Intel)
__compiler_intel(C)

# ignore remark #981: operands are evaluated in unspecified order
SET(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} -std=c99")
