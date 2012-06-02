message(STATUS "Setting Intel C compiler options")

# ignore remark #981: operands are evaluated in unspecified order
SET(CMAKE_C_FLAGS_DIAGS_INIT "-Wall -Werror -wd981")

include(Compiler/Intel)
__compiler_intel(C)

SET(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} -std=c99")
