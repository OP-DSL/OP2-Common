message(STATUS "Setting Intel CXX compiler options")

# ignore remark #981: operands are evaluated in unspecified order
SET(CMAKE_CXX_FLAGS_DIAGS_INIT "-Wall -Werror -wd981")

include(Compiler/Intel)
__compiler_intel(CXX)
