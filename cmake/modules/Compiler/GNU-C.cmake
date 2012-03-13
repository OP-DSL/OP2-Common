message(STATUS "Setting GNU C compiler options")
include(Compiler/GNU)
__compiler_gnu(C)

SET(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} -std=c99")

# Custom Developer build type, need to create cache variable for that
set(CMAKE_C_FLAGS_DEVELOPER "-O0 -g -Wall -Werror -pedantic -pipe" CACHE STRING
  "Flags used by the compiler during Developer builds.")
mark_as_advanced(CMAKE_C_FLAGS_DEVELOPER)
