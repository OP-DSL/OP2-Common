message(STATUS "Setting GNU CXX compiler options")
include(Compiler/GNU)
__compiler_gnu(CXX)

SET(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} -std=c++98")

# Custom Developer build type, need to create cache variable for that
set(CMAKE_CXX_FLAGS_DEVELOPER "-O0 -g -Wall -Werror -pipe -Wno-long-long -std=c++98" CACHE STRING
  "Flags used by the compiler during Developer builds.")
mark_as_advanced(CMAKE_CXX_FLAGS_DEVELOPER)
