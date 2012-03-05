message(STATUS "Setting PGI CXX compiler options")
include(Compiler/PGI)
__compiler_pgi(CXX)
set(CMAKE_CXX_FLAGS_MINSIZEREL_INIT "${CMAKE_CXX_FLAGS_MINSIZEREL_INIT} -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "${CMAKE_CXX_FLAGS_RELEASE_INIT} -fastsse -Minline=levels:10 -Mvect -Minfo -DNDEBUG")

# Custom Developer build type, need to create cache variable for that
set(CMAKE_CXX_FLAGS_DEVELOPER "-O2 -gopt -Minform=inform --remarks" CACHE STRING
  "Flags used by the compiler during Developer builds.")
mark_as_advanced(CMAKE_CXX_FLAGS_DEVELOPER)
