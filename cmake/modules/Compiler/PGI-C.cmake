message("-- Setting PGI C compiler options")
include(Compiler/PGI)
__compiler_pgi(C)
SET(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} -c99")
set(CMAKE_C_FLAGS_MINSIZEREL_INIT "${CMAKE_C_FLAGS_MINSIZEREL_INIT} -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE_INIT "${CMAKE_C_FLAGS_RELEASE_INIT} -fastsse -Minline=levels:10 -Mvect -Minfo -DNDEBUG")

# Custom Developer build type, need to create cache variable for that
set(CMAKE_C_FLAGS_DEVELOPER "-O2 -gopt -Minform=inform" CACHE STRING
  "Flags used by the compiler during Developer builds.")
