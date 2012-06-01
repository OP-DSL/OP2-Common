message(STATUS "Setting GNU Fortran compiler options")

# Include from CMake module directory (need to clear CMake Module path)
set(CMAKE_MODULE_PATH_SAVE ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "")

include(Compiler/GNU-Fortran)

# Restore CMake Module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH_SAVE})
unset(CMAKE_MODULE_PATH)

# Custom Developer build type, need to create cache variable for that
set(CMAKE_Fortran_FLAGS_DEVELOPER "-O2 -g -Wall -Werror -pedantic -pipe" CACHE STRING
  "Flags used by the compiler during Developer builds.")
mark_as_advanced(CMAKE_Fortran_FLAGS_DEVELOPER)

