# This module is shared by multiple languages; use include blocker.
if(__COMPILER_INTEL)
  return()
endif()
set(__COMPILER_INTEL 1)

macro(__compiler_intel lang)
  # Feature flags.
  set(CMAKE_${lang}_VERBOSE_FLAG "-v")

  # Initial configuration flags.
  set(CMAKE_${lang}_FLAGS_INIT "-vec-report")
  set(CMAKE_${lang}_FLAGS_DEBUG_INIT "-g -DDEBUG")
  set(CMAKE_${lang}_FLAGS_MINSIZEREL_INIT "-Os -DNDEBUG")
  set(CMAKE_${lang}_FLAGS_RELEASE_INIT "-O3 -xSSE4.2 -DNDEBUG")
  set(CMAKE_${lang}_FLAGS_RELWITHDEBINFO_INIT "-O2 -g")

  # Custom Developer build type, need to create cache variable for that
  # ignore remark #981: operands are evaluated in unspecified order
  set(CMAKE_${lang}_FLAGS_DEVELOPER "-O2 -g -Wall -Werror -wd981" CACHE STRING
    "Flags used by the compiler during Developer builds.")

  # Preprocessing and assembly rules.
  set(CMAKE_${lang}_CREATE_PREPROCESSED_SOURCE "<CMAKE_${lang}_COMPILER> <DEFINES> <FLAGS> -E <SOURCE> > <PREPROCESSED_SOURCE>")
  set(CMAKE_${lang}_CREATE_ASSEMBLY_SOURCE "<CMAKE_${lang}_COMPILER> <DEFINES> <FLAGS> -S <SOURCE> -o <ASSEMBLY_SOURCE>")
endmacro()
