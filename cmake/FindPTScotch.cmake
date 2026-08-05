# FindPTScotch.cmake
#
# Locates PT-Scotch (parallel graph partitioning library).
# Sets imported target PTScotch::PTScotch on success.
#
# Hints (cache and env, checked in order):
#   PTScotch_ROOT   - matches find_package(PTScotch)
#   SCOTCH_DIR      - Spack / module-system convention
#   PTSCOTCH_DIR    - alternate module-system convention
#
# Validates that PT-Scotch was built with 64-bit indices (sizeof(SCOTCH_Num)==8)
# as required by OP2 (idx_g_t is long long).  Fails if a 32-bit build is found.
#
# Cache variables (advanced, but set any of them by hand to point this module
# at an install whose layout the search below doesn't guess correctly):
#   PTScotch_INCLUDE_DIR      - directory holding ptscotch.h / scotch.h
#   PTScotch_LIBRARY          - libptscotch
#   PTScotch_ERR_LIBRARY      - libptscotcherr
#   PTScotch_SCOTCH_LIBRARY   - libscotch (serial, needed at link time)

include(FindPackageHandleStandardArgs)

set(_ptscotch_hints
    "${PTScotch_ROOT}"      "$ENV{PTScotch_ROOT}"
    "${SCOTCH_DIR}"         "$ENV{SCOTCH_DIR}"
    "${PTSCOTCH_DIR}"       "$ENV{PTSCOTCH_DIR}"
)

find_path(PTScotch_INCLUDE_DIR NAMES ptscotch.h scotch.h
    HINTS ${_ptscotch_hints} PATH_SUFFIXES include NO_DEFAULT_PATH)
if(NOT PTScotch_INCLUDE_DIR)
    find_path(PTScotch_INCLUDE_DIR NAMES ptscotch.h scotch.h PATH_SUFFIXES include)
endif()

find_library(PTScotch_LIBRARY NAMES ptscotch
    HINTS ${_ptscotch_hints} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
if(NOT PTScotch_LIBRARY)
    find_library(PTScotch_LIBRARY NAMES ptscotch PATH_SUFFIXES lib lib64)
endif()

find_library(PTScotch_ERR_LIBRARY NAMES ptscotcherr
    HINTS ${_ptscotch_hints} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
if(NOT PTScotch_ERR_LIBRARY)
    find_library(PTScotch_ERR_LIBRARY NAMES ptscotcherr PATH_SUFFIXES lib lib64)
endif()

# Scotch (serial) is needed at link time alongside PT-Scotch
find_library(PTScotch_SCOTCH_LIBRARY NAMES scotch
    HINTS ${_ptscotch_hints} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
if(NOT PTScotch_SCOTCH_LIBRARY)
    find_library(PTScotch_SCOTCH_LIBRARY NAMES scotch PATH_SUFFIXES lib lib64)
endif()

# Validate 64-bit index width via a compile-time sizeof check.
# ptscotch.h includes mpi.h, so MPI include dirs must be on the search path.
if(PTScotch_INCLUDE_DIR AND PTScotch_LIBRARY)
    set(_ptscotch_mpi_incs "")
    if(MPI_C_INCLUDE_DIRS)
        set(_ptscotch_mpi_incs "${MPI_C_INCLUDE_DIRS}")
    elseif(MPI_CXX_INCLUDE_DIRS)
        set(_ptscotch_mpi_incs "${MPI_CXX_INCLUDE_DIRS}")
    else()
        find_path(PTScotch_MPI_INCLUDE_DIR NAMES mpi.h)
        if(PTScotch_MPI_INCLUDE_DIR)
            set(_ptscotch_mpi_incs "${PTScotch_MPI_INCLUDE_DIR}")
        endif()
    endif()

    # Probe in whichever language is enabled: OP2's own build enables C, but
    # a consumer re-finding this through OP2Config.cmake may have only CXX,
    # and check_<lang>_source_compiles is a hard error on a disabled
    # language. scotch.h compiles cleanly either way.
    if(CMAKE_C_COMPILER_LOADED)
        set(_ptscotch_probe_lang C)
    else()
        set(_ptscotch_probe_lang CXX)
    endif()

    include(CheckSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_INCLUDES "${PTScotch_INCLUDE_DIR}" ${_ptscotch_mpi_incs})
    check_source_compiles(${_ptscotch_probe_lang} "
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <scotch.h>
/* Array-trick sizeof check: zero-length array is a compile error */
typedef char _scotch_idx64_check[sizeof(SCOTCH_Num) == 8 ? 1 : -1];
int main(void) { return 0; }
" PTScotch_HAS_IDX64)
    cmake_pop_check_state()
    unset(_ptscotch_probe_lang)
endif()

# No zlib: libscotch references gz* only from its compressed-file members, and
# OP2 builds graphs in memory - it calls no SCOTCH_*Load/*Save - so those
# members are never pulled out of the archive.
if(PTScotch_INCLUDE_DIR AND PTScotch_LIBRARY)
    if(NOT TARGET PTScotch::PTScotch)
        add_library(PTScotch::PTScotch UNKNOWN IMPORTED)
        set_target_properties(PTScotch::PTScotch PROPERTIES
            IMPORTED_LOCATION             "${PTScotch_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${PTScotch_INCLUDE_DIR}")
        foreach(_extra PTScotch_ERR_LIBRARY PTScotch_SCOTCH_LIBRARY)
            if(${_extra})
                set_property(TARGET PTScotch::PTScotch APPEND PROPERTY
                    INTERFACE_LINK_LIBRARIES "${${_extra}}")
            endif()
        endforeach()
    endif()
endif()

find_package_handle_standard_args(PTScotch
    REQUIRED_VARS PTScotch_INCLUDE_DIR PTScotch_LIBRARY PTScotch_HAS_IDX64
    FAIL_MESSAGE  "PT-Scotch not found or built with 32-bit indices (set PTScotch_ROOT and rebuild with --64bit-indices)")

mark_as_advanced(PTScotch_INCLUDE_DIR PTScotch_LIBRARY PTScotch_ERR_LIBRARY PTScotch_SCOTCH_LIBRARY PTScotch_HAS_IDX64)
