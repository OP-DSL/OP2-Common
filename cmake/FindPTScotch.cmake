# FindPTScotch.cmake
#
# Locates PT-Scotch (parallel graph partitioning library).
# Results come back in the cache variables below, not an imported target: OP2
# links the partitioners by path so install(EXPORT) pins the exact artefacts,
# and this module is not installed for a consumer to load. An imported target
# here would be recorded in OP2Targets.cmake and fail every consumer with
# "target not found".
#
# Hints (cache and env):
#   PTScotch_ROOT   - searched automatically by find_* inside a module loaded
#                     by find_package(PTScotch), so it needs no explicit hint
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
    "${SCOTCH_DIR}"         "$ENV{SCOTCH_DIR}"
    "${PTSCOTCH_DIR}"       "$ENV{PTSCOTCH_DIR}"
)

find_path(PTScotch_INCLUDE_DIR NAMES ptscotch.h scotch.h
    HINTS ${_ptscotch_hints} PATH_SUFFIXES include)

find_library(PTScotch_LIBRARY NAMES ptscotch
    HINTS ${_ptscotch_hints} PATH_SUFFIXES lib lib64)

find_library(PTScotch_ERR_LIBRARY NAMES ptscotcherr
    HINTS ${_ptscotch_hints} PATH_SUFFIXES lib lib64)

# Scotch (serial) is needed at link time alongside PT-Scotch
find_library(PTScotch_SCOTCH_LIBRARY NAMES scotch
    HINTS ${_ptscotch_hints} PATH_SUFFIXES lib lib64)

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

# No zlib among the libraries above: libscotch references gz* only from its
# compressed-file members, and OP2 builds graphs in memory - it calls no
# SCOTCH_*Load/*Save - so those members are never pulled out of the archive.

find_package_handle_standard_args(PTScotch
    REQUIRED_VARS PTScotch_INCLUDE_DIR PTScotch_LIBRARY PTScotch_HAS_IDX64
    FAIL_MESSAGE  "PT-Scotch not found or built with 32-bit indices (set PTScotch_ROOT and rebuild with --64bit-indices)")

mark_as_advanced(PTScotch_INCLUDE_DIR PTScotch_LIBRARY PTScotch_ERR_LIBRARY PTScotch_SCOTCH_LIBRARY PTScotch_HAS_IDX64)
