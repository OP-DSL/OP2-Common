# FindParMETIS.cmake
#
# Locates ParMETIS and its METIS dependency.
# Sets imported target ParMETIS::ParMETIS on success.
#
# Hints (cache and env, checked in order):
#   ParMETIS: ParMETIS_ROOT (matches find_package(ParMETIS)), ParMETIS_DIR, PARMETIS_DIR
#   METIS:    METIS_ROOT, METIS_DIR (fall back to ParMETIS hints if not set)
#
# Validates that ParMETIS was built with 64-bit indices (sizeof(idx_t) == 8) as
# required by OP2 (idx_g_t is long long).  The check fails if a 32-bit METIS build is found.
#
# Cache variables (advanced, but set any of them by hand to point this module
# at an install whose layout the search below doesn't guess correctly):
#   ParMETIS_INCLUDE_DIR - directory holding parmetis.h
#   ParMETIS_LIBRARY     - libparmetis
#   METIS_INCLUDE_DIR    - directory holding metis.h
#   METIS_LIBRARY        - libmetis

include(FindPackageHandleStandardArgs)

set(_parmetis_hints
    "${ParMETIS_ROOT}"      "$ENV{ParMETIS_ROOT}"
    "${ParMETIS_DIR}"       "$ENV{ParMETIS_DIR}"
    "${PARMETIS_DIR}"       "$ENV{PARMETIS_DIR}"
)
set(_metis_hints
    "${METIS_ROOT}"         "$ENV{METIS_ROOT}"
    "${METIS_DIR}"          "$ENV{METIS_DIR}"
    ${_parmetis_hints}
)

find_path(ParMETIS_INCLUDE_DIR NAMES parmetis.h
    HINTS ${_parmetis_hints} PATH_SUFFIXES include NO_DEFAULT_PATH)
if(NOT ParMETIS_INCLUDE_DIR)
    find_path(ParMETIS_INCLUDE_DIR NAMES parmetis.h PATH_SUFFIXES include)
endif()

find_library(ParMETIS_LIBRARY NAMES parmetis
    HINTS ${_parmetis_hints} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
if(NOT ParMETIS_LIBRARY)
    find_library(ParMETIS_LIBRARY NAMES parmetis PATH_SUFFIXES lib lib64)
endif()

find_path(METIS_INCLUDE_DIR NAMES metis.h
    HINTS ${_metis_hints} PATH_SUFFIXES include NO_DEFAULT_PATH)
if(NOT METIS_INCLUDE_DIR)
    find_path(METIS_INCLUDE_DIR NAMES metis.h PATH_SUFFIXES include)
endif()

find_library(METIS_LIBRARY NAMES metis
    HINTS ${_metis_hints} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
if(NOT METIS_LIBRARY)
    find_library(METIS_LIBRARY NAMES metis PATH_SUFFIXES lib lib64)
endif()

# Validate 64-bit index width via a compile-time sizeof check on idx_t.
# parmetis.h includes metis.h which defines idx_t; it also includes mpi.h.
# OP2's idx_g_t is long long; a 32-bit partitioner build would silently truncate global indices.
if(ParMETIS_INCLUDE_DIR AND METIS_INCLUDE_DIR AND ParMETIS_LIBRARY AND METIS_LIBRARY)
    set(_parmetis_mpi_incs "")
    if(MPI_C_INCLUDE_DIRS)
        set(_parmetis_mpi_incs "${MPI_C_INCLUDE_DIRS}")
    elseif(MPI_CXX_INCLUDE_DIRS)
        set(_parmetis_mpi_incs "${MPI_CXX_INCLUDE_DIRS}")
    else()
        find_path(ParMETIS_MPI_INCLUDE_DIR NAMES mpi.h)
        if(ParMETIS_MPI_INCLUDE_DIR)
            set(_parmetis_mpi_incs "${ParMETIS_MPI_INCLUDE_DIR}")
        endif()
    endif()

    # Probe in whichever language the project has enabled - see the equivalent
    # note in FindPTScotch.cmake.  parmetis.h compiles cleanly as either.
    if(CMAKE_C_COMPILER_LOADED)
        set(_parmetis_probe_lang C)
    else()
        set(_parmetis_probe_lang CXX)
    endif()

    include(CheckSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_INCLUDES
        "${ParMETIS_INCLUDE_DIR}" "${METIS_INCLUDE_DIR}" ${_parmetis_mpi_incs})
    # The skip macros must precede <mpi.h>: as C++ an unguarded include drags
    # in the deprecated MPI C++ bindings, which this probe would then have to
    # link against (check_source_compiles links as well as compiles).
    check_source_compiles(${_parmetis_probe_lang} "
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#define MPI_NO_CPPBIND 1
#include <stddef.h>
#include <mpi.h>
#include <parmetis.h>
typedef char _parmetis_idx64_check[sizeof(idx_t) == 8 ? 1 : -1];
int main(void) { return 0; }
" ParMETIS_HAS_IDX64)
    cmake_pop_check_state()
    unset(_parmetis_probe_lang)
endif()

if(ParMETIS_INCLUDE_DIR AND ParMETIS_LIBRARY AND METIS_LIBRARY)
    if(NOT TARGET ParMETIS::ParMETIS)
        add_library(ParMETIS::ParMETIS UNKNOWN IMPORTED)
        set_target_properties(ParMETIS::ParMETIS PROPERTIES
            IMPORTED_LOCATION             "${ParMETIS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ParMETIS_INCLUDE_DIR}")
        set_property(TARGET ParMETIS::ParMETIS APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES "${METIS_LIBRARY}")
        if(METIS_INCLUDE_DIR)
            set_property(TARGET ParMETIS::ParMETIS APPEND PROPERTY
                INTERFACE_INCLUDE_DIRECTORIES "${METIS_INCLUDE_DIR}")
        endif()
    endif()
endif()

find_package_handle_standard_args(ParMETIS
    REQUIRED_VARS ParMETIS_INCLUDE_DIR ParMETIS_LIBRARY METIS_LIBRARY ParMETIS_HAS_IDX64
    FAIL_MESSAGE  "ParMETIS not found or built with 32-bit indices (set ParMETIS_ROOT and METIS_ROOT and rebuild with --64bit-indices)")

mark_as_advanced(ParMETIS_INCLUDE_DIR ParMETIS_LIBRARY METIS_INCLUDE_DIR METIS_LIBRARY ParMETIS_HAS_IDX64)
