# FindKaHIP.cmake
#
# Locates the parallel KaHIP interface (ParHIP) required for MPI mesh
# partitioning.
# Results come back in the cache variables below, not an imported target: OP2
# links the partitioners by path so install(EXPORT) pins the exact artefacts,
# and this module is not installed for a consumer to load. An imported target
# here would be recorded in OP2Targets.cmake and fail every consumer with
# "target not found".
#
# OP2 only uses KaHIP in MPI contexts (op_mpi_part_core.cpp calls
# ParHIPPartitionKWay), so only the parallel library is needed.
#
# Hints (cache and env):
#   KaHIP_ROOT - searched automatically by find_* inside a module loaded by
#                find_package(KaHIP), so it needs no explicit hint
#   KAHIP_DIR  - Spack / module-system convention
#
# Installed layout produced by scripts/setup_deps.sh:
#   include/kaHIP_interface.h
#   include/parhip_interface.h
#   lib/libkahip_static.a
#   lib/libparhip_interface_static.a
#   lib/libmodified_kahip_interface.a   (sequential internals used by parhip)
#
# Cache variables (advanced, but set any of them by hand to point this module
# at an install whose layout the search below doesn't guess correctly):
#   KaHIP_INCLUDE_DIR        - directory holding parhip_interface.h
#   KaHIP_ParHIP_LIBRARY     - libparhip_interface
#   KaHIP_MODIFIED_LIBRARY   - libmodified_kahip_interface

include(FindPackageHandleStandardArgs)

set(_kahip_hints
    "${KAHIP_DIR}"      "$ENV{KAHIP_DIR}"
)

find_path(KaHIP_INCLUDE_DIR NAMES parhip_interface.h
    HINTS ${_kahip_hints} PATH_SUFFIXES include)

find_library(KaHIP_ParHIP_LIBRARY NAMES parhip_interface_static parhip_interface
    HINTS ${_kahip_hints} PATH_SUFFIXES lib lib64)

# Sequential internals that parhip_interface_static depends on at link time.
find_library(KaHIP_MODIFIED_LIBRARY NAMES modified_kahip_interface
    HINTS ${_kahip_hints} PATH_SUFFIXES lib lib64)

find_package_handle_standard_args(KaHIP
    REQUIRED_VARS KaHIP_INCLUDE_DIR KaHIP_ParHIP_LIBRARY
    FAIL_MESSAGE  "KaHIP (ParHIP) not found - set KaHIP_ROOT to the install prefix")

mark_as_advanced(KaHIP_INCLUDE_DIR KaHIP_ParHIP_LIBRARY KaHIP_MODIFIED_LIBRARY)
