# FindKaHIP.cmake
#
# Locates the parallel KaHIP interface (ParHIP) required for MPI mesh
# partitioning.  Sets the imported target KaHIP::ParHIP on success.
#
# OP2 only uses KaHIP in MPI contexts (op_mpi_part_core.cpp calls
# ParHIPPartitionKWay), so only the parallel library is needed.
#
# Hints (cache and env, checked in order):
#   KaHIP_ROOT - matches find_package(KaHIP)
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
    "${KaHIP_ROOT}"     "$ENV{KaHIP_ROOT}"
    "${KAHIP_DIR}"      "$ENV{KAHIP_DIR}"
)

find_path(KaHIP_INCLUDE_DIR NAMES parhip_interface.h
    HINTS ${_kahip_hints} PATH_SUFFIXES include NO_DEFAULT_PATH)
if(NOT KaHIP_INCLUDE_DIR)
    find_path(KaHIP_INCLUDE_DIR NAMES parhip_interface.h PATH_SUFFIXES include)
endif()

find_library(KaHIP_ParHIP_LIBRARY NAMES parhip_interface_static parhip_interface
    HINTS ${_kahip_hints} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
if(NOT KaHIP_ParHIP_LIBRARY)
    find_library(KaHIP_ParHIP_LIBRARY NAMES parhip_interface_static parhip_interface
        PATH_SUFFIXES lib lib64)
endif()

# Sequential internals that parhip_interface_static depends on at link time.
find_library(KaHIP_MODIFIED_LIBRARY NAMES modified_kahip_interface
    HINTS ${_kahip_hints} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)

if(KaHIP_INCLUDE_DIR AND KaHIP_ParHIP_LIBRARY)
    if(NOT TARGET KaHIP::ParHIP)
        add_library(KaHIP::ParHIP UNKNOWN IMPORTED)
        set_target_properties(KaHIP::ParHIP PROPERTIES
            IMPORTED_LOCATION             "${KaHIP_ParHIP_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${KaHIP_INCLUDE_DIR}")
        if(KaHIP_MODIFIED_LIBRARY)
            set_property(TARGET KaHIP::ParHIP APPEND PROPERTY
                INTERFACE_LINK_LIBRARIES "${KaHIP_MODIFIED_LIBRARY}")
        endif()
    endif()
endif()

find_package_handle_standard_args(KaHIP
    REQUIRED_VARS KaHIP_INCLUDE_DIR KaHIP_ParHIP_LIBRARY
    FAIL_MESSAGE  "KaHIP (ParHIP) not found - set KaHIP_ROOT to the install prefix")

mark_as_advanced(KaHIP_INCLUDE_DIR KaHIP_ParHIP_LIBRARY KaHIP_MODIFIED_LIBRARY)
