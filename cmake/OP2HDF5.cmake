# OP2HDF5.cmake - discover HDF5 and pick the right imported target.
#
# Included from the top-level CMakeLists.txt.  Reads OP2_ENABLE_HDF5, sets:
#   HDF5_FOUND              - the standard find_package output
#   OP2_HDF5_IS_PARALLEL    - TRUE when the HDF5 install was built with MPI
#   _op2_hdf5_target        - the (unnamespaced/namespaced) base HDF5 target
#   _op2_hdf5_hl_target     - the equivalent HL target (may be empty)
#   _op2_hdf5_link_targets  - [base + HL] concatenated, for target_link_libraries

set(OP2_HDF5_IS_PARALLEL FALSE)
set(_op2_hdf5_target "")
set(_op2_hdf5_hl_target "")
set(_op2_hdf5_link_targets "")

if(NOT OP2_ENABLE_HDF5)
    return()
endif()

# Listing the flavours as components is what makes the config report the target
# names it exports, in HDF5_<COMPONENT>_<FLAVOUR>_LIBRARY; asking for one the
# install lacks is not an error, it just reports FOUND 0.
find_package(HDF5 CONFIG QUIET COMPONENTS C HL static shared)

if(NOT HDF5_FOUND)
    message(STATUS
        "OP2: HDF5         = not found (set HDF5_ROOT or install HDF5 with a CMake config file)")
    return()
endif()

# HDF5's config file sets HDF5_ENABLE_PARALLEL; CMake's built-in FindHDF5
# sets HDF5_IS_PARALLEL.  Accept either.
if(DEFINED HDF5_ENABLE_PARALLEL)
    set(OP2_HDF5_IS_PARALLEL "${HDF5_ENABLE_PARALLEL}")
else()
    set(OP2_HDF5_IS_PARALLEL "${HDF5_IS_PARALLEL}")
endif()

# Prefer the flavour matching how OP2 itself is being built. A static HDF5 is
# absorbed into each libop2_*.so that links it - eight of them - so a process
# loading two of those libraries gets two private HDF5 instances with separate
# global state, where one shared HDF5 is shared by all of them. Only a
# preference: an install offering just one flavour still works.
if(BUILD_SHARED_LIBS)
    set(_op2_hdf5_flavours shared static)
else()
    set(_op2_hdf5_flavours static shared)
endif()

foreach(_flavour IN LISTS _op2_hdf5_flavours)
    if(NOT HDF5_${_flavour}_C_FOUND)
        continue()
    endif()
    string(TOUPPER "${_flavour}" _f)
    set(_op2_hdf5_target "${HDF5_C_${_f}_LIBRARY}")
    if(HDF5_${_flavour}_HL_FOUND)
        set(_op2_hdf5_hl_target "${HDF5_HL_${_f}_LIBRARY}")
    endif()
    break()
endforeach()

if(NOT _op2_hdf5_target)
    message(WARNING
        "OP2: HDF5 config found but reported neither a static nor a shared C "
        "library - HDF5-dependent libraries will not be built")
    set(HDF5_FOUND FALSE)
    return()
endif()

set(_op2_hdf5_link_targets "${_op2_hdf5_target}")
if(_op2_hdf5_hl_target)
    list(APPEND _op2_hdf5_link_targets "${_op2_hdf5_hl_target}")
endif()

message(STATUS
    "OP2: HDF5         = found (${HDF5_VERSION}, parallel=${OP2_HDF5_IS_PARALLEL}, target=${_op2_hdf5_target})")
