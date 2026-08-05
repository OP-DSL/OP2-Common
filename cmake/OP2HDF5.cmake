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

find_package(HDF5 CONFIG QUIET COMPONENTS C)

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

# Target name varies by HDF5 version/build (1.14 uses "hdf5-static", others
# "hdf5::hdf5-static"); pick the first available.
foreach(_cand
        hdf5::hdf5-static hdf5::hdf5-shared hdf5::hdf5
        hdf5-static       hdf5-shared)
    if(TARGET ${_cand})
        set(_op2_hdf5_target "${_cand}")
        break()
    endif()
endforeach()
foreach(_cand
        hdf5::hdf5_hl-static hdf5::hdf5_hl-shared hdf5::hdf5_hl
        hdf5_hl-static       hdf5_hl-shared)
    if(TARGET ${_cand})
        set(_op2_hdf5_hl_target "${_cand}")
        break()
    endif()
endforeach()

if(NOT _op2_hdf5_target)
    message(WARNING
        "OP2: HDF5 config found but no usable target exported "
        "(looked for hdf5::hdf5-static/-shared/hdf5::hdf5) - "
        "HDF5-dependent libraries will not be built")
    set(HDF5_FOUND FALSE)
    return()
endif()

set(_op2_hdf5_link_targets "${_op2_hdf5_target}")
if(_op2_hdf5_hl_target)
    list(APPEND _op2_hdf5_link_targets "${_op2_hdf5_hl_target}")
endif()

message(STATUS
    "OP2: HDF5         = found (${HDF5_VERSION}, parallel=${OP2_HDF5_IS_PARALLEL}, target=${_op2_hdf5_target})")
