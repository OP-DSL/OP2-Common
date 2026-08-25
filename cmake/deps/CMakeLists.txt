cmake_minimum_required(VERSION 3.24)
project(OP2Deps LANGUAGES C CXX)

include(ExternalProject)
include(CheckLanguage)

# ---------------------------------------------------------------------------
# Parallel jobs - the bash wrapper forwards -j N as -DPARALLEL_LEVEL=N
# ---------------------------------------------------------------------------
set(PARALLEL_LEVEL "4" CACHE STRING "Number of parallel jobs for external project builds")

# ---------------------------------------------------------------------------
# GNU Make - required for autoconf/makefile-based external projects
# ---------------------------------------------------------------------------
find_program(MAKE_EXECUTABLE NAMES make gmake REQUIRED
    DOC "GNU Make (required by HDF5 and PT-Scotch builds)")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(OP2_DEPS_HDF5     "Build HDF5 serial and parallel libraries" ON)
option(OP2_DEPS_PARMETIS "Build ParMETIS + METIS (64-bit indices)"  ON)
option(OP2_DEPS_PTSCOTCH "Build PT-Scotch (64-bit indices)"         ON)
option(OP2_DEPS_KAHIP    "Build KaHIP with ParHIP (MPI)"            ON)

# Pinned version and the SHA-256 of that version's tarball, each checked
# against a fresh upstream download. Bump the pair together.
#
# PT-Scotch and KaHIP are fetched as forge-generated archives rather than
# uploaded release assets, and those are not contractually byte-stable. A
# mismatch there may mean the forge regenerated the tarball rather than
# anything sinister - check, then set the SHA256 cache entry below.
set(_op2_deps_hdf5_version     "1.14.4-3")
set(_op2_deps_hdf5_sha256      "019ac451d9e1cf89c0482ba2a06f07a46166caf23f60fea5ef3c37724a318e03")
set(_op2_deps_parmetis_version "4.0.3")
set(_op2_deps_parmetis_sha256  "f2d9a231b7cf97f1fee6e8c9663113ebf6c240d407d3c118c55b3633d6be6e5f")
set(_op2_deps_ptscotch_version "7.0.4")
set(_op2_deps_ptscotch_sha256  "8ef4719d6a3356e9c4ca7fefd7e2ac40deb69779a5c116f44da75d13b3d2c2c3")
set(_op2_deps_kahip_version    "3.25")
set(_op2_deps_kahip_sha256     "3abad20158887b585b4d4792c35fa9023b7fd634b2bc4494334a3951b69f4842")

set(OP2_DEPS_HDF5_VERSION     "${_op2_deps_hdf5_version}"     CACHE STRING "HDF5 version to fetch and build")
set(OP2_DEPS_PARMETIS_VERSION "${_op2_deps_parmetis_version}" CACHE STRING "ParMETIS version to fetch and build")
set(OP2_DEPS_PTSCOTCH_VERSION "${_op2_deps_ptscotch_version}" CACHE STRING "PT-Scotch version to fetch and build")
set(OP2_DEPS_KAHIP_VERSION    "${_op2_deps_kahip_version}"    CACHE STRING "KaHIP version to fetch and build")

foreach(_dep HDF5 PARMETIS PTSCOTCH KAHIP)
    set(OP2_DEPS_${_dep}_SHA256 "" CACHE STRING
        "SHA-256 of the tarball, needed only when overriding OP2_DEPS_${_dep}_VERSION")
    mark_as_advanced(OP2_DEPS_${_dep}_SHA256)
endforeach()

# Yields the URL_HASH argument pair for ExternalProject_Add, or nothing.
# Overriding a version without supplying its checksum turns verification off
# for that download rather than failing on a checksum meant for another
# tarball - loudly, because an unverified download is worth noticing.
function(_op2_deps_url_hash dep out_var)
    string(TOUPPER "${dep}" _DEP)

    if(OP2_DEPS_${_DEP}_SHA256)
        set(_hash "${OP2_DEPS_${_DEP}_SHA256}")
    elseif(OP2_DEPS_${_DEP}_VERSION STREQUAL "${_op2_deps_${dep}_version}")
        set(_hash "${_op2_deps_${dep}_sha256}")
    else()
        message(WARNING
            "OP2Deps: ${dep} ${OP2_DEPS_${_DEP}_VERSION} has no checksum - the "
            "pinned one belongs to ${_op2_deps_${dep}_version}. This download "
            "will NOT be verified. Set OP2_DEPS_${_DEP}_SHA256 to restore it.")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    set(${out_var} URL_HASH "SHA256=${_hash}" PARENT_SCOPE)
endfunction()

# HDF5 variant: parallel (with MPI) or serial.
# AUTO -> ON if MPI found, OFF otherwise.  Explicit ON errors out if MPI is missing.
set(OP2_DEPS_HDF5_PARALLEL "AUTO" CACHE STRING
    "Build HDF5 with MPI support (ON/OFF/AUTO; AUTO=ON if MPI found)")
set_property(CACHE OP2_DEPS_HDF5_PARALLEL PROPERTY STRINGS AUTO ON OFF)

# ---------------------------------------------------------------------------
# Install prefix - defaults to <repo_root>/deps
# ---------------------------------------------------------------------------
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    get_filename_component(_default_prefix "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    set(CMAKE_INSTALL_PREFIX "${_default_prefix}/deps"
        CACHE PATH "Root install directory for OP2 dependencies" FORCE)
endif()
set(_prefix "${CMAKE_INSTALL_PREFIX}")

# ---------------------------------------------------------------------------
# Optional Fortran (for HDF5 Fortran bindings)
# ---------------------------------------------------------------------------
check_language(Fortran)
if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
endif()

# ---------------------------------------------------------------------------
# MPI - required for all parallel deps
# ---------------------------------------------------------------------------
find_package(MPI COMPONENTS C CXX REQUIRED)
if(NOT MPI_C_COMPILER)
    message(FATAL_ERROR
        "MPI C wrapper compiler not found.\n"
        "Set -DMPI_C_COMPILER=<path-to-mpicc> on the cmake command line, "
        "or ensure mpicc is on PATH.")
endif()
if(NOT MPI_CXX_COMPILER)
    message(FATAL_ERROR
        "MPI C++ wrapper compiler not found.\n"
        "Set -DMPI_CXX_COMPILER=<path-to-mpicxx> on the cmake command line, "
        "or ensure mpicxx is on PATH.")
endif()
if(CMAKE_Fortran_COMPILER)
    find_package(MPI COMPONENTS Fortran QUIET)
endif()

# ---------------------------------------------------------------------------
# Status report
# ---------------------------------------------------------------------------
message(STATUS "OP2Deps: install prefix = ${_prefix}")
message(STATUS "OP2Deps: C compiler     = ${CMAKE_C_COMPILER}")
message(STATUS "OP2Deps: C++ compiler   = ${CMAKE_CXX_COMPILER}")
if(CMAKE_Fortran_COMPILER)
    message(STATUS "OP2Deps: Fortran        = ${CMAKE_Fortran_COMPILER}")
else()
    message(STATUS "OP2Deps: Fortran        = (none) - HDF5 built without Fortran bindings")
endif()
message(STATUS "OP2Deps: MPI C          = ${MPI_C_COMPILER}")
message(STATUS "OP2Deps: MPI C++        = ${MPI_CXX_COMPILER}")
if(MPI_Fortran_FOUND)
    message(STATUS "OP2Deps: MPI Fortran    = ${MPI_Fortran_COMPILER}")
endif()
message(STATUS "OP2Deps: parallel jobs  = ${PARALLEL_LEVEL}")
message(STATUS "OP2Deps: HDF5           = ${OP2_DEPS_HDF5}  (${OP2_DEPS_HDF5_VERSION}, parallel=${OP2_DEPS_HDF5_PARALLEL})")
message(STATUS "OP2Deps: ParMETIS       = ${OP2_DEPS_PARMETIS}  (${OP2_DEPS_PARMETIS_VERSION})")
message(STATUS "OP2Deps: PT-Scotch      = ${OP2_DEPS_PTSCOTCH}  (${OP2_DEPS_PTSCOTCH_VERSION})")
message(STATUS "OP2Deps: KaHIP          = ${OP2_DEPS_KAHIP}  (${OP2_DEPS_KAHIP_VERSION})")

# ---------------------------------------------------------------------------
# Forward CMAKE_TOOLCHAIN_FILE to cmake-based sub-builds so cross-compilation
# stays consistent between the deps and the main OP2 build.
# ---------------------------------------------------------------------------
set(_toolchain_args "")
if(CMAKE_TOOLCHAIN_FILE)
    get_filename_component(_toolchain_abs "${CMAKE_TOOLCHAIN_FILE}" ABSOLUTE)
    list(APPEND _toolchain_args "-DCMAKE_TOOLCHAIN_FILE=${_toolchain_abs}")
    message(STATUS "OP2Deps: toolchain file = ${_toolchain_abs}")
endif()

set(_all_dep_targets "")

# ---------------------------------------------------------------------------
# HDF5 (single install; parallel if MPI available, otherwise serial)
# ---------------------------------------------------------------------------
if(OP2_DEPS_HDF5)
    # Resolve AUTO / ON / OFF for the parallel flag
    if(OP2_DEPS_HDF5_PARALLEL STREQUAL "AUTO")
        if(MPI_C_COMPILER)
            set(_hdf5_parallel TRUE)
        else()
            set(_hdf5_parallel FALSE)
        endif()
    elseif(OP2_DEPS_HDF5_PARALLEL)
        if(NOT MPI_C_COMPILER)
            message(FATAL_ERROR
                "OP2_DEPS_HDF5_PARALLEL=ON but no MPI C wrapper found - "
                "set MPI_C_COMPILER or use AUTO/OFF.")
        endif()
        set(_hdf5_parallel TRUE)
    else()
        set(_hdf5_parallel FALSE)
    endif()

    # Parse "1.14.4-3" into tag components: hdf5_1.14.4.3
    if(NOT OP2_DEPS_HDF5_VERSION MATCHES "^([0-9]+\\.[0-9]+\\.[0-9]+)-([0-9]+)$")
        message(FATAL_ERROR
            "OP2_DEPS_HDF5_VERSION must be in X.Y.Z-N form, got '${OP2_DEPS_HDF5_VERSION}'")
    endif()
    set(_hdf5_xyz "${CMAKE_MATCH_1}")
    set(_hdf5_n   "${CMAKE_MATCH_2}")
    set(_hdf5_tag "hdf5_${_hdf5_xyz}.${_hdf5_n}")
    set(_hdf5_url
        "https://github.com/HDFGroup/hdf5/releases/download/${_hdf5_tag}/hdf5-${OP2_DEPS_HDF5_VERSION}.tar.gz")

    # HDF5's CMake build produces hdf5-config.cmake, which OP2 discovers via
    # find_package(HDF5 CONFIG); the autoconf build ships no config file.
    if(_hdf5_parallel)
        set(_hdf5_c_compiler ${MPI_C_COMPILER})
        if(MPI_Fortran_FOUND)
            set(_hdf5_fortran_compiler ${MPI_Fortran_COMPILER})
        elseif(CMAKE_Fortran_COMPILER)
            set(_hdf5_fortran_compiler ${CMAKE_Fortran_COMPILER})
        else()
            set(_hdf5_fortran_compiler "")
        endif()
    else()
        set(_hdf5_c_compiler ${CMAKE_C_COMPILER})
        if(CMAKE_Fortran_COMPILER)
            set(_hdf5_fortran_compiler ${CMAKE_Fortran_COMPILER})
        else()
            set(_hdf5_fortran_compiler "")
        endif()
    endif()

    set(_hdf5_cmake_args
        ${_toolchain_args}
        -DCMAKE_C_COMPILER=${_hdf5_c_compiler}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DBUILD_STATIC_LIBS=ON
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_TESTING=OFF
        -DHDF5_BUILD_EXAMPLES=OFF
        -DHDF5_BUILD_TOOLS=OFF
        -DHDF5_BUILD_HL_LIB=ON
        -DHDF5_BUILD_CPP_LIB=OFF
    )
    if(_hdf5_parallel)
        list(APPEND _hdf5_cmake_args -DHDF5_ENABLE_PARALLEL=ON)
    else()
        list(APPEND _hdf5_cmake_args -DHDF5_ENABLE_PARALLEL=OFF)
    endif()
    if(_hdf5_fortran_compiler)
        list(APPEND _hdf5_cmake_args
            -DHDF5_BUILD_FORTRAN=ON
            -DCMAKE_Fortran_COMPILER=${_hdf5_fortran_compiler})
    else()
        list(APPEND _hdf5_cmake_args -DHDF5_BUILD_FORTRAN=OFF)
    endif()

    _op2_deps_url_hash(hdf5 _hdf5_url_hash)
    ExternalProject_Add(hdf5
        URL            "${_hdf5_url}"
        ${_hdf5_url_hash}
        DOWNLOAD_DIR   "${CMAKE_BINARY_DIR}/downloads"
        SOURCE_DIR     "${CMAKE_BINARY_DIR}/hdf5-src"
        BINARY_DIR     "${CMAKE_BINARY_DIR}/hdf5-build"
        INSTALL_DIR    "${_prefix}/hdf5"
        CMAKE_ARGS     ${_hdf5_cmake_args}
        BUILD_COMMAND
            ${CMAKE_COMMAND} --build <BINARY_DIR> --parallel ${PARALLEL_LEVEL}
        INSTALL_COMMAND
            ${CMAKE_COMMAND} --install <BINARY_DIR>
        LOG_CONFIGURE ON LOG_BUILD ON LOG_INSTALL ON
        LOG_OUTPUT_ON_FAILURE ON
    )
    list(APPEND _all_dep_targets hdf5)
endif()

# ---------------------------------------------------------------------------
# ParMETIS + METIS (always 64-bit)
# ---------------------------------------------------------------------------
if(OP2_DEPS_PARMETIS)
    set(_parmetis_url
        "https://ftp.mcs.anl.gov/pub/pdetools/spack-pkgs/parmetis-${OP2_DEPS_PARMETIS_VERSION}.tar.gz")
    set(_parmetis_src "${CMAKE_BINARY_DIR}/parmetis-src")
    set(_parmetis_bin "${CMAKE_BINARY_DIR}/parmetis-build")

    _op2_deps_url_hash(parmetis _parmetis_url_hash)
    ExternalProject_Add(parmetis
        URL            "${_parmetis_url}"
        ${_parmetis_url_hash}
        SOURCE_DIR     "${_parmetis_src}"
        BINARY_DIR     "${_parmetis_bin}"
        INSTALL_DIR    "${_prefix}/parmetis"
        PATCH_COMMAND
            ${CMAKE_COMMAND}
                -DSRC=${_parmetis_src}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/patch_parmetis.cmake"
        CMAKE_ARGS
            ${_toolchain_args}
            -DCMAKE_C_COMPILER=${MPI_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${MPI_CXX_COMPILER}
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DCMAKE_BUILD_TYPE=Release
            -DBUILD_SHARED_LIBS=OFF
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DMETIS_PATH=${_parmetis_src}/metis
            -DGKLIB_PATH=${_parmetis_src}/metis/GKlib
        BUILD_COMMAND
            ${CMAKE_COMMAND} --build ${_parmetis_bin} --parallel ${PARALLEL_LEVEL}
        INSTALL_COMMAND
            ${CMAKE_COMMAND} --install ${_parmetis_bin}
            COMMAND
                ${CMAKE_COMMAND}
                    -DSRC=${_parmetis_bin}
                    -DSOURCE_DIR=${_parmetis_src}
                    -DDST=${_prefix}/metis
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/install_metis.cmake"
        LOG_CONFIGURE ON LOG_BUILD ON LOG_INSTALL ON
        LOG_OUTPUT_ON_FAILURE ON
    )
    list(APPEND _all_dep_targets parmetis)
endif()

# ---------------------------------------------------------------------------
# PT-Scotch (always 64-bit via -DINTSIZE64 in Makefile.inc)
# ---------------------------------------------------------------------------
if(OP2_DEPS_PTSCOTCH)
    set(_ptscotch_url
        "https://gitlab.inria.fr/scotch/scotch/-/archive/v${OP2_DEPS_PTSCOTCH_VERSION}/scotch-v${OP2_DEPS_PTSCOTCH_VERSION}.tar.gz")
    set(_ptscotch_src "${CMAKE_BINARY_DIR}/ptscotch-src")

    # Configure Makefile.inc from template at cmake configure time
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/Makefile.inc.in"
        "${CMAKE_BINARY_DIR}/scotch_Makefile.inc"
        @ONLY
    )

    _op2_deps_url_hash(ptscotch _ptscotch_url_hash)
    ExternalProject_Add(ptscotch
        URL            "${_ptscotch_url}"
        ${_ptscotch_url_hash}
        SOURCE_DIR     "${_ptscotch_src}"
        BINARY_DIR     "${CMAKE_BINARY_DIR}/ptscotch-stamps"
        INSTALL_DIR    "${_prefix}/ptscotch"
        CONFIGURE_COMMAND
            ${CMAKE_COMMAND} -E copy
                "${CMAKE_BINARY_DIR}/scotch_Makefile.inc"
                ${_ptscotch_src}/src/Makefile.inc
        BUILD_COMMAND
            ${MAKE_EXECUTABLE} -j${PARALLEL_LEVEL} -C ${_ptscotch_src}/src ptscotch
        INSTALL_COMMAND
            ${MAKE_EXECUTABLE} -C ${_ptscotch_src}/src install prefix=<INSTALL_DIR>
        LOG_CONFIGURE ON LOG_BUILD ON LOG_INSTALL ON
        LOG_OUTPUT_ON_FAILURE ON
    )
    list(APPEND _all_dep_targets ptscotch)
endif()

# ---------------------------------------------------------------------------
# KaHIP (always built with MPI/ParHIP)
# ---------------------------------------------------------------------------
if(OP2_DEPS_KAHIP)
    set(_kahip_url
        "https://github.com/KaHIP/KaHIP/archive/refs/tags/v${OP2_DEPS_KAHIP_VERSION}.tar.gz")
    set(_kahip_src "${CMAKE_BINARY_DIR}/kahip-src")
    set(_kahip_bin "${CMAKE_BINARY_DIR}/kahip-build")

    _op2_deps_url_hash(kahip _kahip_url_hash)
    ExternalProject_Add(kahip
        URL            "${_kahip_url}"
        ${_kahip_url_hash}
        SOURCE_DIR     "${_kahip_src}"
        BINARY_DIR     "${_kahip_bin}"
        INSTALL_DIR    "${_prefix}/kahip"
        CMAKE_ARGS
            ${_toolchain_args}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DMPI_C_COMPILER=${MPI_C_COMPILER}
            -DMPI_CXX_COMPILER=${MPI_CXX_COMPILER}
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DNONATIVEOPTIMIZATIONS=ON
        BUILD_COMMAND
            ${CMAKE_COMMAND} --build ${_kahip_bin}
                --target kahip_static parhip_interface_static
                --parallel ${PARALLEL_LEVEL}
        INSTALL_COMMAND
            ${CMAKE_COMMAND}
                -DSRC=${_kahip_bin}
                -DSOURCE_DIR=${_kahip_src}
                -DDST=<INSTALL_DIR>
                -P "${CMAKE_CURRENT_SOURCE_DIR}/install_kahip.cmake"
        LOG_CONFIGURE ON LOG_BUILD ON LOG_INSTALL ON
        LOG_OUTPUT_ON_FAILURE ON
    )
    list(APPEND _all_dep_targets kahip)
endif()

# ---------------------------------------------------------------------------
# Generate op2-deps.cmake - a CMake cache initializer for the main OP2 build
# (-C deps/op2-deps.cmake sets all *_ROOT and compiler paths in one shot)
# ---------------------------------------------------------------------------
set(_cache_lines "# Auto-generated by cmake/deps - pass to the OP2 cmake with -C <this-file>\n")
string(APPEND _cache_lines
    "set(CMAKE_C_COMPILER   \"${CMAKE_C_COMPILER}\"   CACHE FILEPATH \"\" FORCE)\n"
    "set(CMAKE_CXX_COMPILER \"${CMAKE_CXX_COMPILER}\" CACHE FILEPATH \"\" FORCE)\n"
)
if(CMAKE_Fortran_COMPILER)
    string(APPEND _cache_lines
        "set(CMAKE_Fortran_COMPILER \"${CMAKE_Fortran_COMPILER}\" CACHE FILEPATH \"\" FORCE)\n")
endif()
if(OP2_DEPS_HDF5)
    string(APPEND _cache_lines
        "set(HDF5_ROOT \"${_prefix}/hdf5\" CACHE PATH \"\" FORCE)\n")
endif()
if(OP2_DEPS_PARMETIS)
    string(APPEND _cache_lines
        "set(ParMETIS_ROOT \"${_prefix}/parmetis\" CACHE PATH \"\" FORCE)\n"
        "set(METIS_ROOT    \"${_prefix}/metis\"    CACHE PATH \"\" FORCE)\n")
endif()
if(OP2_DEPS_PTSCOTCH)
    string(APPEND _cache_lines
        "set(PTScotch_ROOT \"${_prefix}/ptscotch\" CACHE PATH \"\" FORCE)\n")
endif()
if(OP2_DEPS_KAHIP)
    string(APPEND _cache_lines
        "set(KaHIP_ROOT    \"${_prefix}/kahip\"    CACHE PATH \"\" FORCE)\n")
endif()

file(WRITE "${CMAKE_BINARY_DIR}/op2-deps.cmake" "${_cache_lines}")

# Deliberately not DEPENDS on the dep targets: this file's content is settled
# at configure time, and gating it on every dependency building meant one
# failure left the user with the others installed but nothing naming them. A
# *_ROOT pointing at a prefix that was never populated costs nothing - OP2's
# find_package for that dependency reports it as not found and the build drops
# those variants, which is the right outcome.
if(_all_dep_targets)
    add_custom_target(op2_deps_cache ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_prefix}"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_BINARY_DIR}/op2-deps.cmake"
            "${_prefix}/op2-deps.cmake"
        COMMENT "Writing ${_prefix}/op2-deps.cmake"
        VERBATIM
    )
endif()
