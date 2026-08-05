cmake_minimum_required(VERSION 3.26)

project(OP2 VERSION 1.2.0 LANGUAGES C CXX)

# 3.26: first CMake release whose NVIDIA-CUDA module has a C++20 flag.

# ---------------------------------------------------------------------------
# Module path - must be set before any find_package calls
# ---------------------------------------------------------------------------
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Must precede add_subdirectory(op2): its targets bake
# CMAKE_INSTALL_INCLUDEDIR into their INSTALL_INTERFACE.
include(GNUInstallDirs)

# Imported targets default to directory scope; without this,
# op2_add_app_variants()'s if(TARGET) guards can't see them when OP2 is
# consumed via add_subdirectory().
set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL TRUE)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(OP2_ENABLE_MPI        "Build MPI-parallel library variants"          ON)
option(OP2_ENABLE_CUDA       "Build CUDA GPU library variants"              ON)
option(OP2_ENABLE_HIP        "Build HIP GPU library variants"               ON)
option(OP2_ENABLE_OPENMP     "Build OpenMP shared-memory variants"          ON)
option(OP2_ENABLE_HDF5       "Build HDF5 I/O library variants"              ON)
option(OP2_ENABLE_FORTRAN    "Build Fortran binding libraries"              ON)
option(OP2_ENABLE_APPS       "Build the example apps under apps/"           OFF)
option(OP2_ENABLE_TESTS      "Build the functional unit tests under tests/" OFF)
option(OP2_ENABLE_TRANSLATOR "Enable the OP2 translator (needs Python 3)"    ON)

# OFF by default as a subproject, on the assumption the parent owns packaging.
option(OP2_ENABLE_INSTALL    "Emit install() rules" "${PROJECT_IS_TOP_LEVEL}")

# Named to match each package's find_package(<PackageName>) spelling exactly.
set(HDF5_ROOT     "" CACHE PATH "Install prefix of HDF5 (with CMake config file)")
set(ParMETIS_ROOT "" CACHE PATH "Install prefix of ParMETIS")
set(METIS_ROOT    "" CACHE PATH "Install prefix of METIS (required by ParMETIS)")
set(PTScotch_ROOT "" CACHE PATH "Install prefix of PT-Scotch")
set(KaHIP_ROOT    "" CACHE PATH "Install prefix of KaHIP")

# Convenience root: fills in any *_ROOT above still left empty, from the layout
# scripts/setup_deps.sh produces.
set(OP2_DEPS_ROOT "" CACHE PATH
    "Root directory of OP2 dependencies built by setup_deps.sh / cmake/deps")
if(OP2_DEPS_ROOT)
    set(_op2_dep_vars    HDF5_ROOT ParMETIS_ROOT METIS_ROOT PTScotch_ROOT KaHIP_ROOT)
    set(_op2_dep_subdirs hdf5      parmetis      metis      ptscotch      kahip)
    foreach(_var _subdir IN ZIP_LISTS _op2_dep_vars _op2_dep_subdirs)
        if(NOT ${_var})
            # Carry the original help text over, rather than blanking it.
            get_property(_doc CACHE ${_var} PROPERTY HELPSTRING)
            set(${_var} "${OP2_DEPS_ROOT}/${_subdir}" CACHE PATH "${_doc}" FORCE)
        endif()
    endforeach()
    unset(_op2_dep_vars)
    unset(_op2_dep_subdirs)
endif()

# ---------------------------------------------------------------------------
# Compiler status
# ---------------------------------------------------------------------------
message(STATUS "OP2: C compiler   = ${CMAKE_C_COMPILER}")
message(STATUS "OP2: C++ compiler = ${CMAKE_CXX_COMPILER}")

# ---------------------------------------------------------------------------
# Optional language enablement - soft-fail if the compiler is absent
# ---------------------------------------------------------------------------
include(CheckLanguage)

if(OP2_ENABLE_FORTRAN)
    check_language(Fortran)
    if(CMAKE_Fortran_COMPILER)
        enable_language(Fortran)

        # gfortran/classic ifort pass a CONTAINS-internal kernel via a stack
        # trampoline, which needs an executable stack; this only silences the
        # linker's (correct) warning about that, not the requirement itself.
        include(CheckLinkerFlag)
        check_linker_flag(Fortran "-Wl,--no-warn-execstack" OP2_LINKER_HAS_NO_WARN_EXECSTACK)
    else()
        message(STATUS "OP2: Fortran compiler not found - Fortran libraries will not be built")
    endif()
endif()

if(CMAKE_Fortran_COMPILER)
    message(STATUS "OP2: Fortran      = ${CMAKE_Fortran_COMPILER}")
endif()

if(OP2_ENABLE_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        # Ampere/Hopper default: atomicAdd(double) needs sm_60+. Must precede
        # enable_language(CUDA) (CMP0104). Override with
        # -DCMAKE_CUDA_ARCHITECTURES=... or "native".
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
            set(CMAKE_CUDA_ARCHITECTURES 80 90)
        endif()
        enable_language(CUDA)
    else()
        message(STATUS "OP2: CUDA compiler not found - CUDA libraries will not be built")
    endif()
endif()

if(OP2_ENABLE_HIP)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        enable_language(HIP)
        # No default architecture - set CMAKE_HIP_ARCHITECTURES for your GPU
        # (e.g. gfx90a, gfx942).
    else()
        message(STATUS "OP2: HIP compiler not found - HIP libraries will not be built")
    endif()
endif()

# ---------------------------------------------------------------------------
# MPI
# ---------------------------------------------------------------------------
if(OP2_ENABLE_MPI)
    # So the status line below reports which MPI implementation was found,
    # not just the standard version it claims to implement.
    set(MPI_DETERMINE_LIBRARY_VERSION ON)

    # C and CXX only - OP2 uses no MPI Fortran bindings; an app doing
    # `use mpi` finds Fortran itself.
    find_package(MPI COMPONENTS C CXX QUIET)
    if(MPI_FOUND)
        set(_op2_mpi_ver "${MPI_VERSION}")
        if(NOT _op2_mpi_ver)
            set(_op2_mpi_ver "${MPI_C_VERSION}")
        endif()
        if(NOT _op2_mpi_ver)
            set(_op2_mpi_ver "unknown")
        endif()

        # Report the linked library path, not the wrapper's - OP2 links
        # MPI::MPI_* directly, not via mpicc/mpicxx.
        set(_op2_mpi_lib "unknown")
        if(MPI_C_LIBRARIES)
            list(GET MPI_C_LIBRARIES 0 _op2_mpi_lib)
        endif()

        set(_op2_mpi_impl "")
        if(MPI_C_LIBRARY_VERSION_STRING)
            string(REGEX MATCH "^[^\n]+" _op2_mpi_impl "${MPI_C_LIBRARY_VERSION_STRING}")
        endif()

        if(_op2_mpi_impl)
            message(STATUS "OP2: MPI          = found (${_op2_mpi_impl}; MPI ${_op2_mpi_ver}; ${_op2_mpi_lib})")
        else()
            message(STATUS "OP2: MPI          = found (MPI ${_op2_mpi_ver}; ${_op2_mpi_lib})")
        endif()
    else()
        message(STATUS "OP2: MPI          = not found - MPI variants will not be built")
    endif()
endif()

# ---------------------------------------------------------------------------
# OpenMP
# ---------------------------------------------------------------------------
if(OP2_ENABLE_OPENMP)
    # CXX and Fortran independently - OP2 never needs their conjunction, and
    # one call means no second one can clobber the first's result.
    set(_op2_openmp_components CXX)
    if(CMAKE_Fortran_COMPILER)
        list(APPEND _op2_openmp_components Fortran)
    endif()
    find_package(OpenMP COMPONENTS ${_op2_openmp_components} QUIET)
    unset(_op2_openmp_components)

    set(_op2_openmp_found "")
    if(OpenMP_CXX_FOUND)
        list(APPEND _op2_openmp_found "C++ v${OpenMP_CXX_VERSION}")
    endif()
    if(OpenMP_Fortran_FOUND)
        list(APPEND _op2_openmp_found "Fortran v${OpenMP_Fortran_VERSION}")
    endif()
    if(_op2_openmp_found)
        string(REPLACE ";" ", " _op2_openmp_found "${_op2_openmp_found}")
        message(STATUS "OP2: OpenMP       = found (${_op2_openmp_found})")
    else()
        message(STATUS "OP2: OpenMP       = not found - OpenMP variants will not be built")
    endif()
    unset(_op2_openmp_found)
endif()

# ---------------------------------------------------------------------------
# CUDA runtime library (needed by C++ TUs that include CUDA headers)
# ---------------------------------------------------------------------------
if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit REQUIRED QUIET)
endif()

# ---------------------------------------------------------------------------
# HIP runtime (host-side, for C++ TUs that include HIP headers)
# ---------------------------------------------------------------------------
if(CMAKE_HIP_COMPILER)
    find_package(hip REQUIRED CONFIG QUIET)
endif()

# ---------------------------------------------------------------------------
# HDF5 - single install via config-mode; HDF5_IS_PARALLEL gates MPI variants
# ---------------------------------------------------------------------------
include(OP2HDF5)

# ---------------------------------------------------------------------------
# Mesh partitioners (all optional, only attempted when MPI is available)
# ---------------------------------------------------------------------------
if(MPI_FOUND)
    find_package(PTScotch QUIET)
    if(PTScotch_FOUND)
        message(STATUS "OP2: PT-Scotch    = found (${PTScotch_LIBRARY})")
    else()
        message(STATUS "OP2: PT-Scotch    = not found (set PTScotch_ROOT or OP2_DEPS_ROOT)")
    endif()

    find_package(ParMETIS QUIET)
    if(ParMETIS_FOUND)
        message(STATUS "OP2: ParMETIS     = found (${ParMETIS_LIBRARY})")
    else()
        message(STATUS "OP2: ParMETIS     = not found (set ParMETIS_ROOT/METIS_ROOT or OP2_DEPS_ROOT)")
    endif()

    find_package(KaHIP QUIET)
    if(KaHIP_FOUND)
        message(STATUS "OP2: KaHIP        = found (${KaHIP_ParHIP_LIBRARY})")
    else()
        message(STATUS "OP2: KaHIP        = not found (set KaHIP_ROOT or OP2_DEPS_ROOT)")
    endif()
endif()

# ---------------------------------------------------------------------------
# Library targets
# ---------------------------------------------------------------------------
add_subdirectory(op2)

# ---------------------------------------------------------------------------
# Translator (Python) - find a suitable interpreter, install its source
# alongside the runtime libs
# ---------------------------------------------------------------------------
include(OP2Translator)

# ---------------------------------------------------------------------------
# App-build support (translator wrapper + variant executable helper)
# ---------------------------------------------------------------------------
# In-tree counterpart of what OP2Config.cmake sets downstream. CACHE INTERNAL
# so op2_add_app_variants() (called from inside a function()) sees it from
# any directory scope.
set(OP2_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/op2/include" CACHE INTERNAL "")
if(MPI_FOUND)
    set(OP2_HAS_MPI TRUE CACHE INTERNAL "")
else()
    set(OP2_HAS_MPI FALSE CACHE INTERNAL "")
endif()
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/OP2AppSupport.cmake")

if(OP2_ENABLE_APPS)
    add_subdirectory(apps)
endif()

if(OP2_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Configuration summary
# ---------------------------------------------------------------------------
include(OP2Summary)

# ---------------------------------------------------------------------------
# Install / export
# ---------------------------------------------------------------------------
if(NOT OP2_ENABLE_INSTALL)
    return()
endif()

include(CMakePackageConfigHelpers)

install(EXPORT OP2Targets
    FILE        OP2Targets.cmake
    NAMESPACE   op2::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/op2")

# PATH_VARS: OP2Config.cmake refers to these as @PACKAGE_<var>@, resolved
# against PACKAGE_PREFIX_DIR, so it stays correct for any install layout.
set(OP2_INSTALL_INCLUDEDIR  "${CMAKE_INSTALL_INCLUDEDIR}/op2")
set(OP2_INSTALL_TRANSLATORDIR "${CMAKE_INSTALL_LIBEXECDIR}/op2/translator")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/OP2Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/OP2Config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/op2"
    PATH_VARS OP2_INSTALL_INCLUDEDIR OP2_INSTALL_TRANSLATORDIR)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/OP2ConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/OP2Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/OP2ConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/op2")

# Find modules are deliberately not installed: partitioners are pinned by
# path via install(EXPORT) and never searched again, so a consumer needs no
# CMAKE_MODULE_PATH for them.
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/OP2AppSupport.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/OP2TranslatorProbe.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/op2")

# The translator payload's own install rules live in cmake/OP2Translator.cmake.
