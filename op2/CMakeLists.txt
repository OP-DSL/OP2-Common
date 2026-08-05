# ---------------------------------------------------------------------------
# Source roots
# ---------------------------------------------------------------------------
set(_src "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(_inc "${CMAKE_CURRENT_SOURCE_DIR}/include")

# ---------------------------------------------------------------------------
# Fortran preprocessing - cpp + OP2's @-token line-continuation fixup, run via
# `cmake -P` (portable, no shell/sed/tr). Output lands in
# PROJECT_BINARY_DIR/preproc/.
# ---------------------------------------------------------------------------
set(_op2_preproc_script "${PROJECT_SOURCE_DIR}/cmake/op2_preprocess_fortran.cmake")

function(_op2_preprocess_fortran SRC OUT_VAR)
    get_filename_component(_base "${SRC}" NAME_WE)
    set(_out "${PROJECT_BINARY_DIR}/preproc/${_base}.F90")

    add_custom_command(
        OUTPUT  "${_out}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${PROJECT_BINARY_DIR}/preproc"
        COMMAND "${CMAKE_COMMAND}"
            "-DINPUT=${SRC}"
            "-DOUTPUT=${_out}"
            "-DINCLUDE_DIR=${_inc}"
            "-DCPP=${CMAKE_C_COMPILER}"
            -P "${_op2_preproc_script}"
        DEPENDS "${SRC}" "${_op2_preproc_script}"
        COMMENT "Preprocessing ${_base}.F90"
        VERBATIM)

    set(${OUT_VAR} "${_out}" PARENT_SCOPE)
endfunction()

_op2_preprocess_fortran(
    "${_src}/fortran/op2_for_declarations.F90"
    _for_declarations_pp)
_op2_preprocess_fortran(
    "${_src}/fortran/op2_for_reference.F90"
    _for_reference_pp)

# ===========================================================================
# Source groups - each variant is core + backend [+ MPI overlay] [+ Fortran]
# ===========================================================================

# --- C/C++ groups --------------------------------------------------------

# Core sources for every library with a backend; op_profile is here because
# all of them profile.
set(_op2_core_srcs
    "${_src}/core/op_lib_core.cpp"
    "${_src}/core/op_rt_support.cpp"
    "${_src}/core/SafeLong.cpp"
    "${_src}/externlib/op_profile.cpp"
)

# Backend-specific groups for single-node variants.
set(_op2_backend_seq_srcs
    "${_src}/core/op_dummy_singlenode.cpp"
    "${_src}/sequential/op_seq.cpp"
)

set(_op2_backend_openmp_srcs
    "${_src}/core/op_dummy_singlenode.cpp"
    "${_src}/openmp/op_openmp_decl.cpp"
)

# Single-node CUDA backend, shared verbatim by CUDA and HIP (HIP promotes the
# .cu to LANGUAGE HIP via _op2_mark_cu_as_hip() and defines OP2_HIP).
set(_op2_backend_cuda_srcs
    "${_src}/cuda/op_cuda_decl.cpp"
    "${_src}/cuda/op_cuda_rt_support.cpp"
    "${_src}/cuda/op2_cuda_rt_wrappers.cu"
)

# No core/backend - meant to be linked alongside one, not standalone.
set(_op2_hdf5_only_srcs
    "${_src}/externlib/op_util.cpp"
    "${_src}/externlib/op_hdf5.cpp"
)

# MPI backbone - present in every MPI variant regardless of accelerator.
set(_op2_backend_mpi_backbone_srcs
    "${_src}/mpi/op_mpi_core.cpp"
    "${_src}/mpi/op_mpi_part_core.cpp"
    "${_src}/mpi/op_mpi_hdf5.cpp"
    "${_src}/mpi/op_mpi_util.cpp"
    "${_src}/externlib/op_util.cpp"
    "${_src}/externlib/op_renumber.cpp"
)

# Exactly one overlay stacks on the MPI backbone: cpu provides
# op_init/decl_dat/...; cuda replaces them with GPU versions and adds the
# kernels.
set(_op2_backend_mpi_cpu_overlay_srcs
    "${_src}/mpi/op_mpi_decl.cpp"
    "${_src}/mpi/op_mpi_rt_support.cpp"
)

set(_op2_backend_mpi_cuda_overlay_srcs
    "${_src}/cuda/op_cuda_rt_support.cpp"
    "${_src}/cuda/op2_cuda_rt_wrappers.cu"
    "${_src}/mpi/op_mpi_cuda_decl.cpp"
    "${_src}/mpi/op_mpi_cuda_rt_support.cpp"
    "${_src}/mpi/op_mpi_cuda_kernels.cu"
)

# --- Fortran groups ------------------------------------------------------

# Core Fortran bridge - every Fortran variant except op2_for_hdf5.
set(_op2_fortran_base_srcs
    "${_for_declarations_pp}"
    "${_for_reference_pp}"
    "${_src}/fortran/op2_for_rt_support.F90"
    "${_src}/fortran/op2_for_rt_wrappers.cpp"
    "${_src}/fortran/op2_C_reference.cpp"
    "${_src}/fortran/op2_for_C_wrappers.cpp"
)

# Only in libraries that also link an HDF5 backend, so no link line ever
# names two providers of the OP2_Fortran_hdf5_Declarations module.
set(_op2_fortran_hdf5_srcs
    "${_src}/fortran/op2_for_hdf5_declarations.F90"
)

# CUDA configuration module for Fortran accelerator variants.
set(_op2_fortran_cuda_config_srcs
    "${_src}/fortran/cudaConfigurationParams.F90"
)

# Link-time stubs for the CUDA-rt symbols the Fortran bridge references
# (op2_C_reference.cpp etc.) when the real cuda_rt_wrappers.cu isn't linked.
set(_op2_fortran_cuda_stub_srcs
    "${_src}/fortran/op_dummy_wrappers.cpp"
)

# ===========================================================================
# Tracking lists for install rules
# ===========================================================================
set(_op2_targets_c "")
set(_op2_targets_for "")

macro(_op2_register_c _t)
    list(APPEND _op2_targets_c "${_t}")
endmacro()
macro(_op2_register_for _t)
    list(APPEND _op2_targets_for "${_t}")
endmacro()

# ===========================================================================
# Per-target helpers
# ===========================================================================

# PUBLIC/INTERFACE, not add_compile_options(): generated kernels compile
# against OP2 and need the same dialect, so this must ride install(EXPORT)
# to consumers. COMPILE_LANG_AND_ID keeps a different-toolchain consumer
# from seeing flags it can't parse.

function(_op2_configure_target tgt)
    target_include_directories(${tgt} PUBLIC
        "$<BUILD_INTERFACE:${_inc}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/op2>")
    target_compile_features(${tgt} PUBLIC cxx_std_20)

    # IMPORTED_NO_SYSTEM: no build effect here, but install(EXPORT) carries
    # it onto the generated imported target, keeping op2:: include dirs -I
    # rather than the imported-target default of -isystem - needed because
    # INCTXT's .incbin (extern/incbin.h) only searches -I.
    set_target_properties(${tgt} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        IMPORTED_NO_SYSTEM TRUE)
endfunction()

# Generated kernels call constexpr host helpers from __global__ functions,
# which nvcc rejects without --expt-relaxed-constexpr.
function(_op2_configure_cuda_target tgt)
    target_compile_features(${tgt} PUBLIC cuda_std_20)
    target_compile_options(${tgt} PUBLIC
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>)
endfunction()

# HIP dialect - same C++ standard requirement, no extra flags.
function(_op2_configure_hip_target tgt)
    target_compile_features(${tgt} PUBLIC hip_std_20)
endfunction()

function(_op2_configure_fortran_target tgt)
    _op2_configure_target(${tgt})

    # Generated Fortran runs well past the 132-column limit, and the bridge
    # modules use Cray pointers.
    target_compile_options(${tgt} PUBLIC
        $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-ffree-line-length-none;-ffixed-line-length-none;-fcray-pointer>
        $<$<COMPILE_LANG_AND_ID:Fortran,NVHPC,PGI>:-Mextend>)

    # INTERFACE only: a static archive is never linked itself.
    if(OP2_LINKER_HAS_NO_WARN_EXECSTACK)
        target_link_options(${tgt} INTERFACE
            $<$<LINK_LANGUAGE:Fortran>:-Wl,--no-warn-execstack>)
    endif()

    # PROJECT_BINARY_DIR: CMAKE_BINARY_DIR would be the outermost project's,
    # scattering these across a parent's build tree under add_subdirectory().
    set_target_properties(${tgt} PROPERTIES
        Fortran_MODULE_DIRECTORY "${PROJECT_BINARY_DIR}/mod/${tgt}")
    target_include_directories(${tgt} INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/mod/${tgt}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/op2/fortran/${tgt}>")
endfunction()

# Used by the four HIP flavours (op2_hip, op2_mpi_hip, op2_for_hip,
# op2_for_mpi_hip), reusing the CUDA backend's .cu files.
function(_op2_mark_cu_as_hip tgt)
    foreach(_s IN LISTS ARGN)
        if(_s MATCHES "\\.cu$")
            set_source_files_properties("${_s}"
                TARGET_DIRECTORY ${tgt}
                PROPERTIES LANGUAGE HIP)
        endif()
    endforeach()
endfunction()

# Attach optional partitioner deps + defines to an MPI target.
#
# By library path, not through the Find modules' imported targets: pins the
# exact artefacts install(EXPORT) records, since the partitioner APIs aren't
# ABI-stable across builds (e.g. scotch.h's opaque handles) - a substituted
# build corrupts memory rather than failing to link. Rebuild OP2 to change
# partitioner.
#
# Include dirs are PRIVATE: no public OP2 header exposes these APIs.
function(_op2_link_partitioners tgt)
    if(PTScotch_FOUND)
        target_link_libraries(${tgt} PUBLIC "${PTScotch_LIBRARY}")
        foreach(_extra PTScotch_ERR_LIBRARY PTScotch_SCOTCH_LIBRARY)
            if(${_extra})
                target_link_libraries(${tgt} PUBLIC "${${_extra}}")
            endif()
        endforeach()
        target_include_directories(${tgt} PRIVATE "${PTScotch_INCLUDE_DIR}")
        target_compile_definitions(${tgt} PRIVATE HAVE_PTSCOTCH)
    endif()
    if(ParMETIS_FOUND)
        target_link_libraries(${tgt} PUBLIC "${ParMETIS_LIBRARY}" "${METIS_LIBRARY}")
        target_include_directories(${tgt} PRIVATE
            "${ParMETIS_INCLUDE_DIR}" "${METIS_INCLUDE_DIR}")
        target_compile_definitions(${tgt} PRIVATE HAVE_PARMETIS PARMETIS_VER_4)
    endif()
    if(KaHIP_FOUND)
        target_link_libraries(${tgt} PUBLIC "${KaHIP_ParHIP_LIBRARY}")
        if(KaHIP_MODIFIED_LIBRARY)
            target_link_libraries(${tgt} PUBLIC "${KaHIP_MODIFIED_LIBRARY}")
        endif()
        target_include_directories(${tgt} PRIVATE "${KaHIP_INCLUDE_DIR}")
        target_compile_definitions(${tgt} PRIVATE HAVE_KAHIP)
    endif()
endfunction()

# ===========================================================================
# C/C++ library targets
# ===========================================================================

# --- op2_seq - sequential, single-node -------------------------------------
add_library(op2_seq STATIC ${_op2_core_srcs} ${_op2_backend_seq_srcs})
_op2_configure_target(op2_seq)
_op2_register_c(op2_seq)

# --- op2_openmp - OpenMP shared-memory -------------------------------------
# As with op2_for_openmp below: the link is a usage requirement for the
# generated kernels, not something this library's own sources need.
if(OpenMP_CXX_FOUND)
    add_library(op2_openmp STATIC ${_op2_core_srcs} ${_op2_backend_openmp_srcs})
    _op2_configure_target(op2_openmp)
    target_link_libraries(op2_openmp PUBLIC OpenMP::OpenMP_CXX)
    _op2_register_c(op2_openmp)
endif()

# --- op2_cuda - NVIDIA GPU via CUDA ----------------------------------------
if(CMAKE_CUDA_COMPILER)
    add_library(op2_cuda STATIC ${_op2_core_srcs} ${_op2_backend_cuda_srcs})
    _op2_configure_target(op2_cuda)
    _op2_configure_cuda_target(op2_cuda)
    target_compile_definitions(op2_cuda PRIVATE OP2_CUDA)
    target_link_libraries(op2_cuda PUBLIC CUDA::cudart)
    _op2_register_c(op2_cuda)
endif()

# --- op2_hip - AMD GPU via HIP (same sources as op2_cuda) ------------------
if(CMAKE_HIP_COMPILER)
    add_library(op2_hip STATIC ${_op2_core_srcs} ${_op2_backend_cuda_srcs})
    _op2_configure_target(op2_hip)
    _op2_configure_hip_target(op2_hip)
    target_compile_definitions(op2_hip PRIVATE OP2_HIP)
    target_link_libraries(op2_hip PUBLIC hip::host)
    _op2_mark_cu_as_hip(op2_hip ${_op2_backend_cuda_srcs})
    _op2_register_c(op2_hip)
endif()

# --- op2_hdf5 - sequential HDF5 I/O (standalone, no MPI) -------------------
if(HDF5_FOUND)
    add_library(op2_hdf5 STATIC ${_op2_hdf5_only_srcs})
    _op2_configure_target(op2_hdf5)
    target_link_libraries(op2_hdf5 PUBLIC ${_op2_hdf5_link_targets})
    _op2_register_c(op2_hdf5)
endif()

# --- MPI variants - only built when both MPI and parallel HDF5 available ---
if(MPI_CXX_FOUND AND HDF5_FOUND AND OP2_HDF5_IS_PARALLEL)

    # --- op2_mpi (CPU) ---
    add_library(op2_mpi STATIC
        ${_op2_core_srcs}
        ${_op2_backend_mpi_backbone_srcs}
        ${_op2_backend_mpi_cpu_overlay_srcs})
    _op2_configure_target(op2_mpi)
    target_compile_definitions(op2_mpi PRIVATE OPMPI)
    target_link_libraries(op2_mpi PUBLIC MPI::MPI_CXX ${_op2_hdf5_link_targets})
    _op2_link_partitioners(op2_mpi)
    _op2_register_c(op2_mpi)

    # --- op2_mpi_cuda ---
    if(CMAKE_CUDA_COMPILER)
        add_library(op2_mpi_cuda STATIC
            ${_op2_core_srcs}
            ${_op2_backend_mpi_backbone_srcs}
            ${_op2_backend_mpi_cuda_overlay_srcs})
        _op2_configure_target(op2_mpi_cuda)
        _op2_configure_cuda_target(op2_mpi_cuda)
        target_compile_definitions(op2_mpi_cuda PRIVATE OPMPI OP2_CUDA)
        target_link_libraries(op2_mpi_cuda PUBLIC
            MPI::MPI_CXX CUDA::cudart ${_op2_hdf5_link_targets})
        _op2_link_partitioners(op2_mpi_cuda)
        _op2_register_c(op2_mpi_cuda)
    endif()

    # --- op2_mpi_hip ---
    if(CMAKE_HIP_COMPILER)
        add_library(op2_mpi_hip STATIC
            ${_op2_core_srcs}
            ${_op2_backend_mpi_backbone_srcs}
            ${_op2_backend_mpi_cuda_overlay_srcs})
        _op2_configure_target(op2_mpi_hip)
        _op2_configure_hip_target(op2_mpi_hip)
        target_compile_definitions(op2_mpi_hip PRIVATE OPMPI OP2_HIP)
        target_link_libraries(op2_mpi_hip PUBLIC
            MPI::MPI_CXX hip::host ${_op2_hdf5_link_targets})
        _op2_mark_cu_as_hip(op2_mpi_hip ${_op2_backend_mpi_cuda_overlay_srcs})
        _op2_link_partitioners(op2_mpi_hip)
        _op2_register_c(op2_mpi_hip)
    endif()

endif() # MPI + parallel HDF5

# ===========================================================================
# Fortran binding libraries
# ===========================================================================
if(CMAKE_Fortran_COMPILER)

    # --- op2_for_seq ---
    add_library(op2_for_seq STATIC
        ${_op2_core_srcs}
        ${_op2_backend_seq_srcs}
        ${_op2_fortran_base_srcs}
        ${_op2_fortran_cuda_stub_srcs})
    _op2_configure_fortran_target(op2_for_seq)
    _op2_register_for(op2_for_seq)

    # --- op2_for_openmp ---
    # No source here uses OpenMP - the !$omp directives live in generated
    # kernels, so this link is a usage requirement, not a build need.
    if(OpenMP_Fortran_FOUND)
        add_library(op2_for_openmp STATIC
            ${_op2_core_srcs}
            ${_op2_backend_openmp_srcs}
            ${_op2_fortran_base_srcs}
            ${_op2_fortran_cuda_stub_srcs})
        _op2_configure_fortran_target(op2_for_openmp)
        target_link_libraries(op2_for_openmp PUBLIC OpenMP::OpenMP_Fortran)
        _op2_register_for(op2_for_openmp)
    endif()

    # --- op2_for_hdf5 ---
    # Needs only the Fortran API declarations module, not the full runtime
    # bridge, so it references _for_declarations_pp directly.
    if(HDF5_FOUND)
        add_library(op2_for_hdf5 STATIC
            ${_op2_hdf5_only_srcs}
            "${_for_declarations_pp}"
            ${_op2_fortran_hdf5_srcs})
        _op2_configure_fortran_target(op2_for_hdf5)
        target_link_libraries(op2_for_hdf5 PUBLIC ${_op2_hdf5_link_targets})
        _op2_register_for(op2_for_hdf5)
    endif()

    # --- op2_for_cuda - CUDA C backend with Fortran bindings.
    # OP2_WITH_CUDAFOR (enabling 'use cudafor') only fires for compilers that
    # support CUDA Fortran (NVHPC/PGI with -cuda).
    if(CMAKE_CUDA_COMPILER)
        add_library(op2_for_cuda STATIC
            ${_op2_core_srcs}
            ${_op2_backend_cuda_srcs}
            ${_op2_fortran_base_srcs}
            ${_op2_fortran_cuda_config_srcs})
        _op2_configure_fortran_target(op2_for_cuda)
        _op2_configure_cuda_target(op2_for_cuda)
        target_compile_definitions(op2_for_cuda PRIVATE OP2_CUDA)
        target_link_libraries(op2_for_cuda PUBLIC CUDA::cudart)
        if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
            target_compile_definitions(op2_for_cuda PRIVATE OP2_WITH_CUDAFOR)
            target_compile_options(op2_for_cuda PRIVATE
                $<$<COMPILE_LANGUAGE:Fortran>:-cuda>)
        endif()
        _op2_register_for(op2_for_cuda)
    endif()

    # --- op2_for_hip - same sources as op2_for_cuda, promotes .cu to HIP ---
    if(CMAKE_HIP_COMPILER)
        add_library(op2_for_hip STATIC
            ${_op2_core_srcs}
            ${_op2_backend_cuda_srcs}
            ${_op2_fortran_base_srcs}
            ${_op2_fortran_cuda_config_srcs})
        _op2_configure_fortran_target(op2_for_hip)
        _op2_configure_hip_target(op2_for_hip)
        target_compile_definitions(op2_for_hip PRIVATE OP2_HIP)
        target_link_libraries(op2_for_hip PUBLIC hip::host)
        _op2_mark_cu_as_hip(op2_for_hip ${_op2_backend_cuda_srcs})
        _op2_register_for(op2_for_hip)
    endif()

    # --- MPI Fortran variants ---
    # MPI_CXX_FOUND, not MPI_Fortran_FOUND: the Fortran bridge reaches MPI
    # through C++ (MPI_Barrier in op2_for_C_wrappers.cpp); an app doing
    # `use mpi` finds MPI::MPI_Fortran itself.
    if(MPI_CXX_FOUND AND HDF5_FOUND AND OP2_HDF5_IS_PARALLEL)

        # --- op2_for_mpi ---
        add_library(op2_for_mpi STATIC
            ${_op2_core_srcs}
            ${_op2_backend_mpi_backbone_srcs}
            ${_op2_backend_mpi_cpu_overlay_srcs}
            ${_op2_fortran_base_srcs}
            ${_op2_fortran_hdf5_srcs}
            ${_op2_fortran_cuda_stub_srcs})
        _op2_configure_fortran_target(op2_for_mpi)
        target_compile_definitions(op2_for_mpi PRIVATE OPMPI)
        target_link_libraries(op2_for_mpi PUBLIC
            MPI::MPI_CXX ${_op2_hdf5_link_targets})
        _op2_link_partitioners(op2_for_mpi)
        _op2_register_for(op2_for_mpi)

        # --- op2_for_mpi_cuda ---
        if(CMAKE_CUDA_COMPILER)
            add_library(op2_for_mpi_cuda STATIC
                ${_op2_core_srcs}
                ${_op2_backend_mpi_backbone_srcs}
                ${_op2_backend_mpi_cuda_overlay_srcs}
                ${_op2_fortran_base_srcs}
                ${_op2_fortran_hdf5_srcs}
                ${_op2_fortran_cuda_config_srcs})
            _op2_configure_fortran_target(op2_for_mpi_cuda)
            _op2_configure_cuda_target(op2_for_mpi_cuda)
            target_compile_definitions(op2_for_mpi_cuda PRIVATE OPMPI OP2_CUDA)
            target_link_libraries(op2_for_mpi_cuda PUBLIC
                MPI::MPI_CXX CUDA::cudart ${_op2_hdf5_link_targets})
            if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
                target_compile_definitions(op2_for_mpi_cuda PRIVATE OP2_WITH_CUDAFOR)
                target_compile_options(op2_for_mpi_cuda PRIVATE
                    $<$<COMPILE_LANGUAGE:Fortran>:-cuda>)
            endif()
            _op2_link_partitioners(op2_for_mpi_cuda)
            _op2_register_for(op2_for_mpi_cuda)
        endif()

        # --- op2_for_mpi_hip ---
        if(CMAKE_HIP_COMPILER)
            add_library(op2_for_mpi_hip STATIC
                ${_op2_core_srcs}
                ${_op2_backend_mpi_backbone_srcs}
                ${_op2_backend_mpi_cuda_overlay_srcs}
                ${_op2_fortran_base_srcs}
                ${_op2_fortran_hdf5_srcs}
                ${_op2_fortran_cuda_config_srcs})
            _op2_configure_fortran_target(op2_for_mpi_hip)
            _op2_configure_hip_target(op2_for_mpi_hip)
            target_compile_definitions(op2_for_mpi_hip PRIVATE OPMPI OP2_HIP)
            target_link_libraries(op2_for_mpi_hip PUBLIC
                MPI::MPI_CXX hip::host ${_op2_hdf5_link_targets})
            _op2_mark_cu_as_hip(op2_for_mpi_hip ${_op2_backend_mpi_cuda_overlay_srcs})
            _op2_link_partitioners(op2_for_mpi_hip)
            _op2_register_for(op2_for_mpi_hip)
        endif()

    endif() # MPI + parallel HDF5

endif() # Fortran compiler

# ===========================================================================
# Namespaced aliases - mirrors the install-time NAMESPACE op2:: on OP2Targets
# so in-tree callers use the same name-form as downstream users.
# ===========================================================================
foreach(_t IN LISTS _op2_targets_c _op2_targets_for)
    if(NOT TARGET op2::${_t})
        add_library(op2::${_t} ALIAS ${_t})
    endif()
endforeach()

# ===========================================================================
# Status report
# ===========================================================================
# Detailed lists appear at VERBOSE only; the top-level configuration summary
# repeats them in the final block.
message(VERBOSE "OP2: C/C++ libraries: ${_op2_targets_c}")
message(VERBOSE "OP2: Fortran libraries: ${_op2_targets_for}")
# Promote to global properties so the top-level summary can read them.
set_property(GLOBAL PROPERTY OP2_LIBS_C   "${_op2_targets_c}")
set_property(GLOBAL PROPERTY OP2_LIBS_FOR "${_op2_targets_for}")

# ===========================================================================
# Install rules
# ===========================================================================
if(NOT OP2_ENABLE_INSTALL)
    return()
endif()

if(_op2_targets_c)
    install(TARGETS ${_op2_targets_c}
        EXPORT  OP2Targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()

if(_op2_targets_for)
    install(TARGETS ${_op2_targets_for}
        EXPORT  OP2Targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
    foreach(_t IN LISTS _op2_targets_for)
        install(DIRECTORY "${PROJECT_BINARY_DIR}/mod/${_t}/"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/op2/fortran/${_t}")
    endforeach()
endif()

install(DIRECTORY "${_inc}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/op2"
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp")
