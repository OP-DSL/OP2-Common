# OP2AppSupport.cmake
#
# Public helpers for building OP2 apps with translator integration.
# Included via find_package(OP2 CONFIG); also usable in-tree.
#
# Public entry points:
#   op2_add_app_variants(...)    - build every supported variant of an app
#   op2_translate(...)           - low-level: a single translator invocation
#
# Works with or without a translator: `seq` needs no translation and stays
# buildable with no Python present; every other variant gates on
# OP2_HAS_TRANSLATOR the same way CUDA/HIP/OpenMP variants gate on their own
# toolchains.

# ---------------------------------------------------------------------------
# Internal: variant -> attributes lookup table. Per (language, variant): the
# toolchain it needs (see _op2_toolchain_available), translator target
# ("" = none, compile the raw source), non-MPI/MPI library, master-kernel
# source + language, extra generated sources. The only description of a
# variant - buildability is derived from it on demand, not precomputed.
#
# extra_srcs are fixed filenames, not globs: the translator amalgamates
# per-loop compile units regardless of how many op_par_loop calls exist.
#
# CACHE INTERNAL, not plain variables: read from inside a function(), which
# resolves against the *calling* scope, so a consumer's own directory would
# otherwise see none of it.
# ---------------------------------------------------------------------------

set(_op2_variants_cpp seq genseq openmp cuda hip c_cuda c_hip CACHE INTERNAL "")
set(_op2_variants_fortran seq genseq openmp cuda c_cuda c_hip CACHE INTERNAL "")

macro(_op2_defvariant lang var needs xt lib mpilib msrc mlang extra)
    set(_op2_variant_${lang}_${var}_needs         "${needs}"  CACHE INTERNAL "")
    set(_op2_variant_${lang}_${var}_xlator_target "${xt}"     CACHE INTERNAL "")
    set(_op2_variant_${lang}_${var}_lib           "${lib}"    CACHE INTERNAL "")
    set(_op2_variant_${lang}_${var}_mpi_lib       "${mpilib}" CACHE INTERNAL "")
    set(_op2_variant_${lang}_${var}_master_src    "${msrc}"   CACHE INTERNAL "")
    set(_op2_variant_${lang}_${var}_master_lang   "${mlang}"  CACHE INTERNAL "")
    set(_op2_variant_${lang}_${var}_extra_srcs    "${extra}"  CACHE INTERNAL "")
endmacro()
#                lang    var     needs            xlator   lib                    mpi_lib                 master_src        master_lang extra_srcs
_op2_defvariant(cpp     seq     ""               ""       "op2::op2_seq"         "op2::op2_mpi"          ""                ""          "")
_op2_defvariant(cpp     genseq  ""               "seq"    "op2::op2_seq"         "op2::op2_mpi"          "op2_kernels.cpp" "CXX"       "")
_op2_defvariant(cpp     openmp  "openmp_cxx"     "openmp" "op2::op2_openmp"      "op2::op2_mpi"          "op2_kernels.cpp" "CXX"       "")
_op2_defvariant(cpp     cuda    "cuda"           "cuda"   "op2::op2_cuda"        "op2::op2_mpi_cuda"     "op2_kernels.cu"  "CUDA"      "")
_op2_defvariant(cpp     hip     "hip"            "hip"    "op2::op2_hip"         "op2::op2_mpi_hip"      "op2_kernels.cpp" "HIP"       "")
_op2_defvariant(cpp     c_cuda  "cuda"           "c_cuda" "op2::op2_cuda"        "op2::op2_mpi_cuda"     "op2_kernels.cu"  "CUDA"      "")
_op2_defvariant(cpp     c_hip   "hip"            "c_hip"  "op2::op2_hip"         "op2::op2_mpi_hip"      "op2_kernels.cpp" "HIP"       "")
_op2_defvariant(fortran seq     ""               ""       "op2::op2_for_seq"     "op2::op2_for_mpi"      ""                ""          "")
_op2_defvariant(fortran genseq  ""               "seq"    "op2::op2_for_seq"     "op2::op2_for_mpi"      "op2_kernels.F90" "Fortran"   "op2_consts.F90")
_op2_defvariant(fortran openmp  "openmp_fortran" "openmp" "op2::op2_for_openmp"  "op2::op2_for_mpi"      "op2_kernels.F90" "Fortran"   "op2_consts.F90")
_op2_defvariant(fortran cuda    "cuda_fortran"   "cuda"   "op2::op2_for_cuda"    "op2::op2_for_mpi_cuda" "op2_kernels.F90" "Fortran"   "op2_consts.F90")
_op2_defvariant(fortran c_cuda  "cuda"           "c_cuda" "op2::op2_for_cuda"    "op2::op2_for_mpi_cuda" "op2_kernels.F90" "Fortran"   "op2_consts.F90;op2_kernels_aux1.cu")
_op2_defvariant(fortran c_hip   "hip"            "c_hip"  "op2::op2_for_hip"     "op2::op2_for_mpi_hip"  "op2_kernels.F90" "Fortran"   "op2_consts.F90;op2_kernels_aux1.hip.cpp")

# ---------------------------------------------------------------------------
# Internal: toolchain availability. OpenMP is the one input not already a
# CMake cache entry (CMAKE_CUDA_COMPILER etc. are), so it's recorded here;
# this is what makes mpi_openmp possible, since op2_mpi is CPU-only.
# ---------------------------------------------------------------------------
if(NOT DEFINED OpenMP_CXX_FOUND)
    find_package(OpenMP COMPONENTS CXX QUIET)
endif()
if(NOT DEFINED OpenMP_Fortran_FOUND AND CMAKE_Fortran_COMPILER)
    find_package(OpenMP COMPONENTS Fortran QUIET)
endif()
set(_op2_have_openmp_cxx     "${OpenMP_CXX_FOUND}"     CACHE INTERNAL "")
set(_op2_have_openmp_fortran "${OpenMP_Fortran_FOUND}" CACHE INTERNAL "")

# Is the toolchain named in a variant's `needs` column present?  An unknown
# name is a typo in the variant table, and fails loudly rather than quietly
# marking the variant unbuildable.
function(_op2_toolchain_available NEEDS OUT_VAR)
    if(NEEDS STREQUAL "")
        set(_have TRUE)
    elseif(NEEDS STREQUAL "openmp_cxx")
        set(_have "${_op2_have_openmp_cxx}")
    elseif(NEEDS STREQUAL "openmp_fortran")
        set(_have "${_op2_have_openmp_fortran}")
    elseif(NEEDS STREQUAL "cuda")
        set(_have "${CMAKE_CUDA_COMPILER}")
    elseif(NEEDS STREQUAL "hip")
        set(_have "${CMAKE_HIP_COMPILER}")
    elseif(NEEDS STREQUAL "cuda_fortran")
        # Not just nvcc: the generated Fortran does `use cudafor`, which needs a
        # CUDA-Fortran-capable compiler.
        set(_have FALSE)
        if(CMAKE_CUDA_COMPILER AND CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
            set(_have TRUE)
        endif()
    else()
        message(FATAL_ERROR "OP2: unknown toolchain requirement '${NEEDS}' in the variant table")
    endif()

    # Normalise here, so the branches above can hand back a compiler path.
    if(_have)
        set(${OUT_VAR} TRUE PARENT_SCOPE)
    else()
        set(${OUT_VAR} FALSE PARENT_SCOPE)
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Internal: can this environment build (LANG, VARIANT)? Needs the variant's
# toolchain, plus the translator unless xlator_target is empty (seq: nothing
# to translate).
# ---------------------------------------------------------------------------
function(_op2_variant_buildable LANG VARIANT OUT_VAR)
    if(_op2_variant_${LANG}_${VARIANT}_xlator_target AND NOT OP2_HAS_TRANSLATOR)
        set(${OUT_VAR} FALSE PARENT_SCOPE)
        return()
    endif()
    _op2_toolchain_available("${_op2_variant_${LANG}_${VARIANT}_needs}" _have)
    set(${OUT_VAR} "${_have}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# Internal: compute the set of variants buildable in this environment.
# ---------------------------------------------------------------------------
function(_op2_compute_buildable_variants OUT_VAR LANG)
    set(_out "")
    if(LANG STREQUAL "cpp")
        set(_candidates ${_op2_variants_cpp})
    elseif(LANG STREQUAL "fortran")
        set(_candidates ${_op2_variants_fortran})
    else()
        message(FATAL_ERROR "op2_add_app_variants: LANGUAGE must be cpp or fortran (got ${LANG})")
    endif()

    foreach(_v IN LISTS _candidates)
        _op2_variant_buildable("${LANG}" "${_v}" _ok)
        if(NOT _ok)
            continue()
        endif()
        # Non-MPI form
        set(_lib "${_op2_variant_${LANG}_${_v}_lib}")
        if(_lib AND TARGET ${_lib})
            list(APPEND _out ${_v})
        endif()
        # MPI form: exists iff its lib target exists AND MPI is enabled
        set(_mpilib "${_op2_variant_${LANG}_${_v}_mpi_lib}")
        if(_mpilib AND TARGET ${_mpilib} AND OP2_HAS_MPI)
            list(APPEND _out mpi_${_v})
        endif()
    endforeach()

    set(${OUT_VAR} "${_out}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# Internal: the exact files the translator will emit, known at configure
# time - no globbing, no priming run.
# ---------------------------------------------------------------------------
function(_op2_translator_outputs LANG VARIANTS OUT_DIR SOURCES OUT_VAR)
    set(_out "")

    # Rewritten user programs land at <out_dir>/<basename>.<ext>.
    foreach(_s IN LISTS SOURCES)
        get_filename_component(_bn "${_s}" NAME)
        list(APPEND _out "${OUT_DIR}/${_bn}")
    endforeach()

    # For each variant, look up its translator-target subdir + master + extras.
    foreach(_v IN LISTS VARIANTS)
        # Strip the mpi_ prefix - MPI variants use the same generated output
        # as their non-MPI base.
        set(_base "${_v}")
        string(REGEX REPLACE "^mpi_" "" _base "${_base}")

        set(_xt "${_op2_variant_${LANG}_${_base}_xlator_target}")
        if(NOT _xt)
            continue()  # seq variant: no translator output
        endif()

        set(_master "${_op2_variant_${LANG}_${_base}_master_src}")
        if(_master)
            list(APPEND _out "${OUT_DIR}/${_xt}/${_master}")
        endif()
        foreach(_extra IN LISTS _op2_variant_${LANG}_${_base}_extra_srcs)
            list(APPEND _out "${OUT_DIR}/${_xt}/${_extra}")
        endforeach()
    endforeach()

    list(REMOVE_DUPLICATES _out)
    set(${OUT_VAR} "${_out}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# op2_translate(OUT_DIR <dir> LANGUAGE <cpp|fortran>
#               SOURCES <src...> VARIANTS <variant...>
#               [PROPS_TARGET <tgt>] [EXTRA_ARGS <arg...>]
#               [OUTPUTS_VAR <var>] [STAMP_VAR <var>])
# ---------------------------------------------------------------------------
# Emits one add_custom_command that reruns the translator when SOURCES
# change. PROPS_TARGET's compile definitions/include dirs are forwarded to
# the translator. OUTPUTS_VAR gets the generated files, STAMP_VAR the stamp
# that sequences the translator ahead of anything compiling them.
# ---------------------------------------------------------------------------
function(op2_translate)
    cmake_parse_arguments(_A
        ""
        "OUT_DIR;LANGUAGE;PROPS_TARGET;OUTPUTS_VAR;STAMP_VAR"
        "SOURCES;VARIANTS;EXTRA_ARGS"
        ${ARGN})

    foreach(_req OUT_DIR LANGUAGE SOURCES VARIANTS)
        if(NOT DEFINED _A_${_req})
            message(FATAL_ERROR "op2_translate: ${_req} is required")
        endif()
    endforeach()
    if(_A_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "op2_translate: unrecognised arguments: ${_A_UNPARSED_ARGUMENTS}")
    endif()

    # Multiple variants can share a translator target (e.g. genseq +
    # mpi_genseq both use `-t seq`).
    set(_xlator_targets "")
    foreach(_v IN LISTS _A_VARIANTS)
        set(_base "${_v}")
        string(REGEX REPLACE "^mpi_" "" _base "${_base}")
        set(_xt "${_op2_variant_${_A_LANGUAGE}_${_base}_xlator_target}")
        if(_xt)
            list(APPEND _xlator_targets "${_xt}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _xlator_targets)

    set(_target_args "")
    foreach(_t IN LISTS _xlator_targets)
        list(APPEND _target_args -t ${_t})
    endforeach()

    # PROPS_TARGET-derived -D <macro> / -I <dir> via generator expression.
    # $<JOIN:<list>,;-D;> yields "FOO;-D;BAR"; prefixed with "-D;" and expanded
    # via COMMAND_EXPAND_LISTS at build time yields four argv entries.
    set(_defs_flags "")
    set(_incs_flags "")
    if(_A_PROPS_TARGET)
        set(_defs_genex "$<TARGET_PROPERTY:${_A_PROPS_TARGET},INTERFACE_COMPILE_DEFINITIONS>")
        set(_incs_genex "$<TARGET_PROPERTY:${_A_PROPS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>")
        set(_defs_flags "$<$<BOOL:${_defs_genex}>:-D;$<JOIN:${_defs_genex},;-D;>>")
        set(_incs_flags "$<$<BOOL:${_incs_genex}>:-I;$<JOIN:${_incs_genex},;-I;>>")
    endif()

    file(MAKE_DIRECTORY "${_A_OUT_DIR}")
    _op2_translator_outputs("${_A_LANGUAGE}" "${_A_VARIANTS}" "${_A_OUT_DIR}"
        "${_A_SOURCES}" _outputs)

    # Use a stamp file as the single OUTPUT so Make/Ninja run the translator
    # once per rebuild; the real generated files are BYPRODUCTS.  Downstream
    # source files depend on the stamp via OBJECT_DEPENDS (set in
    # _op2_add_variant_executable).
    set(_stamp "${_A_OUT_DIR}/.op2-translate.stamp")
    add_custom_command(
        OUTPUT "${_stamp}"
        BYPRODUCTS ${_outputs}
        COMMAND ${OP2_TRANSLATOR_COMMAND}
            ${_target_args}
            "${_defs_flags}"
            "${_incs_flags}"
            ${_A_EXTRA_ARGS}
            -o "${_A_OUT_DIR}"
            ${_A_SOURCES}
        COMMAND ${CMAKE_COMMAND} -E touch "${_stamp}"
        DEPENDS ${_A_SOURCES}
        COMMAND_EXPAND_LISTS
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "OP2 translator → ${_A_OUT_DIR}"
        VERBATIM)

    if(_A_OUTPUTS_VAR)
        set(${_A_OUTPUTS_VAR} "${_outputs}" PARENT_SCOPE)
    endif()
    if(_A_STAMP_VAR)
        set(${_A_STAMP_VAR} "${_stamp}" PARENT_SCOPE)
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Internal: configuration-summary accounting.  Each bucket is a pair of GLOBAL
# properties, OP2_SUMMARY_<bucket>_NAMES and _COUNT.  Callers pick a bucket
# instead of reimplementing the counting.
# ---------------------------------------------------------------------------
function(_op2_summary_track bucket name count)
    set_property(GLOBAL APPEND PROPERTY OP2_SUMMARY_${bucket}_NAMES "${name}")
    get_property(_tot GLOBAL PROPERTY OP2_SUMMARY_${bucket}_COUNT)
    if(NOT _tot)
        set(_tot 0)
    endif()
    math(EXPR _tot "${_tot} + ${count}")
    set_property(GLOBAL PROPERTY OP2_SUMMARY_${bucket}_COUNT "${_tot}")
endfunction()

# ---------------------------------------------------------------------------
# op2_add_app_variants(NAME <name> LANGUAGE <cpp|fortran> SOURCES <src...>
#                      [VARIANTS <variant...>] [EXCLUDE_VARIANTS <variant...>]
#                      [COMPILE_DEFINITIONS <def...>]
#                      [INCLUDE_DIRECTORIES <dir...>]
#                      [TRANSLATOR_ONLY_ARGS <arg...>] [OUTPUT_DIR <dir>]
#                      [SUMMARY_BUCKET <name>] [WITH_HDF5] [INSTALL]
#                      [TARGETS_VAR <var>])
# ---------------------------------------------------------------------------
# Builds <name>_<variant> for every variant this environment supports.
# VARIANTS/EXCLUDE_VARIANTS narrow the set; asking for an unavailable variant
# is not an error. COMPILE_DEFINITIONS/INCLUDE_DIRECTORIES must be passed
# here, not applied to the resulting targets - the translator runs against
# them at configure time. WITH_HDF5 links the standalone HDF5 API alongside
# non-MPI variants; INSTALL adds the executables to the install set.
#
# TARGETS_VAR receives the created executables, empty (not undefined) if none
# were buildable, so a caller can attach its own dependencies:
#
#   op2_add_app_variants(NAME myapp ... TARGETS_VAR myapp_targets)
#   foreach(t IN LISTS myapp_targets)
#       target_link_libraries(${t} PRIVATE SomeDep::SomeDep)
#   endforeach()
# ---------------------------------------------------------------------------
function(op2_add_app_variants)
    cmake_parse_arguments(_A
        "INSTALL;WITH_HDF5"
        "NAME;LANGUAGE;OUTPUT_DIR;SUMMARY_BUCKET;TARGETS_VAR"
        "SOURCES;VARIANTS;EXCLUDE_VARIANTS;TRANSLATOR_ONLY_ARGS;COMPILE_DEFINITIONS;INCLUDE_DIRECTORIES"
        ${ARGN})

    # Set up front so every return path below leaves it defined.
    if(_A_TARGETS_VAR)
        set(${_A_TARGETS_VAR} "" PARENT_SCOPE)
    endif()

    if(NOT _A_SUMMARY_BUCKET)
        set(_A_SUMMARY_BUCKET APPS)
    endif()

    foreach(_req NAME LANGUAGE SOURCES)
        if(NOT DEFINED _A_${_req})
            message(FATAL_ERROR "op2_add_app_variants: ${_req} is required")
        endif()
    endforeach()
    if(_A_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "op2_add_app_variants: unrecognised arguments: ${_A_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT _A_LANGUAGE MATCHES "^(cpp|fortran)$")
        message(FATAL_ERROR "op2_add_app_variants: LANGUAGE must be cpp or fortran (got ${_A_LANGUAGE})")
    endif()

    if(NOT _A_OUTPUT_DIR)
        set(_A_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/${_A_NAME}")
    endif()

    # 1. Resolve buildable variants.
    _op2_compute_buildable_variants(_buildable "${_A_LANGUAGE}")
    if(_A_VARIANTS)
        set(_selected "")
        foreach(_v IN LISTS _A_VARIANTS)
            if(_v IN_LIST _buildable)
                list(APPEND _selected ${_v})
            endif()
        endforeach()
        set(_buildable "${_selected}")
    endif()
    if(_A_EXCLUDE_VARIANTS)
        foreach(_v IN LISTS _A_EXCLUDE_VARIANTS)
            list(REMOVE_ITEM _buildable ${_v})
        endforeach()
    endif()

    if(NOT _buildable)
        message(VERBOSE "OP2 app ${_A_NAME}: no buildable variants")
        return()
    endif()
    message(VERBOSE "OP2 app ${_A_NAME}: variants = ${_buildable}")

    list(LENGTH _buildable _n_variants)
    _op2_summary_track(${_A_SUMMARY_BUCKET} "${_A_NAME}" ${_n_variants})

    # 2. Shared-flags target, carrying everything both the translator and the
    # variants' compile lines need.  OP2_INCLUDE_DIR is a plain path, not a
    # genex, so the translator can consume it at configure time.
    set(_props "${_A_NAME}_app_props")
    if(NOT TARGET ${_props})
        add_library(${_props} INTERFACE)
        target_include_directories(${_props} INTERFACE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${OP2_INCLUDE_DIR}")
    endif()
    if(_A_COMPILE_DEFINITIONS)
        target_compile_definitions(${_props} INTERFACE ${_A_COMPILE_DEFINITIONS})
    endif()
    if(_A_INCLUDE_DIRECTORIES)
        target_include_directories(${_props} INTERFACE ${_A_INCLUDE_DIRECTORIES})
    endif()

    # libclang has to resolve #include <mpi.h> when it parses an MPI variant's
    # source.  Fortran needs no equivalent: fparser does not resolve `use mpi`.
    if(_A_LANGUAGE STREQUAL "cpp" AND MPI_CXX_INCLUDE_DIRS)
        foreach(_v IN LISTS _buildable)
            if(_v MATCHES "^mpi_")
                target_include_directories(${_props} INTERFACE ${MPI_CXX_INCLUDE_DIRS})
                break()
            endif()
        endforeach()
    endif()

    # 3. One translator invocation covering every buildable variant.
    set(_needs_translator FALSE)
    foreach(_v IN LISTS _buildable)
        string(REGEX REPLACE "^mpi_" "" _base "${_v}")
        if(_op2_variant_${_A_LANGUAGE}_${_base}_xlator_target)
            set(_needs_translator TRUE)
            break()
        endif()
    endforeach()
    set(_stamp "")
    if(_needs_translator)
        op2_translate(
            OUT_DIR      "${_A_OUTPUT_DIR}"
            LANGUAGE     "${_A_LANGUAGE}"
            SOURCES      ${_A_SOURCES}
            VARIANTS     ${_buildable}
            PROPS_TARGET ${_props}
            EXTRA_ARGS   ${_A_TRANSLATOR_ONLY_ARGS}
            STAMP_VAR    _stamp)
    endif()

    # 4. Emit each executable variant.
    set(_created "")
    foreach(_v IN LISTS _buildable)
        _op2_add_variant_executable(
            NAME     "${_A_NAME}"
            VARIANT  "${_v}"
            LANGUAGE "${_A_LANGUAGE}"
            SOURCES  ${_A_SOURCES}
            OUT_DIR  "${_A_OUTPUT_DIR}"
            STAMP    "${_stamp}"
            PROPS    "${_props}"
            WITH_HDF5 "${_A_WITH_HDF5}")
        list(APPEND _created "${_A_NAME}_${_v}")
        # Skipped when the whole build has OP2_ENABLE_INSTALL=OFF.
        if(_A_INSTALL AND (NOT DEFINED OP2_ENABLE_INSTALL OR OP2_ENABLE_INSTALL))
            install(TARGETS "${_A_NAME}_${_v}"
                RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
        endif()
    endforeach()

    if(_A_TARGETS_VAR)
        set(${_A_TARGETS_VAR} "${_created}" PARENT_SCOPE)
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Internal: build one executable for (name, variant).
# ---------------------------------------------------------------------------
function(_op2_add_variant_executable)
    cmake_parse_arguments(_A
        ""
        "NAME;VARIANT;LANGUAGE;OUT_DIR;STAMP;PROPS;WITH_HDF5"
        "SOURCES"
        ${ARGN})

    # Split off mpi_ prefix.
    set(_mpi FALSE)
    set(_base "${_A_VARIANT}")
    if(_base MATCHES "^mpi_(.*)")
        set(_mpi TRUE)
        set(_base "${CMAKE_MATCH_1}")
    endif()

    set(_xt "${_op2_variant_${_A_LANGUAGE}_${_base}_xlator_target}")
    if(_mpi)
        set(_lib "${_op2_variant_${_A_LANGUAGE}_${_base}_mpi_lib}")
    else()
        set(_lib "${_op2_variant_${_A_LANGUAGE}_${_base}_lib}")
    endif()

    # Build the source list - always deterministic.
    set(_srcs "")
    if(NOT _xt)
        # seq variant: raw user source, no translator.
        foreach(_s IN LISTS _A_SOURCES)
            if(NOT IS_ABSOLUTE "${_s}")
                set(_s "${CMAKE_CURRENT_SOURCE_DIR}/${_s}")
            endif()
            list(APPEND _srcs "${_s}")
        endforeach()
    else()
        _op2_translator_outputs("${_A_LANGUAGE}" "${_A_VARIANT}" "${_A_OUT_DIR}"
            "${_A_SOURCES}" _srcs)
    endif()

    set(_tgt "${_A_NAME}_${_A_VARIANT}")
    add_executable(${_tgt} ${_srcs})

    # GENERATED silences the generate step for files that don't exist yet;
    # OBJECT_DEPENDS on the stamp routes the whole variant through one
    # dependency edge, so the translator runs once per rebuild, not per file.
    if(_xt)
        foreach(_s IN LISTS _srcs)
            set_source_files_properties("${_s}" PROPERTIES GENERATED TRUE)
        endforeach()
        if(_A_STAMP)
            set_source_files_properties(${_srcs} PROPERTIES
                OBJECT_DEPENDS "${_A_STAMP}")
        endif()
    endif()

    target_link_libraries(${_tgt} PRIVATE ${_A_PROPS} ${_lib})

    # MPI backends already bundle the HDF5 API; non-MPI variants need the
    # standalone library alongside their backend for op_decl_dat_hdf5 et al.
    if(_A_WITH_HDF5 AND NOT _mpi)
        if(_A_LANGUAGE STREQUAL "fortran")
            if(TARGET op2::op2_for_hdf5)
                target_link_libraries(${_tgt} PRIVATE op2::op2_for_hdf5)
            endif()
        else()
            if(TARGET op2::op2_hdf5)
                target_link_libraries(${_tgt} PRIVATE op2::op2_hdf5)
            endif()
        endif()
    endif()

    if(_mpi)
        target_compile_definitions(${_tgt} PRIVATE USE_MPI)
    endif()

    # op_gpu_shims.h / op_f2c_helpers.h gate their gpu* aliases on these.
    if(_base MATCHES "^(cuda|c_cuda)$")
        target_compile_definitions(${_tgt} PRIVATE OP2_CUDA)
    elseif(_base MATCHES "^(hip|c_hip)$")
        target_compile_definitions(${_tgt} PRIVATE OP2_HIP)
    endif()

    # c_* variants JIT their kernels at runtime; the precompiled ones do not.
    if(_base STREQUAL "c_cuda")
        if(TARGET CUDA::nvrtc AND TARGET CUDA::cuda_driver)
            target_link_libraries(${_tgt} PRIVATE CUDA::nvrtc CUDA::cuda_driver)
        endif()
    elseif(_base STREQUAL "c_hip")
        if(TARGET hip::hiprtc)
            target_link_libraries(${_tgt} PRIVATE hip::hiprtc)
        endif()
    endif()

    # op2_openmp propagates OpenMP publicly, but the MPI backends are CPU-only
    # and do not, so mpi_openmp needs it named explicitly.
    if(_base STREQUAL "openmp")
        if(_A_LANGUAGE STREQUAL "cpp" AND TARGET OpenMP::OpenMP_CXX)
            target_link_libraries(${_tgt} PRIVATE OpenMP::OpenMP_CXX)
        elseif(_A_LANGUAGE STREQUAL "fortran" AND TARGET OpenMP::OpenMP_Fortran)
            target_link_libraries(${_tgt} PRIVATE OpenMP::OpenMP_Fortran)
        endif()
    endif()

    # HIP variants reuse .cu/.hip.cpp sources; promote them to LANGUAGE HIP.
    set(_master_lang "${_op2_variant_${_A_LANGUAGE}_${_base}_master_lang}")
    if(_master_lang STREQUAL "HIP")
        foreach(_s IN LISTS _srcs)
            if(_s MATCHES "\\.(cu|hip\\.cpp)$")
                set_source_files_properties("${_s}"
                    TARGET_DIRECTORY ${_tgt}
                    PROPERTIES LANGUAGE HIP)
            endif()
        endforeach()
    endif()

    # CUDA Fortran, as in the core library.
    if(_A_LANGUAGE STREQUAL "fortran" AND _base STREQUAL "cuda"
            AND CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
        target_compile_definitions(${_tgt} PRIVATE OP2_WITH_CUDAFOR)
        target_compile_options(${_tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:Fortran>:-cuda>)
    endif()

    # Explicit linker language: mixed Fortran + .cu/.hip.cpp targets must link
    # with the Fortran driver.
    if(_A_LANGUAGE STREQUAL "fortran")
        set_target_properties(${_tgt} PROPERTIES
            Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mod/${_tgt}"
            LINKER_LANGUAGE Fortran)
    endif()
endfunction()
