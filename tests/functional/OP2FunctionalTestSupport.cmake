# OP2FunctionalTestSupport.cmake - CTest wiring for tests/functional/*.
#
# Provides op2_add_functional_tests(), built entirely on op2_add_app_variants()
# (cmake/OP2AppSupport.cmake): no reimplemented variant computation,
# translation, or linking, just the ctest layer on top.

# ---------------------------------------------------------------------------
# op2_add_functional_tests(
#     LANGUAGE cpp|fortran
#     SOURCES <src...>
#     [NAME <name>]                # the tests' name, default the directory's
#     [TRANSLATOR_ARGS <a...>]     # applied to every family, e.g. Fortran's
#                                  # --consts-module <file>
#     [FAMILIES <f...>]            # which builds to test, default serial:
#                                  #   serial   every variant
#                                  #   mpi      every mpi_ variant
#                                  #   soa      the GPU variants with
#                                  #            --force_soa (the CPU ones
#                                  #            compile the same code)
#                                  #   soa_mpi  their mpi_ variants
#     [EXCLUDE_VARIANTS <v...>]    # threaded into every family
#     [LINK_LIBRARIES <lib...>]    # linked to every executable created here,
#                                  # for dependencies of the test source itself
#                                  # (OP2 links only what OP2 needs)
# )
# ---------------------------------------------------------------------------
set(_OP2_TEST_MPI_RANKS_LOWCOST 8)
set(_OP2_TEST_MPI_RANKS_GPU 4)
set(_OP2_TEST_OMP_NUM_THREADS 6)

# A test of the variant in the caller's _cmd, _labels and _nproc, with the
# environment given after its name.
macro(_op2_variant_test _name)
    add_test(NAME ${_name} COMMAND ${_cmd})
    set_tests_properties(${_name} PROPERTIES
        LABELS      "${_labels}"
        PROCESSORS  "${_nproc}"
        ENVIRONMENT "${ARGN}")
endmacro()

function(op2_add_functional_tests)
    cmake_parse_arguments(_A
        ""
        "LANGUAGE;NAME"
        "SOURCES;TRANSLATOR_ARGS;FAMILIES;EXCLUDE_VARIANTS;LINK_LIBRARIES"
        ${ARGN})

    if(NOT _A_FAMILIES)
        set(_A_FAMILIES serial)
    endif()
    foreach(_family IN LISTS _A_FAMILIES)
        if(NOT _family MATCHES "^(serial|mpi|soa|soa_mpi)$")
            message(FATAL_ERROR "unknown test family '${_family}'")
        endif()
    endforeach()

    get_filename_component(_category "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
    if(NOT _A_NAME)
        set(_A_NAME "${_category}")
    endif()

    set(_all_targets "")
    set(_all_variants "")

    # op2_add_app_variants() reports exactly which targets it created, so take
    # its word rather than rebuilding <name>_<variant> from a variant list and
    # probing with if(TARGET). The family name is a known prefix, which is what
    # makes the variant recoverable - a target name alone is ambiguous, since
    # the base can contain underscores too (const_soa_par_mpi_seq).
    macro(_op2_take_family _prefix)
        foreach(_t IN LISTS _fam_targets)
            list(APPEND _all_targets "${_t}")
            string(REGEX REPLACE "^${_prefix}_" "" _variant "${_t}")
            list(APPEND _all_variants "${_variant}")
        endforeach()
    endmacro()

    # 1. Build each family asked for.  The MPI families pass USE_MPI to the
    # call, so the translator sees it too. SUMMARY_BUCKET TESTS routes these
    # into the test count instead of the "Apps:" line, reusing
    # op2_add_app_variants()'s own tracking rather than a separate counter.
    if("serial" IN_LIST _A_FAMILIES)
        op2_add_app_variants(
            NAME "${_A_NAME}" LANGUAGE "${_A_LANGUAGE}" SOURCES ${_A_SOURCES}
            EXCLUDE_VARIANTS mpi_* ${_A_EXCLUDE_VARIANTS}
            TRANSLATOR_ONLY_ARGS ${_A_TRANSLATOR_ARGS}
            SUMMARY_BUCKET TESTS
            TARGETS_VAR _fam_targets)
        _op2_take_family("${_A_NAME}")
    endif()

    if("mpi" IN_LIST _A_FAMILIES)
        op2_add_app_variants(
            NAME "${_A_NAME}_par" LANGUAGE "${_A_LANGUAGE}" SOURCES ${_A_SOURCES}
            VARIANTS mpi_* EXCLUDE_VARIANTS ${_A_EXCLUDE_VARIANTS}
            COMPILE_DEFINITIONS USE_MPI
            TRANSLATOR_ONLY_ARGS ${_A_TRANSLATOR_ARGS}
            SUMMARY_BUCKET TESTS
            TARGETS_VAR _fam_targets)
        _op2_take_family("${_A_NAME}_par")
    endif()

    if("soa" IN_LIST _A_FAMILIES)
        op2_add_app_variants(
            NAME "${_A_NAME}_soa" LANGUAGE "${_A_LANGUAGE}" SOURCES ${_A_SOURCES}
            VARIANTS cuda* hip* c_cuda* c_hip* EXCLUDE_VARIANTS ${_A_EXCLUDE_VARIANTS}
            TRANSLATOR_ONLY_ARGS ${_A_TRANSLATOR_ARGS} --force_soa
            SUMMARY_BUCKET TESTS
            TARGETS_VAR _fam_targets)
        _op2_take_family("${_A_NAME}_soa")
    endif()

    if("soa_mpi" IN_LIST _A_FAMILIES)
        op2_add_app_variants(
            NAME "${_A_NAME}_soa_par" LANGUAGE "${_A_LANGUAGE}" SOURCES ${_A_SOURCES}
            VARIANTS mpi_cuda* mpi_hip* mpi_c_cuda* mpi_c_hip* EXCLUDE_VARIANTS ${_A_EXCLUDE_VARIANTS}
            COMPILE_DEFINITIONS USE_MPI
            TRANSLATOR_ONLY_ARGS ${_A_TRANSLATOR_ARGS} --force_soa
            SUMMARY_BUCKET TESTS
            TARGETS_VAR _fam_targets)
        _op2_take_family("${_A_NAME}_soa_par")
    endif()

    if(_A_LINK_LIBRARIES)
        foreach(_t IN LISTS _all_targets)
            target_link_libraries(${_t} PRIVATE ${_A_LINK_LIBRARIES})
        endforeach()
    endif()

    # 2. Register one ctest per executable target that actually got created -
    # variant narrowing (no CUDA, no MPI, ...) just means fewer add_test()
    # calls, never a configure error.
    # Directory count for the summary line - a different unit (source dirs,
    # one per call here) than the TESTS bucket's own NAMES/COUNT (one entry
    # per variant-family call above), so it's tracked with its own property
    # rather than forced through _op2_summary_track().
    set_property(GLOBAL APPEND PROPERTY OP2_SUMMARY_TESTS_DIRS "${_category}")

    if(NOT _all_targets)
        return()
    endif()

    list(LENGTH _all_targets _n_targets)
    math(EXPR _last_target "${_n_targets} - 1")
    foreach(_i RANGE ${_last_target})
        list(GET _all_targets ${_i} _tgt)
        list(GET _all_variants ${_i} _v)

        # A real list via list(APPEND ...), not a pre-joined ";"-string:
        # MPIEXEC_PREFLAGS/POSTFLAGS are empty by default, and a string
        # join would bake in literal empty-string args once add_test()
        # splits it back apart. Appending unquoted list vars instead
        # makes empty ones vanish rather than becoming stray "" args.
        if(_v MATCHES "^mpi_(.*)")
            set(_base "${CMAKE_MATCH_1}")
            if(_base MATCHES "^(cuda|c_cuda|hip|c_hip)$")
                set(_nproc ${_OP2_TEST_MPI_RANKS_GPU})
            else()
                set(_nproc ${_OP2_TEST_MPI_RANKS_LOWCOST})
            endif()
            set(_cmd "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${_nproc}")
            if(MPIEXEC_PREFLAGS)
                list(APPEND _cmd ${MPIEXEC_PREFLAGS})
            endif()
            list(APPEND _cmd "$<TARGET_FILE:${_tgt}>")
            if(MPIEXEC_POSTFLAGS)
                list(APPEND _cmd ${MPIEXEC_POSTFLAGS})
            endif()
            set(_labels functional "${_A_LANGUAGE}" "${_category}" mpi)
        else()
            set(_cmd "$<TARGET_FILE:${_tgt}>")
            set(_nproc 1)
            set(_labels functional "${_A_LANGUAGE}" "${_category}" serial)
        endif()

        set(_base_env "")
        if(_v MATCHES "^(mpi_)?openmp$")
            set(_base_env "OMP_NUM_THREADS=${_OP2_TEST_OMP_NUM_THREADS}")
        endif()
        _op2_variant_test(${_tgt} ${_base_env})
    endforeach()
endfunction()
