# OP2FunctionalTestSupport.cmake - CTest wiring for tests/functional/*.
#
# Provides op2_add_functional_tests(), built entirely on op2_add_app_variants()
# (cmake/OP2AppSupport.cmake): no reimplemented variant computation,
# translation, or linking, just the ctest layer on top.

# ---------------------------------------------------------------------------
# op2_add_functional_tests(
#     LANGUAGE cpp|fortran
#     SOURCES <src...>
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
#     [STRATEGIES <s...>]          # also run each strategy-capable variant
#                                  # (cuda/hip, _f2c, openmp_f2c) once per
#                                  # strategy, forced, as <target>_<strategy>,
#                                  # and once exploring them all, as
#                                  # <target>_autotune
# )
#
# A JIT variant's test runs its offline kernels; each of its loops' JIT
# kernels runs, asked for from the first call, under a forced strategy, built
# by the call, or, in a directory with no STRATEGIES, as <target>_jit, built
# on threads as a run's are by default, a call waiting for its floor's.  Strategies run on the mpi
# family when there is one (the same kernels, with halo sections) and on
# soa; the autotuner on the first of those.
#
# NAME is not a parameter - it's always derived from the calling directory's
# name (e.g. tests/functional/dat_reductions/ produces dat_reductions_seq,
# dat_reductions_par_mpi_seq, ...), so a directory can't drift onto some
# other naming convention.
# ---------------------------------------------------------------------------
set(_OP2_TEST_MPI_RANKS_LOWCOST 8)
set(_OP2_TEST_MPI_RANKS_GPU 4)
set(_OP2_TEST_OMP_NUM_THREADS 6)

# Every kernel specialised, asked for from its first call: built by the
# call, or on a thread (the default, which CI environments may override).
set(_OP2_TEST_JIT_ENV OP_JIT_LEVEL=2 OP_JIT_FORCE=1 OP_JIT_MIN_CALLS=1)
set(_OP2_TEST_JIT_SEQ_ENV ${_OP2_TEST_JIT_ENV} OP_JIT_SEQ_COMPILE=1)
set(_OP2_TEST_JIT_THREAD_ENV ${_OP2_TEST_JIT_ENV} OP_JIT_SEQ_COMPILE=0)

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
        "LANGUAGE"
        "SOURCES;TRANSLATOR_ARGS;FAMILIES;EXCLUDE_VARIANTS;LINK_LIBRARIES;STRATEGIES"
        ${ARGN})

    if(NOT _A_FAMILIES)
        set(_A_FAMILIES serial)
    endif()
    foreach(_family IN LISTS _A_FAMILIES)
        if(NOT _family MATCHES "^(serial|mpi|soa|soa_mpi)$")
            message(FATAL_ERROR "unknown test family '${_family}'")
        endif()
    endforeach()
    foreach(_strategy IN LISTS _A_STRATEGIES)
        if(NOT _strategy MATCHES "^(plain|atomics|colouring|chunk_atomics|chunk_colouring)$")
            message(FATAL_ERROR "unknown execution strategy '${_strategy}'")
        endif()
    endforeach()

    get_filename_component(_category "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
    set(_A_NAME "${_category}")

    # The family the strategies run on besides soa, and the one the
    # autotuner runs on.
    if("mpi" IN_LIST _A_FAMILIES)
        set(_strategy_family mpi)
    else()
        set(_strategy_family serial)
    endif()
    if(_strategy_family IN_LIST _A_FAMILIES)
        set(_autotune_family ${_strategy_family})
    else()
        set(_autotune_family soa)
    endif()

    set(_all_targets "")
    set(_all_variants "")
    set(_all_families "")

    # op2_add_app_variants() reports exactly which targets it created, so take
    # its word rather than rebuilding <name>_<variant> from a variant list and
    # probing with if(TARGET). The family name is a known prefix, which is what
    # makes the variant recoverable - a target name alone is ambiguous, since
    # the base can contain underscores too (const_soa_par_mpi_seq).
    macro(_op2_take_family _prefix _family)
        foreach(_t IN LISTS _fam_targets)
            list(APPEND _all_targets "${_t}")
            string(REGEX REPLACE "^${_prefix}_" "" _variant "${_t}")
            list(APPEND _all_variants "${_variant}")
            list(APPEND _all_families "${_family}")
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
        _op2_take_family("${_A_NAME}" serial)
    endif()

    if("mpi" IN_LIST _A_FAMILIES)
        op2_add_app_variants(
            NAME "${_A_NAME}_par" LANGUAGE "${_A_LANGUAGE}" SOURCES ${_A_SOURCES}
            VARIANTS mpi_* EXCLUDE_VARIANTS ${_A_EXCLUDE_VARIANTS}
            COMPILE_DEFINITIONS USE_MPI
            TRANSLATOR_ONLY_ARGS ${_A_TRANSLATOR_ARGS}
            SUMMARY_BUCKET TESTS
            TARGETS_VAR _fam_targets)
        _op2_take_family("${_A_NAME}_par" mpi)
    endif()

    if("soa" IN_LIST _A_FAMILIES)
        op2_add_app_variants(
            NAME "${_A_NAME}_soa" LANGUAGE "${_A_LANGUAGE}" SOURCES ${_A_SOURCES}
            VARIANTS cuda* hip* EXCLUDE_VARIANTS ${_A_EXCLUDE_VARIANTS}
            TRANSLATOR_ONLY_ARGS ${_A_TRANSLATOR_ARGS} --force_soa
            SUMMARY_BUCKET TESTS
            TARGETS_VAR _fam_targets)
        _op2_take_family("${_A_NAME}_soa" soa)
    endif()

    if("soa_mpi" IN_LIST _A_FAMILIES)
        op2_add_app_variants(
            NAME "${_A_NAME}_soa_par" LANGUAGE "${_A_LANGUAGE}" SOURCES ${_A_SOURCES}
            VARIANTS mpi_cuda* mpi_hip* EXCLUDE_VARIANTS ${_A_EXCLUDE_VARIANTS}
            COMPILE_DEFINITIONS USE_MPI
            TRANSLATOR_ONLY_ARGS ${_A_TRANSLATOR_ARGS} --force_soa
            SUMMARY_BUCKET TESTS
            TARGETS_VAR _fam_targets)
        _op2_take_family("${_A_NAME}_soa_par" soa_mpi)
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
        list(GET _all_families ${_i} _family)

        # A real list via list(APPEND ...), not a pre-joined ";"-string:
        # MPIEXEC_PREFLAGS/POSTFLAGS are empty by default, and a string
        # join would bake in literal empty-string args once add_test()
        # splits it back apart. Appending unquoted list vars instead
        # makes empty ones vanish rather than becoming stray "" args.
        if(_v MATCHES "^mpi_(.*)")
            set(_base "${CMAKE_MATCH_1}")
            if(_base MATCHES "^(cuda|hip)(_legacy|_f2c)?$")
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
        if(_v MATCHES "^(mpi_)?openmp(_f2c)?$")
            set(_base_env "OMP_NUM_THREADS=${_OP2_TEST_OMP_NUM_THREADS}")
        endif()
        set(_jit OFF)
        if(_v MATCHES "^(mpi_)?(cuda|hip)(_f2c)?$")
            set(_jit ON)
        endif()

        if(_jit)
            _op2_variant_test(${_tgt} OP_JIT_LEVEL=0 ${_base_env})
            if(NOT _A_STRATEGIES)
                _op2_variant_test(${_tgt}_jit ${_OP2_TEST_JIT_THREAD_ENV} ${_base_env})
            endif()
        else()
            _op2_variant_test(${_tgt} ${_base_env})
        endif()

        if(NOT _A_STRATEGIES OR NOT (_jit OR _v MATCHES "^(mpi_)?openmp_f2c$"))
            continue()
        endif()

        if(_family STREQUAL _strategy_family OR _family STREQUAL "soa")
            foreach(_strategy IN LISTS _A_STRATEGIES)
                set(_env OP_STRATEGY=${_strategy} ${_base_env})
                if(_jit)
                    list(APPEND _env ${_OP2_TEST_JIT_SEQ_ENV})
                endif()
                _op2_variant_test(${_tgt}_${_strategy} ${_env})
                # A loop that passes over its forced strategy has not
                # tested it.
                set_tests_properties(${_tgt}_${_strategy} PROPERTIES
                    LABELS "${_labels};strategy"
                    FAIL_REGULAR_EXPRESSION "passes over its forced strategy")
            endforeach()
        endif()

        if(_family STREQUAL _autotune_family)
            _op2_variant_test(${_tgt}_autotune OP_AUTOTUNE=report OP_JIT_LEVEL=2 ${_base_env})
            set_tests_properties(${_tgt}_autotune PROPERTIES LABELS "${_labels};strategy")
        endif()
    endforeach()
endfunction()
