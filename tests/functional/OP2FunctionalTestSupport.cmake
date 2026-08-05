# OP2FunctionalTestSupport.cmake - CTest wiring for tests/functional/*.
#
# Provides op2_add_functional_tests(), built entirely on op2_add_app_variants()
# (cmake/OP2AppSupport.cmake): no reimplemented variant computation,
# translation, or linking, just the ctest layer on top.

# ---------------------------------------------------------------------------
# op2_add_functional_tests(
#     LANGUAGE cpp|fortran
#     SOURCES <src...>
#     [TRANSLATOR_ARGS <a...>]     # applied to every variant family, e.g.
#                                  # Fortran's --consts-module <file>
#     [SOA_TRANSLATOR_ARGS <a...>] # additionally applied to just the soa/
#                                  # soa_par families, default --force_soa
#     [EXCLUDE_VARIANTS <v...>]    # threaded into all four variant-family calls below
#     [MPI_RANKS_LOWCOST <n>]      # default 8 (seq/genseq/openmp)
#     [MPI_RANKS_GPU <n>]          # default 4 (cuda/c_cuda/hip/c_hip)
#     [OMP_NUM_THREADS <n>]        # default 6
#     [LABELS <l...>]              # extra ctest labels beyond the automatic ones
#     [LINK_LIBRARIES <lib...>]    # linked to every executable created here,
#                                  # for dependencies of the test source itself
#                                  # (OP2 links only what OP2 needs)
# )
#
# NAME is not a parameter - it's always derived from the calling directory's
# name (e.g. tests/functional/dat_reductions/ produces dat_reductions_seq,
# dat_reductions_par_mpi_seq, ...), so a directory can't drift onto some
# other naming convention.
# ---------------------------------------------------------------------------
function(op2_add_functional_tests)
    cmake_parse_arguments(_A
        ""
        "LANGUAGE;MPI_RANKS_LOWCOST;MPI_RANKS_GPU;OMP_NUM_THREADS"
        "SOURCES;TRANSLATOR_ARGS;SOA_TRANSLATOR_ARGS;EXCLUDE_VARIANTS;LABELS;LINK_LIBRARIES"
        ${ARGN})

    if(NOT _A_MPI_RANKS_LOWCOST)
        set(_A_MPI_RANKS_LOWCOST 8)
    endif()
    if(NOT _A_MPI_RANKS_GPU)
        set(_A_MPI_RANKS_GPU 4)
    endif()
    if(NOT _A_OMP_NUM_THREADS)
        set(_A_OMP_NUM_THREADS 6)
    endif()
    if(NOT _A_SOA_TRANSLATOR_ARGS)
        set(_A_SOA_TRANSLATOR_ARGS --force_soa)
    endif()

    get_filename_component(_category "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
    set(_A_NAME "${_category}")

    set(_mpi_variants mpi_seq mpi_genseq mpi_openmp mpi_cuda mpi_hip mpi_c_cuda mpi_c_hip)
    set(_all_targets "")

    # 1. Build the four legacy variant-families (plain / par / soa / soa_par).
    # USE_MPI needs no explicit COMPILE_DEFINITIONS: _op2_add_variant_executable()
    # already applies it to every mpi_-prefixed target. SUMMARY_BUCKET TESTS
    # routes these into the test count instead of the "Apps:" line, reusing
    # op2_add_app_variants()'s own tracking rather than a separate counter.
    op2_add_app_variants(
        NAME "${_A_NAME}" LANGUAGE "${_A_LANGUAGE}" SOURCES ${_A_SOURCES}
        EXCLUDE_VARIANTS ${_mpi_variants} ${_A_EXCLUDE_VARIANTS}
        TRANSLATOR_ONLY_ARGS ${_A_TRANSLATOR_ARGS}
        SUMMARY_BUCKET TESTS
        TARGETS_VAR _fam_targets)
    list(APPEND _all_targets ${_fam_targets})

    op2_add_app_variants(
        NAME "${_A_NAME}_par" LANGUAGE "${_A_LANGUAGE}" SOURCES ${_A_SOURCES}
        VARIANTS ${_mpi_variants} EXCLUDE_VARIANTS ${_A_EXCLUDE_VARIANTS}
        TRANSLATOR_ONLY_ARGS ${_A_TRANSLATOR_ARGS}
        SUMMARY_BUCKET TESTS
        TARGETS_VAR _fam_targets)
    list(APPEND _all_targets ${_fam_targets})

    op2_add_app_variants(
        NAME "${_A_NAME}_soa" LANGUAGE "${_A_LANGUAGE}" SOURCES ${_A_SOURCES}
        EXCLUDE_VARIANTS ${_mpi_variants} ${_A_EXCLUDE_VARIANTS}
        TRANSLATOR_ONLY_ARGS ${_A_TRANSLATOR_ARGS} ${_A_SOA_TRANSLATOR_ARGS}
        SUMMARY_BUCKET TESTS
        TARGETS_VAR _fam_targets)
    list(APPEND _all_targets ${_fam_targets})

    op2_add_app_variants(
        NAME "${_A_NAME}_soa_par" LANGUAGE "${_A_LANGUAGE}" SOURCES ${_A_SOURCES}
        VARIANTS ${_mpi_variants} EXCLUDE_VARIANTS ${_A_EXCLUDE_VARIANTS}
        TRANSLATOR_ONLY_ARGS ${_A_TRANSLATOR_ARGS} ${_A_SOA_TRANSLATOR_ARGS}
        SUMMARY_BUCKET TESTS
        TARGETS_VAR _fam_targets)
    list(APPEND _all_targets ${_fam_targets})

    if(_A_LINK_LIBRARIES)
        foreach(_t IN LISTS _all_targets)
            target_link_libraries(${_t} PRIVATE ${_A_LINK_LIBRARIES})
        endforeach()
    endif()

    # 2. Register one ctest per executable target that actually got created -
    # variant narrowing (no CUDA, no MPI, ...) just means fewer add_test()
    # calls, never a configure error.
    set(_all_names "${_A_NAME}" "${_A_NAME}_par" "${_A_NAME}_soa" "${_A_NAME}_soa_par")
    set(_all_variants seq genseq openmp cuda hip c_cuda c_hip ${_mpi_variants})

    # Directory count for the summary line - a different unit (source dirs,
    # one per call here) than the TESTS bucket's own NAMES/COUNT (one entry
    # per variant-family call above), so it's tracked with its own property
    # rather than forced through _op2_summary_track().
    set_property(GLOBAL APPEND PROPERTY OP2_SUMMARY_TESTS_DIRS "${_category}")

    foreach(_base_name IN LISTS _all_names)
        foreach(_v IN LISTS _all_variants)
            set(_tgt "${_base_name}_${_v}")
            if(NOT TARGET ${_tgt})
                continue()
            endif()

            # A real list via list(APPEND ...), not a pre-joined ";"-string:
            # MPIEXEC_PREFLAGS/POSTFLAGS are empty by default, and a string
            # join would bake in literal empty-string args once add_test()
            # splits it back apart. Appending unquoted list vars instead
            # makes empty ones vanish rather than becoming stray "" args.
            if(_v MATCHES "^mpi_(.*)")
                set(_base "${CMAKE_MATCH_1}")
                if(_base MATCHES "^(cuda|c_cuda|hip|c_hip)$")
                    set(_nproc "${_A_MPI_RANKS_GPU}")
                else()
                    set(_nproc "${_A_MPI_RANKS_LOWCOST}")
                endif()
                set(_cmd "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "${_nproc}")
                if(MPIEXEC_PREFLAGS)
                    list(APPEND _cmd ${MPIEXEC_PREFLAGS})
                endif()
                list(APPEND _cmd "$<TARGET_FILE:${_tgt}>")
                if(MPIEXEC_POSTFLAGS)
                    list(APPEND _cmd ${MPIEXEC_POSTFLAGS})
                endif()
            else()
                set(_cmd "$<TARGET_FILE:${_tgt}>")
                set(_nproc 1)
            endif()

            add_test(NAME ${_tgt} COMMAND ${_cmd})

            set(_labels functional "${_A_LANGUAGE}" "${_category}" ${_A_LABELS})
            if(_v MATCHES "^mpi_")
                list(APPEND _labels mpi)
            else()
                list(APPEND _labels serial)
            endif()
            set_tests_properties(${_tgt} PROPERTIES
                LABELS     "${_labels}"
                PROCESSORS "${_nproc}")

            if(_v MATCHES "^(mpi_)?openmp$")
                set_tests_properties(${_tgt} PROPERTIES
                    ENVIRONMENT "OMP_NUM_THREADS=${_A_OMP_NUM_THREADS}")
            endif()
        endforeach()
    endforeach()
endfunction()
