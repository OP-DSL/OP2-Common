# The translator's refusals: each case of rejects.cpp (chosen by -D) is a
# loop a C++ target rejects, and rejects.F90 one a Fortran target
# rejects or warns about and falls back to seq for; and the Fortran corpus,
# which every f2c target translates.  A test passes on the exit status and
# the message.

function(_op2_translator_test name fails target source pattern)
    set(_out "${CMAKE_CURRENT_BINARY_DIR}/${name}")
    file(MAKE_DIRECTORY "${_out}")
    string(JOIN "|" _command ${OP2_TRANSLATOR_COMMAND} -t ${target} ${ARGN}
        -I "${PROJECT_SOURCE_DIR}/op2/include" -o "${_out}" ${source})
    add_test(NAME translator_${name}
        COMMAND ${CMAKE_COMMAND} "-DCOMMAND=${_command}" -DFAILS=${fails}
            "-DPATTERN=${pattern}" -P "${CMAKE_CURRENT_SOURCE_DIR}/run_translator.cmake"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(translator_${name} PROPERTIES LABELS translator)
endfunction()

function(_op2_translator_rejects name target source pattern)
    _op2_translator_test(rejects_${name} ON ${target} ${source} "${pattern}" ${ARGN})
endfunction()

function(_op2_translator_falls_back name target source pattern)
    _op2_translator_test(rejects_${name} OFF ${target} ${source} "${pattern}" ${ARGN})
endfunction()

_op2_translator_rejects(conflicting_dat seq rejects.cpp "rejects.cpp:[0-9]+:[0-9]+: error: q_dat is passed as float of dim 1 here, but as double of dim 2 by an earlier argument" -D CONFLICTING_DAT)
_op2_translator_rejects(fortran_conflicting_dat c_seq rejects.F90 "rejects.F90:[0-9]+: error: p_w is passed as real\\(4\\) of dim 1 here, but as real\\(8\\) of dim 1 by an earlier argument" -D CONFLICTING_DAT)
_op2_translator_falls_back(fortran_inc_assigned c_cuda rejects.F90 "arg 1 \\(x\\) of assign_k marked OP_INC but not incremented")
set(_info "info argument has dimension 1, but the reduction it refers to has 2")
_op2_translator_rejects(info_dim seq rejects.cpp "${_info}" -D INFO_DIM)
_op2_translator_rejects(info_ref seq rejects.cpp "info argument must refer to a gbl min or max argument" -D INFO_REF)
_op2_translator_rejects(fortran_info_dim c_seq rejects.F90 "${_info}" -D INFO_DIM)
_op2_translator_rejects(runtime_index cuda rejects.cpp "kernel: runtime map indices and dims are not supported by cuda" -D RUNTIME_INDEX)
_op2_translator_rejects(const_dim seq rejects.cpp "expected a constant int expression" -D CONST_DIM)
_op2_translator_rejects(recursion seq rejects.cpp "recursive kernel functions or types are not supported: fact" -D RECURSION)
_op2_translator_rejects(template_call seq rejects.cpp "rejects.cpp:[0-9]+:[0-9]+: error: kernel: kernels may not call function templates: twice" -D TEMPLATE_CALL)
_op2_translator_rejects(namespaced seq rejects.cpp "rejects.cpp:[0-9]+:[0-9]+: error: half: kernel functions in a namespace are not supported: helpers" -D NAMESPACED)
_op2_translator_falls_back(fortran_implied_save c_cuda rejects.F90 "SAVE local s is unsupported" -D IMPLIED_SAVE)
_op2_translator_rejects(fortran_vector_map c_cuda rejects.F90 "assign_k: vector map arguments are not supported in Fortran" -D VECTOR_MAP)
_op2_translator_falls_back(fortran_slice_legacy openmp rejects.F90 "slice_k element-wise access incompatible with stride insertion" -D SLICE)
_op2_translator_test(fortran_slice_f2c OFF c_seq rejects.F90 "of [0-9]+: rejects_[0-9]+_slice_k" -D SLICE -v)
_op2_translator_rejects(missing_include seq rejects.cpp "'op2_no_such_header.h' file not found" -D MISSING_INCLUDE)
_op2_translator_test(angle_include OFF seq angle_include.cpp "Translated program 1 of 1" -v)
# Every loop of the Fortran corpus (tests/functional/corpus_fortran) translated by each
# f2c target, none left to its seq fallback, which the corpus runs correctly too.
set(_corpus "../functional/corpus_fortran/corpus_kernels.F90;../functional/corpus_fortran/corpus_tests.F90")
foreach(_target c_seq c_cuda c_hip)
    _op2_translator_test(corpus_fortran_${_target} OFF ${_target} "${_corpus}"
        "1 of 5: corpus_tests_1_steps.*2 of 5: corpus_tests_2_logic.*3 of 5: corpus_tests_3_kinds.*4 of 5: corpus_tests_4_increments.*5 of 5: corpus_tests_5_control" -v)
endforeach()
_op2_translator_rejects(legacy_info cuda rejects.cpp "extreme: op_arg_info is not supported by cuda" -D LEGACY_INFO)
_op2_translator_rejects(legacy_info_hip hip rejects.cpp "extreme: op_arg_info is not supported by hip" -D LEGACY_INFO)
