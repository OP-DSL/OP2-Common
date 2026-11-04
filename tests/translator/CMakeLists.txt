# The translator's refusals: each case of rejects.cpp (chosen by -D) is a
# loop the C++ cuda target rejects, and rejects.F90 one cuda_f2c rejects or
# warns about and falls back to seq for; a test passes on the exit status and
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

set(_indexed "argument q may only be indexed, but")
_op2_translator_rejects(pointer_arithmetic cuda rejects.cpp "${_indexed} it is used as a pointer: q \\+ 1" -D POINTER_ARITHMETIC)
_op2_translator_rejects(element_address cuda rejects.cpp "${_indexed} its element's address is taken" -D ELEMENT_ADDRESS)
_op2_translator_rejects(local_alias cuda rejects.cpp "${_indexed} it is used as a pointer: const double \\* a = q" -D LOCAL_ALIAS)
_op2_translator_rejects(library_call cuda rejects.cpp "${_indexed} a call takes it" -D LIBRARY_CALL)
_op2_translator_rejects(null_test cuda rejects.cpp "${_indexed} it is used as a pointer: if" -D NULL_TEST)
_op2_translator_rejects(inc_assigned cuda rejects.cpp "OP_INC argument p may only be incremented, but it is assigned" -D INC_ASSIGNED)
_op2_translator_rejects(inc_read cuda rejects.cpp "OP_INC argument p may only be incremented, but it is read" -D INC_READ)
_op2_translator_falls_back(fortran_inc_assigned cuda_f2c rejects.F90 "arg 1 \\(x\\) of assign_k marked OP_INC but not incremented")
set(_info "info argument has dimension 1, but the reduction it refers to has 2")
_op2_translator_rejects(info_dim cuda rejects.cpp "${_info}" -D INFO_DIM)
_op2_translator_rejects(info_ref cuda rejects.cpp "info argument must refer to a gbl min or max argument" -D INFO_REF)
_op2_translator_rejects(fortran_info_dim cuda_f2c rejects.F90 "${_info}" -D INFO_DIM)
_op2_translator_rejects(runtime_index cuda_legacy rejects.cpp "kernel: runtime map indices and dims are not supported by cuda_legacy" -D RUNTIME_INDEX)
_op2_translator_rejects(const_dim cuda rejects.cpp "expected a constant int expression" -D CONST_DIM)
_op2_translator_rejects(recursion cuda rejects.cpp "recursive kernel functions or types are not supported: fact" -D RECURSION)
_op2_translator_falls_back(fortran_implied_save cuda_f2c rejects.F90 "SAVE local s is unsupported" -D IMPLIED_SAVE)
_op2_translator_rejects(fortran_vector_map cuda_f2c rejects.F90 "assign_k: vector map arguments are not supported in Fortran" -D VECTOR_MAP)
_op2_translator_rejects(missing_include cuda rejects.cpp "'op2_no_such_header.h' file not found" -D MISSING_INCLUDE)
_op2_translator_test(angle_include OFF cuda angle_include.cpp "Translated program 1 of 1" -v)
_op2_translator_rejects(legacy_info cuda_legacy rejects.cpp "extreme: op_arg_info is not supported by cuda_legacy" -D LEGACY_INFO)
_op2_translator_rejects(legacy_info_hip hip_legacy rejects.cpp "extreme: op_arg_info is not supported by hip_legacy" -D LEGACY_INFO)
