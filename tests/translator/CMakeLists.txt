# The translator's refusals: each case of rejects.cpp (chosen by -D) is a
# loop the C++ cuda target rejects, and rejects.F90 one cuda_f2c warns about
# and falls back to seq for; a test passes on the message.

function(_op2_translator_rejects name target source pattern)
    set(_out "${CMAKE_CURRENT_BINARY_DIR}/${name}")
    file(MAKE_DIRECTORY "${_out}")
    add_test(NAME translator_rejects_${name}
        COMMAND ${OP2_TRANSLATOR_COMMAND} -t ${target} ${ARGN}
            -I "${PROJECT_SOURCE_DIR}/op2/include" -o "${_out}" ${source}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(translator_rejects_${name} PROPERTIES
        LABELS translator
        PASS_REGULAR_EXPRESSION "${pattern}")
endfunction()

set(_indexed "argument q may only be indexed, but")
_op2_translator_rejects(pointer_arithmetic cuda rejects.cpp "${_indexed} it is used as a pointer: q \\+ 1" -D POINTER_ARITHMETIC)
_op2_translator_rejects(element_address cuda rejects.cpp "${_indexed} its element's address is taken" -D ELEMENT_ADDRESS)
_op2_translator_rejects(local_alias cuda rejects.cpp "${_indexed} it is used as a pointer: const double \\* a = q" -D LOCAL_ALIAS)
_op2_translator_rejects(library_call cuda rejects.cpp "${_indexed} a call takes it" -D LIBRARY_CALL)
_op2_translator_rejects(null_test cuda rejects.cpp "${_indexed} it is used as a pointer: if" -D NULL_TEST)
_op2_translator_rejects(inc_assigned cuda rejects.cpp "OP_INC argument p may only be incremented, but it is assigned" -D INC_ASSIGNED)
_op2_translator_rejects(inc_read cuda rejects.cpp "OP_INC argument p may only be incremented, but it is read" -D INC_READ)
_op2_translator_rejects(fortran_inc_assigned cuda_f2c rejects.F90 "arg 1 \\(x\\) of assign_k marked OP_INC but not incremented")
