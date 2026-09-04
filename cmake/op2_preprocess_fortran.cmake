# op2_preprocess_fortran.cmake
#
# Preprocesses an OP2 Fortran source through a C preprocessor and applies the
# custom @-token line-continuation transformations that OP2 relies on.
# Invoked via `cmake -P` from the op2 build:
#
#   cmake -DINPUT=<abs path to .F90 source>
#         -DOUTPUT=<abs path for preprocessed .F90>
#         -DINCLUDE_DIR=<header include dir passed to the preprocessor>
#         -DCPP=<preprocessor executable, typically ${CMAKE_C_COMPILER}>
#         -P op2_preprocess_fortran.cmake
#
# The GNU Make build does the same job with a `cpp | sed | sed | sed | tr`
# shell pipeline (op2/Makefile, the %+cpp.F90 rule); the transformations here
# mirror it exactly.

foreach(_arg IN ITEMS INPUT OUTPUT INCLUDE_DIR CPP)
    if(NOT DEFINED ${_arg})
        message(FATAL_ERROR "op2_preprocess_fortran: ${_arg} not set")
    endif()
endforeach()

# -x c forces C-mode preprocessing regardless of the .F90 extension.
# Without it, gcc/clang inspect the extension and apply Fortran-mode rules
# that expand macros differently (the pipeline expects C-mode expansion).
execute_process(
    COMMAND "${CPP}" -E -x c "-I${INCLUDE_DIR}" "${INPUT}"
    OUTPUT_VARIABLE _content
    RESULT_VARIABLE _rc
    ERROR_VARIABLE  _stderr
)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR
        "op2_preprocess_fortran: preprocess of ${INPUT} failed (exit ${_rc})\n${_stderr}")
endif()

# The sed/tr half of that pipeline:
#   sed 's/##//g'          - strip token-paste artifacts
#   sed 's/"@"//g'         - strip quoted @
#   sed 's|/@/|//|g'       - /@/  -> //
#   tr  '@' '\n'           - @    -> newline
string(REPLACE "##"    ""    _content "${_content}")
string(REPLACE "\"@\"" ""    _content "${_content}")
string(REPLACE "/@/"   "//"  _content "${_content}")
string(REPLACE "@"     "\n"  _content "${_content}")

file(WRITE "${OUTPUT}" "${_content}")
