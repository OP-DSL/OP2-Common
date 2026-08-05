# OP2TranslatorProbe.cmake - is there a Python 3 with the translator's runtime
# dependencies?
#
# Shared by cmake/OP2Translator.cmake (in-tree) and the installed
# OP2Config.cmake (downstream), so the check exists exactly once.  The caller
# sets _op2_translator_pkg_dir (invoked as `python3 <dir>`) and
# _op2_translator_reqs_file (named in the guidance message only).
#
# Sets OP2_HAS_TRANSLATOR and OP2_TRANSLATOR_COMMAND, CACHE INTERNAL so they
# stay visible to op2_add_app_variants() from any directory scope.
set(OP2_HAS_TRANSLATOR FALSE CACHE INTERNAL "")
set(OP2_TRANSLATOR_COMMAND "" CACHE INTERNAL "")

find_package(Python3 3.8 COMPONENTS Interpreter QUIET)
if(NOT Python3_Interpreter_FOUND)
    message(STATUS "OP2: translator   = Python 3 not found - translated app variants will not be built")
    return()
endif()

# Hand-maintained against translator-v2/requirements.txt.  Import names match
# the PyPI names lowercased, except libclang -> clang.cindex.
set(_op2_translator_imports jinja2 fparser pcpp sympy clang.cindex)
list(JOIN _op2_translator_imports "; import " _op2_translator_import_stmt)
set(_op2_translator_import_stmt "import ${_op2_translator_import_stmt}")

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "${_op2_translator_import_stmt}"
    RESULT_VARIABLE _op2_translator_probe
    OUTPUT_QUIET ERROR_QUIET)

if(_op2_translator_probe EQUAL 0)
    set(OP2_TRANSLATOR_COMMAND "${Python3_EXECUTABLE}" "${_op2_translator_pkg_dir}" CACHE INTERNAL "")
    set(OP2_HAS_TRANSLATOR TRUE CACHE INTERNAL "")
    message(STATUS "OP2: translator   = found (${Python3_EXECUTABLE})")
else()
    message(STATUS
        "OP2: translator   = ${Python3_EXECUTABLE} is missing required packages - translated app variants will not be built")
    message(STATUS "    fix with:  ${Python3_EXECUTABLE} -m pip install -r ${_op2_translator_reqs_file}")
    message(STATUS "    or point Python3_EXECUTABLE at an interpreter that already has them")
endif()
