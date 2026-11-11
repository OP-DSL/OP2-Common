# Run one app test in a directory of its own, the meshes it reads linked in
# under the names it opens, and pass only if it exits 0 and prints PASS:
#   cmake -DRUN_DIR=<dir> -DLINKS=<mesh>@<name>|... -DPASS=<regex>
#         -DCOMMAND=<argv>|... -P run_app.cmake

file(REMOVE_RECURSE "${RUN_DIR}")
file(MAKE_DIRECTORY "${RUN_DIR}")

string(REPLACE "|" ";" links "${LINKS}")
foreach(link IN LISTS links)
    string(FIND "${link}" "@" at)
    string(SUBSTRING "${link}" 0 ${at} mesh)
    math(EXPR name_start "${at} + 1")
    string(SUBSTRING "${link}" ${name_start} -1 name)
    file(CREATE_LINK "${mesh}" "${RUN_DIR}/${name}" SYMBOLIC)
endforeach()

string(REPLACE "|" ";" command "${COMMAND}")
execute_process(COMMAND ${command}
    WORKING_DIRECTORY "${RUN_DIR}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output)
message("${output}")

if(NOT result EQUAL 0)
    message(FATAL_ERROR "exited with ${result}")
endif()
if(NOT output MATCHES "${PASS}")
    message(FATAL_ERROR "no \"${PASS}\" in the output")
endif()
