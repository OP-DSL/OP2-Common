# Run one translation, and pass only if it fails (FAILS ON) or succeeds as
# expected and prints PATTERN:
#   cmake -DCOMMAND=<argv>|... -DFAILS=<ON|OFF> -DPATTERN=<regex> -P run_translator.cmake

string(REPLACE "|" ";" command "${COMMAND}")
execute_process(COMMAND ${command}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output)
message("${output}")

if(FAILS AND result EQUAL 0)
    message(FATAL_ERROR "the translator succeeded")
elseif(NOT FAILS AND NOT result EQUAL 0)
    message(FATAL_ERROR "the translator exited with ${result}")
endif()
if(NOT output MATCHES "${PATTERN}")
    message(FATAL_ERROR "no \"${PATTERN}\" in the output")
endif()
