export SCRIPT_RUN_LOC=$PWD
export LIB_LOC=$PWD/../../op2
export APPS_LOC=$PWD/../../apps

precision=( "sp" "dp" )

if [ -f "$SCRIPT_RUN_LOC/${TEST_APP}_test.log" ]; then
    rm $SCRIPT_RUN_LOC/${TEST_APP}_test.log
fi

# call using => validate "<prepend>" "<binary>" "<args>" "<Grep Word>"
function validate {
    local prepend="$1"
    local bin="$2"
    local args="$3"
    local grep_word="$4"

    if ! is_file_available "$bin"; then
        echo "SKIPPING: $bin does not exist" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        return
    fi

    local cmd
    if [[ -n "$prepend" ]]; then
        cmd="$prepend ./$bin $args"
    else
        cmd="./$bin $args"
    fi

    echo "Running: $cmd" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

    file_tail=$bin
    file_tail="${file_tail//./}"     # remove all dots
    file_tail="${file_tail//\//_}"   # replace / with _

    eval "$cmd" > perf_out_$file_tail 2>&1

    grep "Max total runtime" perf_out_$file_tail | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"

    set +e
    grep -q $grep_word perf_out_$file_tail
    local rc=$?
    set -e

    if [[ $rc != 0 ]]; then
        echo $bin "xxxxxxxxxxxxxxxxxxx TEST FAILED" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    else
        echo $bin "+++++++++++++++++++ TEST PASSED"  | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
    fi

    rm perf_out_$file_tail
    echo "" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
}

function is_file_available {
    local file="$1"

    if [ -f "$file" ]; then
        echo "$file exists" >> $SCRIPT_RUN_LOC/${TEST_APP}_test.log
        return 0
    else
        echo "TEST FAILED : $file is missing" >> $SCRIPT_RUN_LOC/${TEST_APP}_test.log
        return 1
    fi
}

function check_all_tests {
    set +e
    grep -q "FAILED" $SCRIPT_RUN_LOC/${TEST_APP}_test.log
    local rc=$?
    set -e
    if [[ $rc != 0 ]]; then
        echo "All ${TEST_APP} Tests Passed" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        return 0
    else
        echo "Some of ${TEST_APP} Tests Failed, Check ${TEST_APP}_test.log file" | tee -a "$SCRIPT_RUN_LOC/${TEST_APP}_test.log"
        return 1
    fi
}