#!/usr/bin/env bash
# The loop runtime's layers, by what includes what:
#  - the core includes from the loop runtime (op2/rt) only op2/rt/lifecycle.h,
#    through which op_init, op_exit and op_profile_output start, release and
#    report on it; the Fortran bindings sit over both and are exempt;
#  - op2/rt/abi, the contract between the library and op2_kernels, includes
#    nothing else of op2/rt;
#  - op2/rt/kernel is compiled into op2_kernels alone, never into a library,
#    so no library source or header includes it, and it includes only
#    op2/rt/abi and itself;
#  - the generated code includes from op2/rt only op2/rt/abi and
#    op2/rt/kernel;
#  - the neutral layers (loop, jit, tune, plan, the top-level headers) include
#    neither backend nor a backend's part of op2/rt/abi, and each backend
#    includes neither the other nor its part;
#  - device headers appear only in the GPU backend and op2/rt/kernel;
#  - variables are read only by their owners: each layer's options, the
#    families, and the backends' inits.
set -euo pipefail
cd "$(dirname "$0")/.."

include='^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]op2/rt/'
contract='[<"]op2/rt/(abi|kernel)/'
status=0

report() {
    if [ -n "$2" ]; then
        echo "$1:" >&2
        echo "$2" >&2
        status=1
    fi
}

report "core sources include the loop runtime beyond op2/rt/lifecycle.h" \
    "$(grep -rnE "$include" op2/include op2/src |
        grep -vE '^op2/(include/op2/rt|src/rt|src/fortran)/' |
        grep -vE '[<"]op2/rt/lifecycle\.h[>"]' || true)"

report "op2/rt/abi includes the loop runtime beyond op2/rt/abi" \
    "$(grep -rnE "$include" op2/include/op2/rt/abi | grep -vE '[<"]op2/rt/abi/' || true)"

report "library sources include op2/rt/kernel, which only op2_kernels compiles" \
    "$(grep -rnE '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]op2/rt/kernel/' op2/src || true)"

report "library headers include op2/rt/kernel, which only op2_kernels compiles" \
    "$(grep -rnE '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]op2/rt/kernel/' op2/include/op2/rt |
        grep -vE '^op2/include/op2/rt/kernel/' || true)"

report "op2/rt/kernel includes the loop runtime beyond op2/rt/abi and op2/rt/kernel" \
    "$(grep -rnE "$include" op2/include/op2/rt/kernel | grep -vE "$contract" || true)"

report "the generated code includes the loop runtime beyond op2/rt/abi and op2/rt/kernel" \
    "$(grep -rnE "$include" translator-v2/resources/templates | grep -vE "$contract" || true)"

neutral='^op2/(include/op2/rt|src/rt)/(loop|jit|tune|plan|[^/]+$)'
report "the neutral layers include a backend" \
    "$(grep -rnE "$include" op2/include/op2/rt op2/src/rt | grep -E "$neutral" |
        grep -E '[<"]op2/rt/(gpu|cpu|abi/gpu|abi/cpu)/' || true)"

report "a backend includes the other" \
    "$(grep -rnE "$include" op2/include/op2/rt/gpu op2/src/rt/gpu | grep -E '[<"]op2/rt/(abi/)?cpu/' || true;
       grep -rnE "$include" op2/include/op2/rt/cpu op2/src/rt/cpu | grep -E '[<"]op2/rt/(abi/)?gpu/' || true)"

report "device headers outside the GPU backend and op2/rt/kernel" \
    "$(grep -rnE '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"](op2/gpu/|op_cuda|op_hip|cuda|hip|nvrtc)' \
        op2/include/op2/rt op2/src/rt | grep -vE '^op2/(include/op2/rt|src/rt)/(gpu|kernel|abi/gpu)/' || true)"

report "variables read outside their owners" \
    "$(grep -rnE 'env::parse_|getenv\(' op2/include/op2/rt op2/src/rt |
        grep -vE '^op2/src/rt/(jit/op_jit_options|tune/op_tune_options|loop/op_loop_family|gpu/op_gpu_init|gpu/op_gpu_heuristics|cpu/op_cpu_init)\.cpp:' || true)"

exit $status
