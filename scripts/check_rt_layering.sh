#!/usr/bin/env bash
# The loop runtime's layers, by what includes what:
#  - the core includes from the loop runtime (op2/rt) only op2/rt/lifecycle.h,
#    through which op_init, op_exit and op_profile_output start, release and
#    report on it; the Fortran bindings sit over both and are exempt;
#  - op2/rt/abi, the contract between the library and op2_kernels, includes
#    nothing else of op2/rt;
#  - op2/rt/kernel is compiled into op2_kernels alone, never into a library,
#    so no library source or header includes it, and it includes of op2 only
#    op2/rt/abi and itself;
#  - the generated code includes from op2/rt only op2/rt/abi and
#    op2/rt/kernel;
#  - the neutral layers (loop, jit, tune, plan, the top-level files) include
#    neither backend nor a backend's part of op2/rt/abi, nor do the top-level
#    headers of op2/rt/abi and op2/rt/kernel;
#  - each backend, with its parts of op2/rt/abi and op2/rt/kernel, includes
#    neither the other backend nor its parts;
#  - device headers appear only in the GPU backend, its parts, and the shared
#    headers of op2/rt/kernel;
#  - only these files read variables: each layer's options, the families,
#    and the backends' inits.
set -euo pipefail
cd "$(dirname "$0")/.."

include='^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]op2/rt/'
contract='[<"]op2/rt/(abi|kernel)/'
sources=(--include='*.h' --include='*.hpp' --include='*.cpp' --include='*.cu')
status=0

report() {
    if [ -n "$2" ]; then
        echo "$1:" >&2
        echo "$2" >&2
        status=1
    fi
}

report "core sources include the loop runtime beyond op2/rt/lifecycle.h" \
    "$(grep -rnE "${sources[@]}" "$include" op2/include op2/src |
        grep -vE '^op2/(include/op2/rt|src/rt|src/fortran)/' |
        grep -vE '[<"]op2/rt/lifecycle\.h[>"]' || true)"

report "op2/rt/abi includes the loop runtime beyond op2/rt/abi" \
    "$(grep -rnE "${sources[@]}" "$include" op2/include/op2/rt/abi |
        grep -vE '[<"]op2/rt/abi/' || true)"

report "library sources include op2/rt/kernel, which only op2_kernels compiles" \
    "$(grep -rnE "${sources[@]}" "${include}kernel/" op2/src || true)"

report "library headers include op2/rt/kernel, which only op2_kernels compiles" \
    "$(grep -rnE "${sources[@]}" "${include}kernel/" op2/include/op2/rt |
        grep -vE '^op2/include/op2/rt/kernel/' || true)"

report "op2/rt/kernel includes op2 beyond op2/rt/abi and op2/rt/kernel" \
    "$(grep -rnE "${sources[@]}" '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]op2/' \
        op2/include/op2/rt/kernel | grep -vE "$contract" || true)"

report "the generated code includes the loop runtime beyond op2/rt/abi and op2/rt/kernel" \
    "$(grep -rnE "$include" translator-v2/resources/templates | grep -vE "$contract" || true)"

neutral='^op2/(include/op2/rt|src/rt)/((loop|jit|tune|plan)/|[^/:]+:)'
shared='^op2/include/op2/rt/(abi|kernel)/[^/:]+:'
report "the neutral layers include a backend or its parts" \
    "$(grep -rnE "${sources[@]}" "$include" op2/include/op2/rt op2/src/rt |
        grep -E "$neutral|$shared" | grep -E '[<"]op2/rt/((abi|kernel)/)?(gpu|cpu)/' || true)"

report "a backend includes the other or its parts" \
    "$(grep -rnE "${sources[@]}" "$include" op2/include/op2/rt op2/src/rt |
        grep -E '^op2/(include/op2/rt|src/rt)/((abi|kernel)/)?gpu/' |
        grep -E '[<"]op2/rt/((abi|kernel)/)?cpu/' || true;
       grep -rnE "${sources[@]}" "$include" op2/include/op2/rt op2/src/rt |
        grep -E '^op2/(include/op2/rt|src/rt)/((abi|kernel)/)?cpu/' |
        grep -E '[<"]op2/rt/((abi|kernel)/)?gpu/' || true)"

device='^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]'
device+='(op2/gpu/|op_cuda|op_hip|cuda|hip|nvrtc|thrust/|cub/|cooperative_groups)'
report "device headers outside the GPU backend and op2/rt/kernel's shared headers" \
    "$(grep -rnE "${sources[@]}" "$device" op2/include/op2/rt op2/src/rt |
        grep -vE '^op2/(include/op2/rt|src/rt)/((abi|kernel)/)?gpu/' |
        grep -vE '^op2/include/op2/rt/kernel/[^/:]+:' || true)"

readers='^op2/src/rt/(jit/op_jit_options|tune/op_tune_options|loop/op_loop_family|'
readers+='gpu/op_gpu_init|gpu/op_gpu_heuristics|cpu/op_cpu_init)\.cpp:'
report "variables read outside their readers" \
    "$(grep -rnE "${sources[@]}" '\bparse_(bool|number|text|list|choice)\(|getenv\(' \
        op2/include/op2/rt op2/src/rt | grep -vE "$readers" || true)"

exit $status
