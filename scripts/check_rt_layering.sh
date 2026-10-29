#!/usr/bin/env bash
# The loop runtime's layers, by what includes what:
#  - the core includes from the loop runtime (op2/rt) only op2/rt/lifecycle.h,
#    through which op_init, op_exit and op_profile_output start, release and
#    report on it; the Fortran bindings sit over both and are exempt;
#  - op2/rt/kernel is compiled into op2_kernels alone, never into a library,
#    so no library source includes it;
#  - op2/rt/kernel includes from the rest of op2/rt only the backends'
#    interfaces (op2/rt/{cpu,gpu}/loop.h) and the staging layout its kernels
#    read;
#  - the generated code includes from op2/rt only op2/rt/kernel and the
#    backends' interfaces.
set -euo pipefail
cd "$(dirname "$0")/.."

include='^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]op2/rt/'
interfaces='[<"]op2/rt/(cpu|gpu)/loop\.h[>"]'
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

report "library sources include op2/rt/kernel, which only op2_kernels compiles" \
    "$(grep -rnE '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]op2/rt/kernel/' op2/src || true)"

report "op2/rt/kernel includes the loop runtime beyond the backends' interfaces" \
    "$(grep -rnE "$include" op2/include/op2/rt/kernel |
        grep -vE "[<\"]op2/rt/kernel/|$interfaces|[<\"]op2/rt/plan/staging_layout\.h[>\"]" || true)"

report "the generated code includes the loop runtime beyond op2/rt/kernel and the backends' interfaces" \
    "$(grep -rnE "$include" translator-v2/resources/templates |
        grep -vE "[<\"]op2/rt/kernel/|$interfaces" || true)"

exit $status
