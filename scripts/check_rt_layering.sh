#!/usr/bin/env bash
# The core includes from the loop runtime (op2/rt) only op2/rt/lifecycle.h,
# through which op_init, op_exit and op_profile_output start, release and
# report on it.  The Fortran bindings sit over both and are exempt.
set -euo pipefail
cd "$(dirname "$0")/../op2"

violations=$(grep -rnE '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]op2/rt/' include src |
    grep -vE '^(include/op2/rt|src/rt|src/fortran)/' |
    grep -vE '[<"]op2/rt/lifecycle\.h[>"]' || true)

if [ -n "$violations" ]; then
    echo "core sources include the loop runtime beyond op2/rt/lifecycle.h:" >&2
    echo "$violations" >&2
    exit 1
fi
