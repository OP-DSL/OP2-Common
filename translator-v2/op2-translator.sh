#!/usr/bin/env bash
#
# op2-translator.sh - legacy-Make entry point for the OP2 translator.
#
# Locates a working Python 3 (with our runtime deps) and runs the translator.
# Python is a dependency the user provides - via a venv, conda env, or their
# system Python - the same way as with the CMake build; this script doesn't
# create or manage one itself.
#
# Python lookup order:
#   1. $OP2_PYTHON            - explicit env override
#   2. $(command -v python3)  - system Python on PATH
#
# In all cases the chosen Python must already have jinja2, fparser, pcpp,
# sympy, and libclang importable, otherwise this script prints how to fix it
# and exits 1.

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

if [ -n "$OP2_PYTHON" ] && [ -x "$OP2_PYTHON" ]; then
    PY="$OP2_PYTHON"
else
    PY=$(command -v python3 || true)
fi

if [ -z "$PY" ]; then
    echo "op2-translator: no Python 3 found on PATH" >&2
    exit 1
fi

if ! "$PY" -c "import jinja2, fparser, pcpp, sympy; import clang.cindex" &> /dev/null; then
    cat >&2 <<EOF
op2-translator: $PY is missing required Python dependencies.

Options:
  * Install the deps directly into $PY:
        $PY -m pip install -r $SCRIPT_DIR/requirements.txt

  * Set OP2_PYTHON to a python3 whose environment already has the deps:
        OP2_PYTHON=/path/to/python3 make …
EOF
    exit 1
fi

exec "$PY" "$SCRIPT_DIR/op2-translator" "$@"
