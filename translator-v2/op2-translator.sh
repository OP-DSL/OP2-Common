#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

make --silent -C ${SCRIPT_DIR} python
${SCRIPT_DIR}/.python/bin/python3 ${SCRIPT_DIR}/op2-translator $@
