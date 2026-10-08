#!/usr/bin/env bash
# setup_deps.sh - Bootstrap OP2 external dependencies.
#
# Thin wrapper around cmake/deps/CMakeLists.txt, which downloads and builds
# HDF5, ParMETIS, PT-Scotch, and KaHIP into <repo>/deps/.
#
# USAGE
#   scripts/setup_deps.sh [OPTIONS] [cmake -D flags...]
#
# OPTIONS
#   -j N, -jN            Parallel build jobs (default: nproc)
#   -h, --help           Show this message
#
# CMAKE FLAGS (passed through to the deps cmake project)
#   -DCMAKE_C_COMPILER=gcc        Set C compiler (also honoured from CC env var)
#   -DCMAKE_CXX_COMPILER=g++      Set C++ compiler (also honoured from CXX env var)
#   -DCMAKE_Fortran_COMPILER=...  Set Fortran compiler (also honoured from FC env var)
#   -DMPI_C_COMPILER=mpicc        Override MPI C wrapper (auto-detected by default)
#   -DMPI_CXX_COMPILER=mpicxx     Override MPI C++ wrapper (auto-detected by default)
#   -DOP2_DEPS_HDF5=OFF           Skip HDF5
#   -DOP2_DEPS_PARMETIS=OFF       Skip ParMETIS + METIS
#   -DOP2_DEPS_PTSCOTCH=OFF       Skip PT-Scotch
#   -DOP2_DEPS_KAHIP=OFF          Skip KaHIP
#   -DOP2_DEPS_HDF5_VERSION=X.Y.Z-N  Override version
#   --print-cmake-flags           Print the -C flag for the built deps and exit
#
# After a successful build, configure the OP2 CMake project with:
#   cmake -B build -C deps/op2-deps.cmake
#
# Or set OP2_DEPS_ROOT manually:
#   cmake -B build -DOP2_DEPS_ROOT=<repo>/deps

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPS_BUILD="${REPO_ROOT}/deps/build"
DEPS_SOURCE="${REPO_ROOT}/cmake/deps"

JOBS=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
CMAKE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -j)               JOBS="$2"; shift 2 ;;
        -j*)              JOBS="${1#-j}"; shift ;;
        --print-cmake-flags)
            if [[ -f "${REPO_ROOT}/deps/op2-deps.cmake" ]]; then
                echo "-C ${REPO_ROOT}/deps/op2-deps.cmake"
            else
                echo "No deps/op2-deps.cmake found - run setup_deps.sh first." >&2
                exit 1
            fi
            exit 0
            ;;
        -h|--help)
            sed -n '2,/^set -/{ /^set -/d; s/^# \?//p }' "$0"
            exit 0
            ;;
        *) CMAKE_ARGS+=("$1"); shift ;;
    esac
done

cmake -B "${DEPS_BUILD}" -S "${DEPS_SOURCE}" \
    -DPARALLEL_LEVEL="${JOBS}" \
    "${CMAKE_ARGS[@]}"

cmake --build "${DEPS_BUILD}" -j"${JOBS}"
