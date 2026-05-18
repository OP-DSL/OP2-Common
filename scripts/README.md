### Scripts
This directory contains various scripts used for testing the build and correct execution of OP2 and its example applications. These are provided and committed to the repository simply as a reference to anyone who is installing and using OP2.


#### Files
 * `source_intel`, `source_gnu_demos`, ... : example files that can be sourced to set the required environment variables to compile and install with the respective compiler suites.
 * `numawrap` -- NUMA-aware MPI process wrapper for GPU clusters. Detects the MPI local rank (supports Open MPI, MVAPICH2, Hydra, and MPISPAWN), sets `CUDA_VISIBLE_DEVICES` to the local rank, and binds the process to the corresponding NUMA node via `numactl`. Usage: `mpirun -np N numawrap ./application`
 * `test_makefiles.sh` -- builds and tests OP2 lib and apps with plain Makefiles (including Fortran libs and apps)
 * `pre-commit.sh`, `pre-commit-check_formatting.sh`, `pre-commit-white_spaces.sh` -- git pre-commit hooks for formatting and whitespace checks. See the repository root for setup instructions.
