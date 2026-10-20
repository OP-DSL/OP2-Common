# Lazy execution and runtime kernel fusion: nine calls a sweep with no fetch
# between them, checked against a host reference.  The test source does
# `use mpi` under USE_MPI - see the MPI Fortran search in ../CMakeLists.txt.
# Each fused kernel compiles on its first call, so the fused and apart runs
# both happen.
op2_add_functional_tests(
    LANGUAGE fortran
    SOURCES  fusion.F90
    EXCLUDE_VARIANTS ${OP2_TEST_MPI_FORTRAN_EXCLUDE}
    LINK_LIBRARIES   ${OP2_TEST_MPI_FORTRAN_LINK}
    ENVIRONMENT OP_JIT_MIN_CALLS=1 OP_JIT_SEQ_COMPILE=1
    STRATEGIES       atomics colouring)
