# Lazy execution and runtime kernel fusion: nine calls a sweep with no fetch
# between them, checked against a host reference.  Serial only: the
# reference indexes the global numbering.  Each fused kernel compiles on
# its first call, so the fused and apart runs both happen.
op2_add_functional_tests(
    LANGUAGE fortran
    SOURCES  fusion.F90
    EXCLUDE_VARIANTS mpi_*
    ENVIRONMENT OP_JIT_MIN_CALLS=1 OP_JIT_SEQ_COMPILE=1
    STRATEGIES       atomics colouring)
