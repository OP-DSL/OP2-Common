# --consts-module applies to every variant family here, not just SoA.
op2_add_functional_tests(
    LANGUAGE fortran
    SOURCES  const_constants.F90 const_kernels.F90 const_tests.F90
    TRANSLATOR_ARGS --consts-module const_constants.F90)
