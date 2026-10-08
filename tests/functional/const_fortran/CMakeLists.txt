# --consts-module applies to every variant family here (not just SoA) -
# matches the legacy Makefile's APP_EXTRA_TRANSLATOR_FLAGS, which is set
# once before any include of f_app.mk and so applies uniformly.
op2_add_functional_tests(
    LANGUAGE fortran
    SOURCES  const_constants.F90 const_kernels.F90 const_tests.F90
    TRANSLATOR_ARGS --consts-module const_constants.F90)
