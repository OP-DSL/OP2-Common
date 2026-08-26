# Mixed-type indirect-increment test used by the hierarchical atomics work.
#
# Only c_cuda/c_hip (and their JIT paths) implement staged shared-memory
# accumulation, so openmp is excluded: it cannot exercise this feature.  seq and
# genseq are kept as the reference results the c_cuda runs are checked against.
#
# Excluding openmp also sidesteps a pre-existing bug in the Fortran openmp
# template, which this app is the first to hit because it declares a dat with a
# runtime-valued dimension: dat_dim_w emits dat<dat.id>_dim while the parameter
# is declared dat<dat.arg_id>_dim, and the vectorised local array is sized from
# dat.dim directly, rendering a literal None.
op2_add_app_variants(
    NAME     smemtest
    LANGUAGE fortran
    SOURCES  smemtest.F90
    EXCLUDE_VARIANTS mpi_* openmp)

# SoA is the normal performance layout, and the staged wrapper indexes target
# dats differently under it, so both layouts are required correctness runs.
# Layout is fixed at translation, not by the runtime OP_AUTO_SOA variable, so it
# needs a separately translated set of binaries rather than a second build tree.
op2_add_app_variants(
    NAME     smemtest_soa
    LANGUAGE fortran
    SOURCES  smemtest.F90
    TRANSLATOR_ONLY_ARGS --force_soa
    EXCLUDE_VARIANTS mpi_* openmp)
