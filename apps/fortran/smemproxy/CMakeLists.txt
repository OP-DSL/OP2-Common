# OP2 port of the atomics proxy's edge-scatter kernel.  This is the
# high-intensity indirect-increment case that motivates hierarchical
# shared-memory atomics, alongside Airfoil's lower-intensity res_calc.
#
# Mesh path comes from the PROXY_MESH environment variable; the maps live in
# the standalone proxy's maps/ directory.
#
# openmp is excluded for the same reason as smemtest: ncomp is a Fortran
# parameter rather than a literal, so the dats have runtime dimensions, and the
# Fortran openmp template mis-renders those.
op2_add_app_variants(
    NAME     smemproxy
    LANGUAGE fortran
    SOURCES  smemproxy_kernels.F90 smemproxy.F90
    EXCLUDE_VARIANTS mpi_* openmp)

# SoA is the layout the standalone proxy uses and the one that matters for
# performance, so it gets its own translated variants rather than a second
# build tree.
op2_add_app_variants(
    NAME     smemproxy_soa
    LANGUAGE fortran
    SOURCES  smemproxy_kernels.F90 smemproxy.F90
    TRANSLATOR_ONLY_ARGS --force_soa
    EXCLUDE_VARIANTS mpi_* openmp)
