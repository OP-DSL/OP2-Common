# A strong-scaling proxy for Hydra-like codes: many small loops per
# iteration over nodes, edges and boundary sets, on a generated box mesh that
# each rank builds as its own slab (see hydra_proxy.F90).
op2_add_app_variants(
    NAME     hydra_proxy
    LANGUAGE fortran
    SOURCES  hydra_proxy_kernels.F90 hydra_proxy.F90)
