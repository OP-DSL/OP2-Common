#! PRE
OP2_COMPILER := gnu

# Link libm and disable math optimisations to prevent link errors with sqrt()
OMP_OFFLOAD_CPPFLAGS := -fopenmp -foffload=nvptx-none="-O3 -lm" -fno-fast-math -fno-associative-math
CPP_HAS_OMP_OFFLOAD := true

OMP_OFFLOAD_FFLAGS := $(OMP_OFFLOAD_CPPFLAGS)
F_HAS_OMP_OFFLOAD := true

#! POST
