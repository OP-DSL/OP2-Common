# Compiler executables and flags
CONFIG_FC := ftn
CONFIG_MPIFC := ftn

ifndef DEBUG
  BASE_FFLAGS += -O3
else
  BASE_FFLAGS += -g -Og
endif

CONFIG_FFLAGS ?= $(BASE_FFLAGS) $(EXTRA_FFLAGS)
CONFIG_F_MOD_OUT_OPT ?= -em -J
CONFIG_F_HAS_PARALLEL_BUILDS ?= true

# Available OpenMP features
CONFIG_OMP_FFLAGS ?= -fopenmp
CONFIG_F_HAS_OMP ?= true

# CONFIG_OMP_OFFLOAD_FFLAGS ?=
CONFIG_F_HAS_OMP_OFFLOAD ?= false

# Available CUDA features
# CONFIG_CUDA_FFLAGS ?=
CONFIG_F_HAS_CUDA ?= false
