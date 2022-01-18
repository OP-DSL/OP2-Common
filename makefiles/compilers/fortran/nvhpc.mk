# Compiler executables and flags
FC := nvfortran

BASE_FFLAGS :=

ifndef DEBUG
  BASE_FFLAGS += -O3
else
  BASE_FFLAGS += -g -O0
endif

FFLAGS ?= $(BASE_FFLAGS)
F_MOD_OUT_OPT ?= -module #

# NVFORTRAN and parallel builds do not mix well...
F_HAS_PARALLEL_BUILDS ?= false

GPU_FFLAG := -gpu=fastmath,ptxinfo,lineinfo
$(foreach arch,$(CUDA_GEN),$(eval GPU_FFLAG := $(GPU_FFLAG),cc$(arch)))

# Available OpenMP features
OMP_FFLAGS ?= -mp
F_HAS_OMP ?= true

OMP_OFFLOAD_FFLAGS ?= -mp=gpu $(GPU_FFLAG)
F_HAS_OMP_OFFLOAD ?= true

CUDA_FFLAGS ?= $(GPU_FFLAG)
F_HAS_CUDA ?= true
