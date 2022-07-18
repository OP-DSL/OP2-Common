# Compiler executables and flags
CONFIG_FC := nvfortran

ifndef DEBUG
  BASE_FFLAGS += -O3

  ifeq ($(TARGET_HOST),true)
    BASE_FFLAGS += -fast
  endif
else
  BASE_FFLAGS += -g -O0
endif

CONFIG_FFLAGS ?= $(BASE_FFLAGS)
CONFIG_F_MOD_OUT_OPT ?= -module #

# NVFORTRAN and parallel builds do not mix well...
CONFIG_F_HAS_PARALLEL_BUILDS ?= false

GPU_FFLAG := -gpu=fastmath,ptxinfo,lineinfo
$(foreach arch,$(CUDA_GEN),$(eval GPU_FFLAG := $(GPU_FFLAG),cc$(arch)))

# Available OpenMP features
CONFIG_OMP_FFLAGS ?= -mp
CONFIG_F_HAS_OMP ?= true

CONFIG_OMP_OFFLOAD_FFLAGS ?= -mp=gpu $(GPU_FFLAG)
CONFIG_F_HAS_OMP_OFFLOAD ?= true

CONFIG_CUDA_FFLAGS ?= -cuda $(GPU_FFLAG)
CONFIG_F_HAS_CUDA ?= true
