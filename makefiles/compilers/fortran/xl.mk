# Compiler executables and flags
CONFIG_FC := xlf_r

ifndef DEBUG
  ifeq ($(TARGET_HOST),true)
    BASE_FFLAGS += -O3
  else
    BASE_FFLAGS += -O4
  endif
else
  BASE_FFLAGS += -g -O0
endif

CONFIG_FFLAGS ?= $(BASE_FFLAGS)
CONFIG_F_MOD_OUT_OPT ?= -qmoddir=
CONFIG_F_HAS_PARALLEL_BUILDS ?= true

# Available OpenMP features
CONFIG_OMP_FFLAGS ?= -qsmp=omp
CONFIG_F_HAS_OMP ?= true

# CONFIG_OMP_OFFLOAD_FFLAGS ?=
CONFIG_F_HAS_OMP_OFFLOAD ?= false

# Available CUDA features
# CONFIG_CUDA_FFLAGS ?=
CONFIG_F_HAS_CUDA ?= false
