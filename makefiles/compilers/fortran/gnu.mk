# Compiler executables and flags
CONFIG_FC := gfortran

BASE_FFLAGS := -Wall -pedantic -ffixed-line-length-none -ffree-line-length-none -fcray-pointer

ifndef DEBUG
  BASE_FFLAGS += -g -O3

  ifeq ($(TARGET_HOST),true)
    BASE_FFLAGS += -march=native
  endif
else
  BASE_FFLAGS += -g -Og -fcheck=all -ffpe-trap=invalid,zero,overflow
endif

CONFIG_FFLAGS ?= $(BASE_FFLAGS)
CONFIG_F_MOD_OUT_OPT ?= -J
CONFIG_F_HAS_PARALLEL_BUILDS ?= true

# Available OpenMP features
CONFIG_OMP_FFLAGS ?= -fopenmp
CONFIG_F_HAS_OMP ?= true

# CONFIG_OMP_OFFLOAD_FFLAGS ?=
CONFIG_F_HAS_OMP_OFFLOAD ?= false

# Available CUDA features
# CONFIG_CUDA_FFLAGS ?=
CONFIG_F_HAS_CUDA ?= false
