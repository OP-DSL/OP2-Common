# Compiler executables and flags
FC ?= nvfortran

BASE_FFLAGS =

ifndef DEBUG
  BASE_FFLAGS += -O3
else
  BASE_FFLAGS += -g -O0
endif

FFLAGS ?= $(BASE_FFLAGS)
F_MOD_OUT_OPT ?= -module #

# NVFORTRAN and parallel builds do not mix well...
F_HAS_PARALLEL_BUILDS ?= false

# Available OpenMP features
OMP_FFLAGS ?= -mp
F_HAS_OMP ?= true

OMP_OFFLOAD_FFLAGS ?=
F_HAS_OMP_OFFLOAD ?= false

# Available CUDA features
ifeq ($(NV_ARCH),Fermi)
  FC_CUDA_GEN = cc20
else
ifeq ($(NV_ARCH),Kepler)
  FC_CUDA_GEN = cc35
else
ifeq ($(NV_ARCH),Maxwell)
  FC_CUDA_GEN = cc50
else
ifeq ($(NV_ARCH),Pascal)
  FC_CUDA_GEN = cc60
else
ifeq ($(NV_ARCH),Volta)
  FC_CUDA_GEN = cc70
endif
endif
endif
endif
endif

CUDA_FFLAGS ?= -Mcuda=$(FC_CUDA_GEN),fastmath,ptxinfo,lineinfo
F_HAS_CUDA ?= true
