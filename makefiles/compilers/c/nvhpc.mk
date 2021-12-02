# Compiler executables and flags
CC ?= nvcc
CXX ?= nvc++

BASE_CPPFLAGS =

ifndef DEBUG
  BASE_CPPFLAGS += -O3
else
  BASE_CPPFLAGS += -g -O0
endif

CFLAGS ?= -c99 $(BASE_CPPFLAGS)
CXXFLAGS ?= $(BASE_CPPFLAGS)

CXXLINK ?= -lstdc++

# Available OpenMP features
OMP_CPPFLAGS ?= -mp
CPP_HAS_OMP ?= true

ifeq ($(NV_ARCH),Fermi)
  CPP_CUDA_GEN = cc20
else
ifeq ($(NV_ARCH),Kepler)
  CPP_CUDA_GEN = cc35
else
ifeq ($(NV_ARCH),Maxwell)
  CPP_CUDA_GEN = cc50
else
ifeq ($(NV_ARCH),Pascal)
  CPP_CUDA_GEN = cc60
else
ifeq ($(NV_ARCH),Volta)
  CPP_CUDA_GEN = cc70
endif
endif
endif
endif
endif

OMP_OFFLOAD_CPPFLAGS ?= -mp=gpu -gpu=$(CPP_CUDA_GEN)
CPP_HAS_OMP_OFFLOAD ?= true
