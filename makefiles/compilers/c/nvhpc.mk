# Compiler executables and flags
CC := nvcc
CXX := nvc++

BASE_CPPFLAGS :=

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

GPU_FFLAG := -gpu=fastmath,ptxinfo,lineinfo
$(foreach arch,$(CUDA_GEN),$(eval GPU_FFLAG := $(GPU_FFLAG),cc$(arch)))

OMP_OFFLOAD_CPPFLAGS ?= -mp=gpu $(GPU_FFLAG)
CPP_HAS_OMP_OFFLOAD ?= true
