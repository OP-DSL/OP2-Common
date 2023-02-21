# Compiler executables and flags
CONFIG_CC := nvcc
CONFIG_CXX := nvc++

ifndef DEBUG
  BASE_CPPFLAGS += -O3
else
  BASE_CPPFLAGS += -g -O0
endif

CONFIG_CFLAGS ?= -c99 $(BASE_CPPFLAGS)
CONFIG_CXXFLAGS ?= $(BASE_CPPFLAGS)

CONFIG_CXXLINK ?= -lstdc++

# Available OpenMP features
CONFIG_OMP_CPPFLAGS ?= -mp
CONFIG_CPP_HAS_OMP ?= true

GPU_FFLAG := -gpu=fastmath,ptxinfo,lineinfo
$(foreach arch,$(CUDA_GEN),$(eval GPU_FFLAG := $(GPU_FFLAG),cc$(arch)))

CONFIG_OMP_OFFLOAD_CPPFLAGS ?= -mp=gpu $(GPU_FFLAG)
CONFIG_CPP_HAS_OMP_OFFLOAD ?= true
