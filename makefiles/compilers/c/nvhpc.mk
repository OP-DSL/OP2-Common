# Compiler executables and flags
CONFIG_CC := nvc
CONFIG_CXX := nvc++

ifndef DEBUG
  BASE_CXXFLAGS += -O3

  ifeq ($(TARGET_HOST),true)
    BASE_CXXFLAGS += -fast
  endif
else
  BASE_CXXFLAGS += -g -O0
endif

CONFIG_CFLAGS ?= -c99 $(BASE_CXXFLAGS) $(EXTRA_CFLAGS)
CONFIG_CXXFLAGS ?= $(BASE_CXXFLAGS) $(EXTRA_CXXFLAGS)

CONFIG_CXXLINK ?= -lstdc++

# Available OpenMP features
CONFIG_OMP_CXXFLAGS ?= -mp
CONFIG_CPP_HAS_OMP ?= true

GPU_FFLAG := -gpu=fastmath,ptxinfo,lineinfo
$(foreach arch,$(CUDA_GEN),$(eval GPU_FFLAG := $(GPU_FFLAG),cc$(arch)))

CONFIG_OMP_OFFLOAD_CXXFLAGS ?= -mp=gpu $(GPU_FFLAG)
CONFIG_CPP_HAS_OMP_OFFLOAD ?= true
