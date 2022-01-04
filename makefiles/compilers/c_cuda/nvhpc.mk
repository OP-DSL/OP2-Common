ifdef CUDA_INSTALL_PATH
  NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc
else
  NVCC ?= nvcc
endif

ifeq ($(NV_ARCH),Fermi)
  NVCC_GEN = -gencode arch=compute_20,code=sm_21
else
ifeq ($(NV_ARCH),Kepler)
  NVCC_GEN = -gencode arch=compute_35,code=sm_35
else
ifeq ($(NV_ARCH),Maxwell)
  NVCC_GEN = -gencode arch=compute_50,code=sm_50
else
ifeq ($(NV_ARCH),Pascal)
  NVCC_GEN = -gencode arch=compute_60,code=sm_60
else
ifeq ($(NV_ARCH),Volta)
  NVCC_GEN = -gencode arch=compute_70,code=sm_70
endif
endif
endif
endif
endif

ifndef DEBUG
  NVCC_OPT = -O3 -use_fast_math
else
  NVCC_OPT = -g -O0
endif

NVCCFLAGS ?= $(NVCC_GEN) -m64 -Xptxas=-v $(NVCC_OPT)
