ifdef CUDA_INSTALL_PATH
  NVCC := $(CUDA_INSTALL_PATH)/bin/nvcc
else
  NVCC := nvcc
endif

ifndef DEBUG
  NVCC_OPT := -O3 -use_fast_math
else
  NVCC_OPT := -g -O0
endif

NVCCFLAGS ?= $(foreach arch,$(CUDA_GEN),-gencode arch=compute_$(arch),code=sm_$(arch)) \
			 -m64 -Xptxas=-v $(NVCC_OPT)
