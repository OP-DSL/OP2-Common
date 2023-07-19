ifdef CUDA_INSTALL_PATH
  CONFIG_NVCC := $(CUDA_INSTALL_PATH)/bin/nvcc
else
  CONFIG_NVCC := nvcc
endif

CUDA_VER_MATCH := "s/Cuda compilation tools, release \([[:digit:]]\+.[[:digit:]]\+\).*$$/\1/p"
NV_CUDA_VER := $(shell $(CONFIG_NVCC) --version | sed -n $(CUDA_VER_MATCH))

ifndef DEBUG
  NVCC_OPT := -O3 -use_fast_math
else
  NVCC_OPT := -g -O0 --device-debug
endif

CONFIG_NVCCFLAGS ?= $(foreach arch,$(CUDA_GEN),-gencode arch=compute_$(arch),code=sm_$(arch)) \
  -m64 -Xptxas=-v $(NVCC_OPT) $(EXTRA_NVCCFLAGS)
