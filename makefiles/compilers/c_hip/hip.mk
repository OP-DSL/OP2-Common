CONFIG_HIP ?= hipcc

ifndef DEBUG
  HIP_OPT := -Ofast
else
  HIP_OPT := -g -O0
endif

CONFIG_HIPFLAGS ?= -x hip --offload-arch=$(HIP_ARCH) $(HIP_OPT)
