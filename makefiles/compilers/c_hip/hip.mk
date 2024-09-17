ifdef HIP_INSTALL_PATH
  CONFIG_HIPCC := $(HIP_INSTALL_PATH)/bin/hipcc
else
  CONFIG_HIPCC := hipcc
endif

ifndef DEBUG
  HIP_OPT := -g -Ofast
else
  HIP_OPT := -g -O0
endif

ifdef HIP_ARCH
  HIP_OPT += --offload-arch=$(HIP_ARCH)
endif

CONFIG_HIPCCFLAGS ?= -x hip -std=c++20 $(HIP_OPT)
