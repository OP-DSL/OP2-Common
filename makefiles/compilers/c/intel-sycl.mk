include $(MAKEFILES_DIR)/compilers/c/gnu.mk

CONFIG_CC := icx
CONFIG_CXX := icpx

ifdef SYCL_INSTALL_PATH
  CONFIG_SYCLCC := $(SYCL_INSTALL_PATH)/bin/icpx
else
  CONFIG_SYCLCC := icpx
endif

ifndef DEBUG
  SYCL_OPT := -g -O3
else
  SYCL_OPT := -g -O0
endif

CONFIG_SYCLCCFLAGS ?= -fsycl -std=c++20 $(SYCL_OPT) $(SYCL_FLAGS)