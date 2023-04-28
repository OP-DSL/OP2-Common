include $(MAKEFILES_DIR)/compilers/c/gnu.mk

CONFIG_CC := clang
CONFIG_CXX := clang++

CONFIG_CXXLINK ?= -lc++

CONFIG_CPP_HAS_OMP_OFFLOAD ?= true
OMP_OFFLOAD_CXXFLAGS = -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a
CONFIG_OMP_OFFLOAD_CXXFLAGS = -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a
