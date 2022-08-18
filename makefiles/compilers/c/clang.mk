include $(MAKEFILES_DIR)/compilers/c/gnu.mk

CONFIG_CC := clang
CONFIG_CXX := clang++

CONFIG_CXXLINK ?= -lc++
CONFIG_CPP_HAS_OMP_OFFLOAD ?= true
OMP_OFFLOAD_CXXFLAGS = -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -ffp-contract=fast -Xcuda-ptxas -v -Xclang -target-feature -Xclang +ptx70 
