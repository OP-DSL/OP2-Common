ifdef CUDA_INSTALL_PATH
  CUDA_INC_PATH := -I$(CUDA_INSTALL_PATH)/include
  CUDA_LIB_PATH := -L$(CUDA_INSTALL_PATH)/lib64 -L$(CUDA_INSTALL_PATH)/lib
endif

CUDA_TEST = $(CONFIG_CXX) $(CUDA_INC_PATH) \
                    $(DEPS_DIR)/tests/cuda.cpp $(CUDA_LIB_PATH) $(CUDA_LINK) \
                    -o $(DEPS_DIR)/tests/cuda $(DEP_DETECT_EXTRA)

$(shell $(CUDA_TEST))

ifneq ($(.SHELLSTATUS),0)
  CUDA_LINK ?= -lculibos -lcudart_static -lpthread -lrt -ldl
  $(shell $(CUDA_TEST))
endif

ifeq ($(.SHELLSTATUS),0)
  $(shell rm -f $(DEPS_DIR)/tests/cuda)

  CONFIG_HAVE_CUDA := true

  CONFIG_CUDA_INC := $(strip $(CUDA_INC_PATH) $(CUDA_DEF))
  CONFIG_CUDA_LIB := $(strip $(CUDA_LIB_PATH) $(CUDA_LINK))
endif
