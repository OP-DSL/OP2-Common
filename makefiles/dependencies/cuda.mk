ifdef CUDA_INSTALL_PATH
  CUDA_INC_PATH := -I$(CUDA_INSTALL_PATH)/include
  CUDA_LIB_PATH := -L$(CUDA_INSTALL_PATH)/lib64 -L$(CUDA_INSTALL_PATH)/lib
endif

CUDA_TEST = $(CXX) $(CUDA_INC_PATH) \
                    $(DEPS_DIR)/tests/cuda.cpp $(CUDA_LIB_PATH) $(CUDA_LINK) \
                    -o $(DEPS_DIR)/tests/cuda 2> /dev/null

$(shell $(CUDA_TEST))

ifneq ($(.SHELLSTATUS),0)
  CUDA_LINK ?= -lculibos -lcudart_static -lpthread -lrt -ldl
  $(shell $(CUDA_TEST))
endif

ifeq ($(.SHELLSTATUS),0)
  $(shell rm -f $(DEPS_DIR)/tests/cuda)

  HAVE_CUDA := true

  CUDA_INC := $(strip $(CUDA_INC_PATH) $(CUDA_DEF))
  CUDA_LIB := $(strip $(CUDA_LIB_PATH) $(CUDA_LINK))
endif
