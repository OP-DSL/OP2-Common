ifdef HDF5_INSTALL_PATH
  HDF5_SEQ_INSTALL_PATH ?= $(HDF5_INSTALL_PATH)
endif

ifdef HDF5_SEQ_INSTALL_PATH
  HDF5_SEQ_INC_PATH := -I$(HDF5_SEQ_INSTALL_PATH)/include
  HDF5_SEQ_LIB_PATH := -L$(HDF5_SEQ_INSTALL_PATH)/lib -Wl,-rpath,$(HDF5_SEQ_INSTALL_PATH)/lib
endif

HDF5_SEQ_TEST = $(CXX) $(HDF5_SEQ_INC_PATH) \
                    $(DEPS_DIR)/tests/hdf5.cpp $(HDF5_SEQ_LIB_PATH) $(HDF5_SEQ_LINK) \
                    -o $(DEPS_DIR)/tests/hdf5 $(DEP_DETECT_EXTRA)

# Test for sequential HDF5
$(shell $(HDF5_SEQ_TEST))

ifneq ($(.SHELLSTATUS),0)
  HDF5_SEQ_LINK ?= -lhdf5 -ldl -lm -lz
  $(shell $(HDF5_SEQ_TEST))
endif

ifeq ($(.SHELLSTATUS),0)
  RESULT != $(DEPS_DIR)/tests/hdf5
  $(shell rm -f $(DEPS_DIR)/tests/hdf5)

  ifeq ($(RESULT),0)
    HAVE_HDF5_SEQ := true

    HDF5_SEQ_INC := $(strip $(HDF5_SEQ_INC_PATH) $(HDF5_SEQ_DEF))
    HDF5_SEQ_LIB := $(strip $(HDF5_SEQ_LIB_PATH) $(HDF5_SEQ_LINK))
  endif
endif
