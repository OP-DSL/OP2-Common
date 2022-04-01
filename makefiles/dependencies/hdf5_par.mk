ifdef HDF5_INSTALL_PATH
  HDF5_PAR_INSTALL_PATH ?= $(HDF5_INSTALL_PATH)
endif

ifdef HDF5_PAR_INSTALL_PATH
  HDF5_PAR_INC_PATH := -I$(HDF5_PAR_INSTALL_PATH)/include
  HDF5_PAR_LIB_PATH := -L$(HDF5_PAR_INSTALL_PATH)/lib -Wl,-rpath,$(HDF5_PAR_INSTALL_PATH)/lib
endif

HDF5_PAR_TEST = $(CONFIG_MPICXX) $(HDF5_PAR_INC_PATH) \
                    $(DEPS_DIR)/tests/hdf5.cpp $(HDF5_PAR_LIB_PATH) $(HDF5_PAR_LINK) \
                    -o $(DEPS_DIR)/tests/hdf5 $(DEP_DETECT_EXTRA)

# Test for parallel HDF5
$(shell $(HDF5_PAR_TEST))

ifneq ($(.SHELLSTATUS),0)
  HDF5_PAR_LINK ?= -lhdf5 -ldl -lm -lz
  $(shell $(HDF5_PAR_TEST))
endif

ifeq ($(.SHELLSTATUS),0)
  RESULT != $(DEPS_DIR)/tests/hdf5
  $(shell rm -f $(DEPS_DIR)/tests/hdf5)

  ifeq ($(RESULT),1)
    CONFIG_HAVE_HDF5_PAR := true

    CONFIG_HDF5_PAR_INC := $(strip $(HDF5_PAR_INC_PATH) $(HDF5_PAR_DEF))
    CONFIG_HDF5_PAR_LIB := $(strip $(HDF5_PAR_LIB_PATH) $(HDF5_PAR_LINK))
  endif
endif
