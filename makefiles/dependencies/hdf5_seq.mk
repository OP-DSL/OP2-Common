ifdef HDF5_INSTALL_PATH
  HDF5_SEQ_INSTALL_PATH ?= $(HDF5_INSTALL_PATH)
endif

ifdef HDF5_SEQ_INSTALL_PATH
  HDF5_SEQ_INC_PATH := -I$(HDF5_SEQ_INSTALL_PATH)/include
  HDF5_SEQ_LIB_PATH := -L$(HDF5_SEQ_INSTALL_PATH)/lib -Wl,-rpath,$(HDF5_SEQ_INSTALL_PATH)/lib
endif

HDF5_SEQ_TEST = $(CONFIG_CXX) $(HDF5_SEQ_INC_PATH) \
                    $(DEPS_DIR)/tests/hdf5.cpp $(HDF5_SEQ_LIB_PATH) $(HDF5_SEQ_LINK) \
                    -o $(DEPS_DIR)/tests/hdf5

$(file > $(DEP_BUILD_LOG),$(HDF5_SEQ_TEST))
$(shell $(HDF5_SEQ_TEST) >> $(DEP_BUILD_LOG) 2>&1)

ifneq ($(.SHELLSTATUS),0)
  HDF5_SEQ_LINK ?= -lhdf5 -ldl -lm -lz

  $(file >> $(DEP_BUILD_LOG),$(HDF5_SEQ_TEST))
  $(shell $(HDF5_SEQ_TEST) >> $(DEP_BUILD_LOG) 2>&1)
endif

ifeq ($(.SHELLSTATUS),0)
  RESULT != $(DEPS_DIR)/tests/hdf5
  $(shell rm -f $(DEPS_DIR)/tests/hdf5)

  ifeq ($(RESULT),0)
    $(call info_bold,  > HDF5 (seq) libraries $(TEXT_FOUND) (link flags: $(or $(HDF5_SEQ_LINK),none)))

    CONFIG_HAVE_HDF5_SEQ := true

    CONFIG_HDF5_SEQ_INC := $(strip $(HDF5_SEQ_INC_PATH) $(HDF5_SEQ_DEF))
    CONFIG_HDF5_SEQ_LIB := $(strip $(HDF5_SEQ_LIB_PATH) $(HDF5_SEQ_LINK))
  else
    $(call info_bold,  > HDF5 (seq) libraries $(TEXT_NOTFOUND) \
      (test compilation succeeded but H5_HAVE_PARALLEL defined))
  endif
else
  $(call info_bold,  > HDF5 (seq) libraries $(TEXT_NOTFOUND):)
  $(info $(file < $(DEP_BUILD_LOG)))
  $(info )
endif
