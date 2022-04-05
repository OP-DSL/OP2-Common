ifdef HDF5_INSTALL_PATH
  HDF5_PAR_INSTALL_PATH ?= $(HDF5_INSTALL_PATH)
endif

ifdef HDF5_PAR_INSTALL_PATH
  HDF5_PAR_INC_PATH := -I$(HDF5_PAR_INSTALL_PATH)/include
  HDF5_PAR_LIB_PATH := -L$(HDF5_PAR_INSTALL_PATH)/lib -Wl,-rpath,$(HDF5_PAR_INSTALL_PATH)/lib
endif

HDF5_PAR_TEST = $(CONFIG_MPICXX) $(HDF5_PAR_INC_PATH) \
                    $(DEPS_DIR)/tests/hdf5.cpp $(HDF5_PAR_LIB_PATH) $(HDF5_PAR_LINK) \
                    -o $(DEPS_DIR)/tests/hdf5

$(file > $(DEP_BUILD_LOG),$(HDF5_PAR_TEST))
$(shell $(HDF5_PAR_TEST) >> $(DEP_BUILD_LOG) 2>&1)

ifneq ($(.SHELLSTATUS),0)
  HDF5_PAR_LINK ?= -lhdf5 -ldl -lm -lz

  $(file >> $(DEP_BUILD_LOG),$(HDF5_PAR_TEST))
  $(shell $(HDF5_PAR_TEST) >> $(DEP_BUILD_LOG) 2>&1)
endif

ifeq ($(.SHELLSTATUS),0)
  RESULT != $(DEPS_DIR)/tests/hdf5
  $(shell rm -f $(DEPS_DIR)/tests/hdf5)

  ifeq ($(RESULT),1)
    $(call info_bold,  > HDF5 (parallel) libraries $(TEXT_FOUND) (link flags: $(or $(HDF5_PAR_LINK),none)))

    CONFIG_HAVE_HDF5_PAR := true

    CONFIG_HDF5_PAR_INC := $(strip $(HDF5_PAR_INC_PATH) $(HDF5_PAR_DEF))
    CONFIG_HDF5_PAR_LIB := $(strip $(HDF5_PAR_LIB_PATH) $(HDF5_PAR_LINK))
  else
    $(call info_bold,  > HDF5 (parallel) libraries $(TEXT_NOTFOUND) \
      (test compilation succeeded but H5_HAVE_PARALLEL undefined))
  endif
else
  $(call info_bold,  > HDF5 (parallel) libraries $(TEXT_NOTFOUND):)
  $(info $(file < $(DEP_BUILD_LOG)))
  $(info )
endif
