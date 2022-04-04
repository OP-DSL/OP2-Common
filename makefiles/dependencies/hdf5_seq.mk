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

$(call info_bold,## Looking for the HDF5 (seq) libraries...)

$(call info_bold,### Testing presence of implicitly linked libraries)
$(info .   $(HDF5_SEQ_TEST))
$(shell $(HDF5_SEQ_TEST))

ifneq ($(.SHELLSTATUS),0)
  HDF5_SEQ_LINK ?= -lhdf5 -ldl -lm -lz

  $(call info_bold,### Testing presence of explicitly linked libraries)
  $(info .   $(HDF5_SEQ_TEST))
  $(shell $(HDF5_SEQ_TEST))
endif

ifeq ($(.SHELLSTATUS),0)
  RESULT != $(DEPS_DIR)/tests/hdf5
  $(shell rm -f $(DEPS_DIR)/tests/hdf5)

  $(call info_bold,### Found HDF5 is parallel: $(RESULT))

  ifeq ($(RESULT),0)
    $(call info_bold,## HDF5 (seq) libraries $(TEXT_FOUND))

    CONFIG_HAVE_HDF5_SEQ := true

    CONFIG_HDF5_SEQ_INC := $(strip $(HDF5_SEQ_INC_PATH) $(HDF5_SEQ_DEF))
    CONFIG_HDF5_SEQ_LIB := $(strip $(HDF5_SEQ_LIB_PATH) $(HDF5_SEQ_LINK))
  else
    $(call info_bold,## HDF5 (seq) libraries $(TEXT_NOTFOUND))
  endif
else
  $(call info_bold,## HDF5 (seq) libraries $(TEXT_NOTFOUND))
endif

$(info )
