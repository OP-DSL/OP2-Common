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

$(info ## Looking for the HDF5 (parallel) libraries...)

$(info ### Testing presence of implicitly linked libraries)
$(info .   $(HDF5_PAR_TEST))
$(shell $(HDF5_PAR_TEST))

ifneq ($(.SHELLSTATUS),0)
  HDF5_PAR_LINK ?= -lhdf5 -ldl -lm -lz

  $(info ### Testing presence of explicitly linked libraries)
  $(info .   $(HDF5_PAR_TEST))
  $(shell $(HDF5_PAR_TEST))
endif

ifeq ($(.SHELLSTATUS),0)
  RESULT != $(DEPS_DIR)/tests/hdf5
  $(shell rm -f $(DEPS_DIR)/tests/hdf5)

  $(info ### Found HDF5 is parallel: $(RESULT))

  ifeq ($(RESULT),1)
    $(info ## HDF5 (parallel) libraries $(TEXT_FOUND))

    CONFIG_HAVE_HDF5_PAR := true

    CONFIG_HDF5_PAR_INC := $(strip $(HDF5_PAR_INC_PATH) $(HDF5_PAR_DEF))
    CONFIG_HDF5_PAR_LIB := $(strip $(HDF5_PAR_LIB_PATH) $(HDF5_PAR_LINK))
  else
    $(info ## HDF5 (parallel) libraries $(TEXT_NOTFOUND))
  endif
else
  $(info ## HDF5 (parallel) libraries $(TEXT_NOTFOUND))
endif

$(info )
