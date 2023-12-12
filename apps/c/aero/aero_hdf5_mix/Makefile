APP_NAME := aero

APP_ENTRY := $(APP_NAME).cpp
APP_ENTRY_MPI := $(APP_ENTRY)

OP2_LIBS_WITH_HDF5 := true

VARIANT_FILTER_OUT := %vec

include ../../../../makefiles/common.mk
include ../../../../makefiles/c_app.mk

.PHONY: clean_write_hdf5

ifeq ($(HAVE_HDF_PAR),true)
  all: write_hdf5
endif

clean: clean_write_hdf5

write_hdf5: write_hdf5.cpp
	$(MPICXX) $(CXXFLAGS) $(OP2_INC) $(HDF5_PAR_INC) $^ $(OP2_LIB_MPI) -o $@

clean_write_hdf5:
	-rm -f write_hdf5
