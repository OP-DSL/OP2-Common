include ../../../../makefiles/common.mk

APP_NAME := aero
APP_SRC := aero.cpp

OP2_LIBS_WITH_HDF5 := true

include ../../../../makefiles/c_app.mk

.PHONY: clean_write_hdf5

ifeq ($(HAVE_HDF5_PAR),true)
  all: write_hdf5
endif

clean: clean_write_hdf5

write_hdf5: write_hdf5.cpp
	$(MPICXX) $(CXXFLAGS) $(OP2_INC) $(HDF5_PAR_INC) $^ $(OP2_LIB_MPI) -o $@

clean_write_hdf5:
	-$(RM) -f write_hdf5
