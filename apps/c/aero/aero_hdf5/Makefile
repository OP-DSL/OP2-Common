APP_NAME := aero
OP2_LIBS_WITH_HDF5 := true

VARIANT_FILTER := mpi_%
VARIANT_FILTER_OUT := %vec

include ../../../../makefiles/common.mk
include ../../../../makefiles/c_app.mk

.PHONY: clean_write_hdf5

all: write_hdf5
clean: clean_write_hdf5

write_hdf5: write_hdf5.cpp
	$(MPICXX) $(CXXFLAGS) $(OP2_INC) $(HDF5_INC) $^ $(OP2_LIB_MPI) -o $@

clean_write_hdf5:
	-rm -f write_hdf5
