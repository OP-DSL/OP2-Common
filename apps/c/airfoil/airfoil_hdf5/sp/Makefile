include ../../../../../makefiles/common.mk

APP_NAME := airfoil
APP_SRC := airfoil.cpp

OP2_LIBS_WITH_HDF5 := true

include ../../../../../makefiles/c_app.mk

.PHONY: clean_convert_mesh

ifeq ($(HAVE_HDF5_PAR),true)
  all: convert_mesh
endif

clean: clean_convert_mesh

convert_mesh: convert_mesh.cpp
	$(MPICXX) $(CXXFLAGS) $(OP2_INC) $(HDF5_PAR_INC) $^ $(OP2_LIB_MPI) -o $@

clean_convert_mesh:
	-rm -f convert_mesh
