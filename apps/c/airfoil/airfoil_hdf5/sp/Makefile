APP_NAME := airfoil
OP2_LIBS_WITH_HDF5 := true

VARIANT_FILTER := mpi_%

include ../../../../../makefiles/common.mk
include ../../../../../makefiles/c_app.mk

.PHONY: clean_convert_mesh

all: convert_mesh
clean: clean_convert_mesh

convert_mesh: convert_mesh.cpp
	$(MPICXX) $(CXXFLAGS) $(OP2_INC) $(HDF5_INC) $^ $(OP2_LIB_MPI) -o $@

clean_convert_mesh:
	-rm -f convert_mesh
