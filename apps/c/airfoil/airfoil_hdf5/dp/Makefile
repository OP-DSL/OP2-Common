include ../../../../../makefiles/common.mk

APP_NAME := airfoil
APP_SRC := airfoil.cpp

OP2_LIBS_WITH_HDF5 := true

include ../../../../../makefiles/c_app.mk

.PHONY: clean_convert_mesh

ifeq ($(HAVE_HDF5_SEQ),true)
  all: convert_mesh
endif

ifeq ($(HAVE_HDF5_PAR),true)
  all: convert_mesh_mpi
endif

clean: clean_convert_mesh

convert_mesh: convert_mesh.cpp
	$(CXX) $(CXXFLAGS) $(OP2_INC) $(HDF5_SEQ_INC) $^ $(OP2_LIB_SEQ) -o $@

convert_mesh_mpi: convert_mesh_mpi.cpp
	$(MPICXX) $(CXXFLAGS) $(OP2_INC) $(HDF5_PAR_INC) $^ $(OP2_LIB_MPI) -o $@

clean_convert_mesh:
	-rm -f convert_mesh
	-rm -f convert_mesh_mpi
