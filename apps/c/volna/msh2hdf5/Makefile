#
# The following environment variables should be predefined:
#
# HDF5_INSTALL_PATH
# LIBMESH_INSTALL_PATH

#
# set paths for header files and libraries
#

LIBMESH_INC = -I$(LIBMESH_INSTALL_PATH)/include
LIBMESH_LIB = -L$(LIBMESH_INSTALL_PATH)/lib

OP2_INC = -I$(OP2_INSTALL_PATH)/c/include
OP2_LIB = -L$(OP2_INSTALL_PATH)/c/lib

#HDF5_INC = -I$(HDF5_INSTALL_PATH)/include
#HDF5_LIB = -L$(HDF5_INSTALL_PATH)/lib -lhdf5 -lz

#LD_LIBRARY_PATH = $(LD_LIBRARY_PATH):/home/endre/workspace/libmesh/lib/x86_64-unknown-linux-gnu_opt

OPENMPI_INC = -I/home/gihan/openmpi-intel/include

include $(LIBMESH_INSTALL_PATH)/Make.common

ifeq ($(OP2_COMPILER),gnu)
	CPP = g++
	CPPFLAGS = -fPIC -DUNIX -Wall
	#CPPFLAGS = -g -fPIC -DUNIX -Wall
	OMPFLAGS = -fopenmp
	MPICPP = /usr/bin/mpiCC
	MPIFLAGS = $(CCFLAGS)
else
ifeq ($(OP2_COMPILER),intel)
	CPP = icpc
	CCFLAGS = -g -O3 -vec-report -xSSE4.2 -parallel	
	#CCFLAGS = -g -vec-report -xSSE4.2 -parallel	
	CPPFLAGS = $(CCFLAGS)
	OMPFLAGS = -openmp -openmp-report2
	MPICPP = mpiCC
	MPIFLAGS = $(CPPFLAGS)
else
print: 
		@echo "unrecognized value for OP2_COMPILER"
endif
endif


#
# master to make all versions
#

all: clean msh2hdf5 

msh2hdf5: msh2hdf5.cpp Makefile
	$(CPP) $(CPPFLAGS) msh2hdf5.cpp $(libmesh_INCLUDE) $(OP2_INC) $(OPENMPI_INC) $(libmesh_LIBS) $(OP2_LIB) -lop2_seq -lop2_hdf5 -o msh2hdf5

	  
#
# cleanup
#

clean:
		rm -f msh2hdf5 *.o

