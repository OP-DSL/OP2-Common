#
# The following environment variables should be predefined:
#
# OP2_INSTALL_PATH
# HDF5_INSTALL_PATH
# MPI_INSTALL_PATH
#

#
# set paths for header files and libraries
#

LIBMESH_INC = -I$(LIBMESH_INSTALL_PATH)/include
LIBMESH_LIB = -L$(LIBMESH_INSTALL_PATH)/lib

OP2_INC = -I$(OP2_INSTALL_PATH)/c/include
OP2_LIB = -L$(OP2_INSTALL_PATH)/c/lib

#VOLNA_INC = -I$(VOLNA_INSTALL_PATH)/src
#VOLNA_SRC = $(VOLNA_INSTALL_PATH)/src

HDF5_INC = -I$(HDF5_INSTALL_PATH)/include
HDF5_LIB = -L$(HDF5_INSTALL_PATH)/lib -lhdf5_hl -lhdf5 -lz

MPI_INC = -I$(MPI_INSTALL_PATH)/include

ifeq ($(OP2_COMPILER),gnu)
	CPP = g++
#	CPPFLAGS = -g0 -msse3 -fPIC -DUNIX -Wall
	CPPFLAGS = -g -fPIC -DUNIX -Wall
	#CPPFLAGS = -g -fPIC -DUNIX -Wall
	OMPFLAGS = -fopenmp
	MPICPP = /usr/bin/mpiCC
	MPIFLAGS = $(CCFLAGS)
else
ifeq ($(OP2_COMPILER),intel)
	CPP = icpc
	CCFLAGS = -g -O0 -vec-report -xSSE4.2 -parallel	
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


#VOLNA_SOURCES = \
# 	$(VOLNA_SRC)/values.cpp\
#	$(VOLNA_SRC)/event.cpp\
#	$(VOLNA_SRC)/geom.cpp

#VOLNA_OBJECTS = $(VOLNA_SOURCES:.cpp=.o)

all: clean volna

volna: volna.cpp Makefile 
	#$(CPP) $(CPPFLAGS) volna2hdf5.cpp $(VOLNA_OBJECTS) $(VOLNA_INC) $(OP2_INC) $(MPI_INC) $(OP2_LIB) -lop2_seq -lop2_hdf5 -o volna2hdf5
	$(MPICPP) $(CPPFLAGS) volna.cpp volna_event.cpp volna_init.cpp volna_output.cpp volna_simulation.cpp $(HDF5_INC) $(OP2_INC) $(HDF5_LIB) $(OP2_LIB) -lop2_seq -lop2_hdf5 -o volna

#
# cleanup
#

clean:
		rm -f volna *.o 

#%.o: %.cpp %.hpp
#	$(CPP) $(CPPFLAGS) -c -o $@ $<
