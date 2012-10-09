#
# The following environment variables should be predefined:
#
# OP2_INSTALL_PATH
# HDF5_INSTALL_PATH
# VOLNA_INSTALL_PATH
#

#
# set paths for header files and libraries
#

LIBMESH_INC = -I$(LIBMESH_INSTALL_PATH)/include
LIBMESH_LIB = -L$(LIBMESH_INSTALL_PATH)/lib

OP2_INC = -I$(OP2_INSTALL_PATH)/c/include
OP2_LIB = -L$(OP2_INSTALL_PATH)/c/lib

VOLNA_INSTALL_PATH = /home/endre/VOLNA_original/volna
VOLNA_INC = -I$(VOLNA_INSTALL_PATH)/src
VOLNA_SRC = $(VOLNA_INSTALL_PATH)/src

HDF5_INC = -I$(HDF5_INSTALL_PATH)/include
HDF5_LIB = -L$(HDF5_INSTALL_PATH)/lib -lhdf5_fortran -lhdf5 -lhdf5_hl -lz

OPENMPI_INC = -I/home/gihan/openmpi-intel/include

ifeq ($(OP2_COMPILER),gnu)
	CPP = g++
#	CPPFLAGS = -g0 -msse3 -fPIC -DUNIX -Wall
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


VOLNA_SOURCES = \
 	$(VOLNA_SRC)/values.cpp\
	$(VOLNA_SRC)/event.cpp\
	$(VOLNA_SRC)/geom.cpp
#	$(VOLNA_INSTALL_PATH)/volna.cpp
#	$(srcdir)/event.cpp\
	volna.cpp
#	$(srcdir)/timeStepper.cpp \
# 	$(srcdir)/classes.cpp \
# 	$(srcdir)/config.cpp \
# 	$(srcdir)/event.cpp \
# 	$(srcdir)/geom.cpp \
#	$(srcdir)/init.cpp \
#	$(srcdir)/mathParser.cpp \ 
#	$(srcdir)/mesh.cpp 
# 	$(srcdir)/meshIo.cpp \
# 	$(srcdir)/meshObjects.cpp \
# 	$(srcdir)/paramFileParser.cpp \
# 	$(srcdir)/physicalParams.cpp \
# 	$(srcdir)/precomputed.cpp \
# 	$(srcdir)/problem.cpp \
# 	$(srcdir)/simulation.cpp \
# 	$(srcdir)/unknowns.cpp \

VOLNA_OBJECTS = $(VOLNA_SOURCES:.cpp=.o)

all: clean volna2hdf5

volna2hdf5: volna2hdf5.cpp $(VOLNA_OBJECTS) Makefile 
	@echo $(OP2_LIB)
	#$(CPP) $(CPPFLAGS) volna2hdf5.cpp $(VOLNA_OBJECTS) $(VOLNA_INC) $(OP2_INC) $(OPENMPI_INC) $(OP2_LIB) -lop2_seq -lop2_hdf5 -o volna2hdf5
	$(CPP) $(CPPFLAGS) volna2hdf5.cpp $(VOLNA_OBJECTS) $(VOLNA_INC) $(HDF5_INC) $(OP2_INC) $(OPENMPI_INC) $(HDF5_LIB) $(OP2_LIB) -lop2_seq -lop2_hdf5 -o volna2hdf5

#
# cleanup
#

clean:
		rm -f volna2hdf5 *.o $(VOLNA_OBJECTS)

%.o: %.cpp %.hpp
	$(CPP) $(CPPFLAGS) -c -o $@ $<
