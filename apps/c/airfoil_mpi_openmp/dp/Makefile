#
# The following environment variables should be predefined:
#
# CUDA_INSTALL_PATH
# PARMETIS_INSTALL_PATH
# PTSCOTCH_INSTALL_PATH
# HDF5_INSTALL_PATH
#
# OP2_INSTALL_PATH
# OP2_COMPILER (gnu,intel,etc)
#

#
# set paths for header files and libraries
#
OP2_INC		= -I$(OP2_INSTALL_PATH)/c/include
OP2_LIB		= -L$(OP2_INSTALL_PATH)/c/lib

CUDA_INC	= -I$(CUDA_INSTALL_PATH)/include
CUDA_LIB	= -L$(CUDA_INSTALL_PATH)/lib64


ifeq ($(OP2_COMPILER),gnu)
  CPP		= g++
  CPPFLAGS	= -g -fPIC -DUNIX -Wall
#  CPPFLAGS	= -O3 -fPIC -DUNIX -Wall -Wextra
  OMPFLAGS	= -fopenmp
  MPICPP	= /usr/bin/mpiCC
  MPIFLAGS	= $(CCFLAGS)
else
ifeq ($(OP2_COMPILER),intel)
  CPP		= icpc
  CPPFLAGS	= -O3 -vec-report -xSSE4.2 #-DCOMM_PERF #-DDEBUG
  OMPFLAGS	= -openmp -openmp-report2
  MPICPP	= mpiCC
  MPIFLAGS	= $(CPPFLAGS)
else
print:
	@echo "unrecognised value for OP2_COMPILER"
endif
endif

#
# set flags for NVCC compilation and linking
#
NVCCFLAGS	= -O3 -arch=sm_20 -Xptxas=-v -use_fast_math
#NVCCFLAGS	= -g -G -arch=sm_20 -Xptxas=-v -use_fast_math


#
# partitioning software for MPI versions
#

PARMETIS_INC 	= -I$(PARMETIS_INSTALL_PATH) -DHAVE_PARMETIS
PARMETIS_LIB 	= -L$(PARMETIS_INSTALL_PATH) -lparmetis \
	       	  -L$(PARMETIS_INSTALL_PATH) -lmetis

PTSCOTCH_INC 	= -I$(PTSCOTCH_INSTALL_PATH)/include -DHAVE_PTSCOTCH
PTSCOTCH_LIB 	= -L$(PTSCOTCH_INSTALL_PATH)/lib/ -lptscotch \
                  -L$(PTSCOTCH_INSTALL_PATH)/lib/ -lptscotcherr

HDF5_INC = -I$(HDF5_INSTALL_PATH)/include
HDF5_LIB = -L$(HDF5_INSTALL_PATH)/lib -lhdf5 -lz


#
# master to make all versions
#

all: clean airfoil_mpi_openmp

#
# mpi with openmp-nodes version
#

airfoil_mpi_openmp: airfoil_mpi_openmp.cpp airfoil_kernels.cpp \
		    save_soln_kernel.cpp  save_soln.h \
		    adt_calc_kernel.cpp   adt_calc.h  \
		    res_calc_kernel.cpp   res_calc.h  \
		    bres_calc_kernel.cpp  bres_calc.h \
		    update_kernel.cpp     update.h    \
		    Makefile
		    $(MPICPP) $(MPIFLAGS) $(OMPFLAGS) airfoil_mpi_openmp.cpp airfoil_kernels.cpp \
		    $(OP2_INC) $(PARMETIS_INC) $(PTSCOTCH_INC) $(HDF5_INC) \
		    $(OP2_LIB) -lop2_mpi $(PARMETIS_LIB) $(PTSCOTCH_LIB) \
		    $(HDF5_LIB) -o airfoil_mpi_openmp


#
# cleanup
#

clean:
		rm -f rm -f airfoil_mpi_openmp *.o
