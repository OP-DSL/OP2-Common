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
  CPPFLAGS	= -O2 -vec-report -xSSE2,SSE3,SSE4.1,SSE4.2 -g #-DCOMM_PERF #-DDEBUG
  OMPFLAGS	= -openmp -openmp-report2
  MPICPP	= mpiCC
  MPIFLAGS	= $(CCFLAGS)
else
print:
	@echo "unrecognised value for OP2_COMPILER"
endif
endif

#
# set flags for NVCC compilation and linking
#
#NVCCFLAGS	= -arch=sm_21 -Xptxas=-v -use_fast_math
NVCCFLAGS	= -arch=sm_20 -Xptxas=-v -use_fast_math
#NVCCFLAGS	= -g -G -arch=sm_20 -Xptxas=-v -use_fast_math


#
# partitioning software for MPI versions
#

PARMETIS_INC 	= -I$(PARMETIS_INSTALL_PATH) -DPARMETIS
PARMETIS_LIB 	= -L$(PARMETIS_INSTALL_PATH) -lparmetis \
	       	  -L$(PARMETIS_INSTALL_PATH) -lmetis

PTSCOTCH_INC 	= -I$(PTSCOTCH_INSTALL_PATH)/include -DPTSCOTCH
PTSCOTCH_LIB 	= -L$(PTSCOTCH_INSTALL_PATH)/lib/ -lptscotch \
                  -L$(PTSCOTCH_INSTALL_PATH)/lib/ -lptscotcherr

HDF5_INC = -I$(HDF5_INSTALL_PATH)/include
HDF5_LIB = -L$(HDF5_INSTALL_PATH)/lib -lhdf5 -lz

MPI_INC = -I$(MPI_INSTALL_PATH)/include

#
# master to make all versions
#

all: clean airfoil_mpi_cuda

#
# mpi with openmp-nodes version
#

airfoil_mpi_cuda: airfoil_mpi_cuda.cpp airfoil_mpi_kernels_cu.o \
		  Makefile
		  $(MPICPP) $(MPIFLAGS) $(OMPFLAGS) airfoil_mpi_cuda.cpp airfoil_mpi_kernels_cu.o \
		  $(OP2_INC) $(PARMETIS_INC) $(PTSCOTCH_INC) $(HDF5_INC) \
		  $(OP2_LIB) -lop2_mpi_cuda $(PARMETIS_LIB) $(PTSCOTCH_LIB) \
		  $(HDF5_LIB) -lcudart -o airfoil_mpi_cuda

airfoil_mpi_kernels_cu.o:	airfoil_mpi_kernels.cu      \
		  save_soln_mpi_kernel.cu  save_soln.h \
		  adt_calc_mpi_kernel.cu   adt_calc.h  \
		  res_calc_mpi_kernel.cu   res_calc.h  \
		  bres_calc_mpi_kernel.cu  bres_calc.h \
		  update_mpi_kernel.cu     update.h    \
		  Makefile
		  nvcc  $(INC) $(NVCCFLAGS) $(OP2_INC) $(MPI_INC) \
		  -c -o airfoil_mpi_kernels_cu.o airfoil_mpi_kernels.cu


#
# cleanup
#

clean:
		rm -f rm -f airfoil_mpi_cuda *.o
