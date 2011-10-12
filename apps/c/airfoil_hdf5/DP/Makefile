#
# set paths for header files and libraries
#

C_OP2		= $(OP2)/c

INC_OP2 = $(C_OP2)/include
SRC_OP2 = $(C_OP2)/src
LIB_OP2 = $(C_OP2)/lib


#CUDA stuff
CUDA := $(CUDA_INSTALL_PATH)
INC_CUDA := -I$(CUDA)/include -I$(C_OP2) -I.
LIB_CUDA := -L$(CUDA)/lib64


# compiler
CC = icc
CPP = icpc
MPI_HOME = /home/gihan/openmpi-intel
MPICPP = $(MPI_HOME)/bin/mpiCC -g

CCFLAGS	:= -O2 -vec-report -xSSE2,SSE3,SSE4.1,SSE4.2 #-g #-DCOMM_PERF #-DDEBUG

#openmp
OMP	:= -openmp -openmp-report2

#
# set flags for compilation and linking
#

# -Xptxas -dlcm=ca   => use L1 cache
# -Xptxas -dlcm=cg   => no L1 cache

#CCFLAGS	:= -g -fPIC -DUNIX -Wall -Wextra
#NVCCFLAGS	:= -g -G -arch=sm_20 -Xptxas=-v -use_fast_math

#CCFLAGS		:= -g
#CCFLAGS		:= -O3 -DUNIX -Wall -Wextra
#NVCCFLAGS	:= -O3 -maxrregcount 48 -arch=sm_20 \
#		   -Xptxas -dlcm=cg -Xptxas=-v -use_fast_math
NVCCFLAGS	:= -O3 -arch=sm_20 \
		   -Xptxas -dlcm=ca -Xptxas=-v -use_fast_math

LIBS		:= $(LIB) -lcudart

C_OP2_H		:= $(C_OP2)/op_seq.h $(C_OP2)/op_lib.h $(C_OP2)/op_lib_core.h
OP2_LIB		:= $(C_OP2)/op_lib.a

EXEC_SEQ = airfoil_seq
EXEC_OPENMP = airfoil_openmp
EXEC_CUDA = airfoil_cuda
EXEC_MPI = airfoil_mpi
EXEC_CONVERT = convert_grid

PARMETIS_LIB 	:= -L$(PARMETIS_INSTALL_PATH) -lparmetis -L$(PARMETIS_INSTALL_PATH) -lmetis
PTSCOTCH_LIB 	:= -L$(PTSCOTCH_INSTALL_PATH)/lib/ -lptscotch -L$(PTSCOTCH_INSTALL_PATH)/lib/ -lptscotcherr


PARMETIS_INC = $(PARMETIS_INSTALL_PATH) -DPARMETIS
PTSCOTCH_INC = $(PTSCOTCH_INSTALL_PATH)/include -DPTSCOTCH

HDF5_INC = -I/home/gihan/hdf5/include/ 
HDF5_LIB = -L/home/gihan/hdf5/lib/ -lhdf5 -lz 


#
# master to make all versions
#

all: airfoil_mpi convert_grid


#
# mpi with sequential-nodes version
#

airfoil_mpi: airfoil_mpi.cpp save_soln.h adt_calc.h res_calc.h bres_calc.h \
	Makefile $(LIB_OP2)/libop2_mpi.a $(INC_OP2)/op_lib_c.h $(INC_OP2)/op_lib_cpp.h
	$(MPICPP) $(CCFLAGS) airfoil_mpi.cpp -I$(INC_OP2) -I$(PARMETIS_INC) -I$(PTSCOTCH_INC) $(HDF5_INC)\
	-L$(LIB_OP2) -lop2_mpi $(PARMETIS_LIB) $(PTSCOTCH_LIB) $(HDF5_LIB)-o $(EXEC_MPI)


#
# convert ASCI new_gird.dat to HDF5 new_grid.h5
#

convert_grid: convert_grid_to_hdf5.cpp save_soln.h adt_calc.h res_calc.h bres_calc.h \
	Makefile $(LIB_OP2)/libop2_mpi.a $(INC_OP2)/op_lib_c.h $(INC_OP2)/op_lib_cpp.h
	$(MPICPP) $(CCFLAGS) convert_grid_to_hdf5.cpp -I$(INC_OP2) -I$(PARMETIS_INC) -I$(PTSCOTCH_INC) $(HDF5_INC)\
	-L$(LIB_OP2) -lop2_mpi $(PARMETIS_LIB) $(PTSCOTCH_LIB) $(HDF5_LIB)-o $(EXEC_CONVERT)


	
	
#
# cleanup
#

clean:
		rm -f $(EXEC_SEQ) $(EXEC_OPENMP) $(EXEC_CUDA) $(EXEC_MPI) $(EXEC_CONVERT) *.o
