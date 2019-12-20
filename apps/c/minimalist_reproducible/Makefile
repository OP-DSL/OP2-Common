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
  CPPFLAGS	= -O2 -fPIC -DUNIX -Wall -O0 -g -Wextra
  OMPFLAGS	= -fopenmp
  MPICPP	= $(MPI_INSTALL_PATH)/bin/mpiCC
  MPIFLAGS	= $(CCFLAGS)
else
ifeq ($(OP2_COMPILER),intel)
  CPP		= icpc
  CCFLAGS	= -O3 -xAVX -DMPICH_IGNORE_CXX_SEEK -restrict -fno-alias -inline-forceinline -qopt-report=5 -parallel -DVECTORIZE -fp-model strict -fp-model source -prec-sqrt  #-parallel #-DCOMM_PERF #-DDEBUG #-vec-report
#  CCFLAGS	= -O3 -xAVX -DMPICH_IGNORE_CXX_SEEK -fno-alias -inline-forceinline -qopt-report -parallel -prec-div -DVECTORIZE #-parallel #-DCOMM_PERF #-DDEBUG #-vec-report
  CPPFLAGS 	= $(CCFLAGS)
  OMPFLAGS	= -qopenmp 
  MPICPP	= $(MPI_INSTALL_PATH)/bin/mpicxx
  NVCCFLAGS	= #-ccbin=$(MPICPP)
  MPIFLAGS	= $(CPPFLAGS)
else
ifeq ($(OP2_COMPILER),xl)
  CPP		= xlc++
  CCFLAGS	= -O3 -qarch=pwr8 -qtune=pwr8 -qhot -qxflag=nrcptpo -qinline=level=10 -Wx,-nvvm-compile-options=-ftz=1 -Wx,-nvvm-compile-options=-prec-div=0 -Wx,-nvvm-compile-options=-prec-sqrt=0
#  CCFLAGS	= -O3 -xAVX -DMPICH_IGNORE_CXX_SEEK -fno-alias -inline-forceinline -qopt-report -parallel -prec-div -DVECTORIZE #-parallel #-DCOMM_PERF #-DDEBUG #-vec-report
  CPPFLAGS 	= $(CCFLAGS)
  OMPFLAGS	= -qsmp=omp -qthreaded
  OMPOFFLOAD	= -qsmp=omp -qoffload -Xptxas -v -g1
  MPICPP	= $(MPI_INSTALL_PATH)/bin/mpicxx
  MPIFLAGS	= $(CPPFLAGS)
else
ifeq ($(OP2_COMPILER),pgi)
  CPP       	= pgc++
  CCFLAGS  	= -O3
  CPPFLAGS 	= $(CCFLAGS)
  OMPFLAGS 	= -mp
  MPICC   	= $(MPI_INSTALL_PATH)/bin/mpicc
  MPICPP   	= $(MPI_INSTALL_PATH)/bin/mpicxx
  MPIFLAGS 	= $(CPPFLAGS)
  NVCCFLAGS	= #-ccbin=$(MPICPP)
  ACCFLAGS      = -acc -Minfo=acc -ta=tesla:cc35,fastmath,lineinfo -DOPENACC -fast -Minfo=accel -Mcuda=ptxinfo
else
ifeq ($(OP2_COMPILER),cray)
  CPP           = CC
  CCFLAGS       = -O3 -h fp3 -h ipa5
  CPPFLAGS      = $(CCFLAGS)
  OMPFLAGS      = -h omp
  MPICPP        = CC
  MPIFLAGS      = $(CPPFLAGS)
else
ifeq ($(OP2_COMPILER),clang)
  CPP       	= clang++ 
  CCFLAGS  	= -O3 -ffast-math
  CPPFLAGS 	= $(CCFLAGS)
  OMPFLAGS 	= -I$(OMPTARGET_LIBS)/../tools/openmp/runtime/src/ -fopenmp=libomp -Rpass-analysis
  OMPOFFLOAD 	= $(OMPFLAGS) -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-nonaliased-maps -ffp-contract=fast -Xcuda-ptxas -v #-Xclang -target-feature -Xclang +ptx35
  MPICC   	= $(MPI_INSTALL_PATH)/bin/mpicc
  MPICPP   	= $(MPI_INSTALL_PATH)/bin/mpicxx
  MPIFLAGS 	= $(CPPFLAGS)
else
print:
	@echo "unrecognised value for OP2_COMPILER"
endif
endif
endif
endif
endif
endif



#
# set flags for NVCC compilation and linking
#
ifndef NV_ARCH
  MESSAGE=select an NVIDA device to compile in CUDA, e.g. make NV_ARCH=KEPLER
  NV_ARCH=Kepler
endif
ifeq ($(NV_ARCH),Fermi)
  CODE_GEN_CUDA=-gencode arch=compute_20,code=sm_21
else
ifeq ($(NV_ARCH),Kepler)
  CODE_GEN_CUDA=-gencode arch=compute_35,code=sm_35
else
ifeq ($(NV_ARCH),Pascal)
  CODE_GEN_CUDA=-gencode arch=compute_60,code=sm_60
endif
endif
endif

ifdef MAX_REG_COUNT
  REG_COUNT = --maxrregcount $(MAX_REG_COUNT)
  PGI_REG_COUNT = -ta=nvidia,maxregcount:$(MAX_REG_COUNT)
  ifeq ($(OP2_COMPILER),xl)
    OMP4_REG_COUNT= -Xptxas -maxrregcount=$(MAX_REG_COUNT)
  else
  ifeq ($(OP2_COMPILER),clang)
    OMP4_REG_COUNT= -Xcuda-ptxas --maxrregcount=$(MAX_REG_COUNT)
  endif
  endif
endif

NVCCFLAGS       := $(NVCCFLAGS) $(CODE_GEN_CUDA) -m64 -Xptxas=-v --use_fast_math -O3 $(REG_COUNT) -lineinfo  #-g -G -O0

VAR		= #-DOP_PART_SIZE_1=160 -DOP_PART_SIZE_2=320 -DOP_PART_SIZE_3=64 #-DOP_BLOCK_SIZE_0=64 -DOP_BLOCK_SIZE_1=64 -DOP_BLOCK_SIZE_2=64 -DOP_BLOCK_SIZE_3=64 -DOP_BLOCK_SIZE_4=64

#
# partitioning software for MPI versions
#
PARMETIS_VER=4
ifeq ($(PARMETIS_VER),4)
  PARMETIS_INC = -I$(PARMETIS_INSTALL_PATH)/include -DHAVE_PARMETIS -DPARMETIS_VER_4
  PARMETIS_LIB = -L$(PARMETIS_INSTALL_PATH)/lib -lparmetis -lmetis
else
  PARMETIS_INC = -I$(PARMETIS_INSTALL_PATH)/ -DHAVE_PARMETIS
  PARMETIS_LIB = -L$(PARMETIS_INSTALL_PATH)/ -lparmetis -lmetis
endif

PTSCOTCH_INC 	= -I$(PTSCOTCH_INSTALL_PATH)/include -DHAVE_PTSCOTCH
PTSCOTCH_LIB 	= -L$(PTSCOTCH_INSTALL_PATH)/lib/ -lptscotch \
                  -L$(PTSCOTCH_INSTALL_PATH)/lib/ -lptscotcherr

HDF5_INC = -I$(HDF5_INSTALL_PATH)/include
HDF5_LIB = -L$(HDF5_INSTALL_PATH)/lib -lhdf5 -lz

REPRO_MPI_INC = -I$(REPRO_MPI_INSTALL_PATH)inc
REPRO_MPI_LIB   = -L$(REPRO_MPI_LIB_PATH) -lbinned -lbinnedblas -lreproblas -lbinnedmpi -lm
#
# master to make all versions
#
ALL_TARGETS = clean minimal_mpi minimal_cuda minimal_openmp minimal_seq minimal_mpi_genseq minimal_mpi_cuda minimal_mpi_cuda_hyb minimal_mpi_openmp convert_mesh_seq 
ifeq ($(OP2_COMPILER),pgi)
	ALL_TARGETS += minimal_openacc minimal_mpi_openacc
endif
ifeq ($(OP2_COMPILER),intel)
	ALL_TARGETS += minimal_mpi_vec
endif

all: $(ALL_TARGETS)

#
# simple sequential version
#

minimal_seq: minimal.cpp increment_log.h
	     $(MPICPP) $(CPPFLAGS) minimal.cpp $(OP2_INC) $(HDF5_INC) $(OP2_LIB) -lop2_seq -lop2_hdf5 $(HDF5_LIB) -o minimal_seq

#
# x86 version using kernel files generated by op2.py
#

minimal_openmp:	minimal_op.cpp openmp/minimal_kernels.cpp \
		openmp/increment_log_kernel.cpp  increment_log.h \
                Makefile
		$(MPICPP) $(VAR) $(CPPFLAGS) $(OMPFLAGS) $(OP2_INC) $(OP2_LIB) $(HDF5_INC) -Iopenmp -I. \
		minimal_op.cpp -lm openmp/minimal_kernels.cpp -lm -lop2_openmp -lop2_hdf5 $(HDF5_LIB) -o minimal_openmp

#
# OpenACC version using kernel files generated by op2.py
#

minimal_openacc: minimal_op.cpp openacc/minimal_acckernels.c \
                openacc/increment_log_acckernel.c  \
                Makefile
		$(MPICC) $(VAR) $(CPPFLAGS) $(ACCFLAGS) $(OMPFLAGS) $(OP2_INC) $(PGI_REG_COUNT) -Iopenacc -I. \
                openacc/minimal_acckernels.c -c -o openacc/minimal_acckernels.o
		$(MPICPP) $(VAR) $(CPPFLAGS) $(ACCFLAGS) $(OMPFLAGS) $(OP2_INC) $(OP2_LIB) $(HDF5_INC) \
                minimal_op.cpp -lm openacc/minimal_acckernels.o -lm $(CUDA_LIB) -lcudart -lop2_cuda -lop2_hdf5 $(HDF5_LIB) -o minimal_openacc


#
# CUDA version using kernel files generated by op2.py
#

minimal_cuda:	minimal_op.cpp cuda/minimal_kernels_cu.o Makefile
		$(MPICPP) $(VAR) $(CPPFLAGS) minimal_op.cpp cuda/minimal_kernels_cu.o \
		$(CUDA_INC) $(OP2_INC) $(HDF5_INC) \
		$(OP2_LIB) $(CUDA_LIB) -lcudart -lop2_cuda -lop2_hdf5 $(HDF5_LIB) -o minimal_cuda

cuda/minimal_kernels_cu.o:	cuda/minimal_kernels.cu      \
		cuda/increment_log_kernel.cu increment_log.h \
                Makefile
		nvcc  $(VAR) $(INC) $(NVCCFLAGS) $(OP2_INC) $(HDF5_INC) -Icuda -I. \
		-c -o cuda/minimal_kernels_cu.o cuda/minimal_kernels.cu

#
#  CUDA with clang
#

minimal_cuda_clang:     minimal_op.cpp cuda/minimal_kernels_cu_clang.o Makefile
	$(MPICPP) $(VAR) $(CPPFLAGS) minimal_op.cpp cuda/minimal_kernels_cu_clang.o \
        $(CUDA_INC) $(OP2_INC) $(HDF5_INC) \
	$(OP2_LIB) $(CUDA_LIB) -lcudart -lop2_cuda -lop2_hdf5 $(HDF5_LIB) -o minimal_cuda_clang

cuda/minimal_kernels_cu_clang.o:     cuda/minimal_kernels.cu      \
                cuda/increment_log_kernel.cu increment_log.h \
                Makefile
	clang++  $(VAR) $(INC) -m64 -O3 $(OP2_INC) $(HDF5_INC) -gline-tables-only -I$(MPI_INSTALL_PATH)/include -Icuda -I. \
	-c -o cuda/minimal_kernels_cu_clang.o cuda/minimal_kernels.cu --cuda-gpu-arch=sm_35 -ffast-math $(OMP4_REG_COUNT) -Xcuda-ptxas -v

#
# mpi with sequential-nodes version
#

minimal_mpi: minimal.cpp increment_log.h Makefile
	$(MPICPP) $(MPIFLAGS) minimal.cpp $(OP2_INC) $(PARMETIS_INC) $(PTSCOTCH_INC) $(HDF5_INC) \
	$(OP2_LIB) -lop2_mpi $(PARMETIS_LIB) $(PTSCOTCH_LIB) $(HDF5_LIB) -o minimal_mpi

#
# mpi genseq version using kernel files generated by op2.py
#

minimal_mpi_genseq: minimal_op.cpp seq/minimal_seqkernels.cpp \
                seq/increment_log_seqkernel.cpp  increment_log.h \
                Makefile
		$(MPICPP) $(VAR) $(CPPFLAGS) $(OP2_INC) $(OP2_INC) $(HDF5_INC) \
		$(PARMETIS_INC) $(PTSCOTCH_INC) -Iseq  -I. $(REPRO_MPI_INC) \
		minimal_op.cpp -lm seq/minimal_seqkernels.cpp -Wl,--start-group $(OP2_LIB) -lop2_mpi \
		$(PARMETIS_LIB) $(PTSCOTCH_LIB)  $(HDF5_LIB) $(REPRO_MPI_LIB) -Wl,--end-group -o minimal_mpi_genseq

#
# mpi vectorized seq version using kernel files generated by op2.py
#

minimal_mpi_vec: minimal_op.cpp vec/minimal_veckernels.cpp \
                vec/increment_log_veckernel.cpp  increment_log.h \
                Makefile
		$(MPICPP) $(VAR) $(CPPFLAGS) $(OP2_INC) $(OP2_INC) $(HDF5_INC) \
                $(PARMETIS_INC) $(PTSCOTCH_INC) -Ivec  -I. \
                minimal_op.cpp -lm vec/minimal_veckernels.cpp $(OP2_LIB) -lop2_mpi \
                $(PARMETIS_LIB) $(PTSCOTCH_LIB) $(HDF5_LIB) -o minimal_mpi_vec


#
# mpi openmp version using kernel files generated by op2.py
#

minimal_mpi_openmp: minimal_op.cpp openmp/minimal_kernels.cpp \
                openmp/increment_log_kernel.cpp  increment_log.h \
                Makefile
		$(MPICPP) $(VAR) $(CPPFLAGS) $(OMPFLAGS) $(OP2_INC) $(OP2_INC) $(HDF5_INC) \
		$(PARMETIS_INC) $(PTSCOTCH_INC) -Iopenmp -I. $(REPRO_MPI_INC)\
		minimal_op.cpp -lm openmp/minimal_kernels.cpp $(OP2_LIB) -lop2_mpi \
		$(PARMETIS_LIB) $(PTSCOTCH_LIB) $(HDF5_LIB) $(REPRO_MPI_LIB) -o minimal_mpi_openmp

#
# mpi openmp 4.5 version using kernel files generated by op2.py
#

minimal_openmp4: minimal_op.cpp openmp4/minimal_omp4kernels.cpp openmp4/minimal_omp4kernel_funcs.cpp \
			openmp4/increment_log_omp4kernel_func.cpp  increment_log.h openmp4/increment_log_omp4kernel.cpp \
			Makefile
	$(CPP) $(VAR) $(CPPFLAGS) $(OMPOFFLOAD) $(OP2_INC) $(HDF5_INC) $(OMP4_REG_COUNT) \
		-Iopenmp4/  -I. -c openmp4/minimal_omp4kernel_funcs.cpp  -o openmp4/minimal_omp4kernel_funcs.o
	$(CPP) $(VAR) $(CPPFLAGS) $(OMPFLAGS) $(OP2_INC) $(HDF5_INC) $(OMP4_REG_COUNT) \
		-Iopenmp4/ -I. -c openmp4/minimal_omp4kernels.cpp -o openmp4/minimal_omp4kernels.o
	$(MPICPP) $(VAR) $(CPPFLAGS) $(OMPOFFLOAD) $(OP2_INC) $(OP2_LIB) $(HDF5_INC)  \
		minimal_op.cpp openmp4/minimal_omp4kernels.o openmp4/minimal_omp4kernel_funcs.o -o minimal_openmp4 \
	   	-lop2_openmp4 -lop2_hdf5 $(HDF5_LIB) -L$(CUDA_INSTALL_PATH)/lib64 -lcudart


minimal_mpi_openmp4: minimal_op.cpp openmp4/minimal_omp4kernels.cpp openmp4/minimal_omp4kernel_funcs.cpp \
			openmp4/increment_log_omp4kernel_func.cpp  increment_log.h openmp4/increment_log_omp4kernel.cpp \
			Makefile
	$(CPP) $(VAR) $(CPPFLAGS) $(OMPOFFLOAD) $(OP2_INC) $(HDF5_INC) \
		-Iopenmp4/ -I. -c openmp4/minimal_omp4kernel_funcs.cpp  -o openmp4/minimal_omp4kernel_funcs.o
	$(CPP) $(VAR) $(CPPFLAGS) $(OMPFLAGS) $(OP2_INC) $(HDF5_INC) \
		-Iopenmp4/ -I. -c openmp4/minimal_omp4kernels.cpp -o openmp4/minimal_omp4kernels.o
	$(MPICPP) $(VAR) $(CPPFLAGS) $(OMPOFFLOAD) $(OP2_INC) $(OP2_LIB) $(HDF5_INC) $(CUDA_LIB) -lcudart  \
		minimal_op.cpp openmp4/minimal_omp4kernels.o openmp4/minimal_omp4kernel_funcs.o -o minimal_mpi_openmp4 \
	   	-lop2_mpi_cuda $(HDF5_LIB) $(PARMETIS_LIB) $(PTSCOTCH_LIB) $(CUDA_LIB) -lcudart
#
# mpi openacc version using kernel files generated by op2.py
#

minimal_mpi_openacc: minimal_op.cpp openacc/minimal_acckernels.c \
                openacc/increment_log_acckernel.c  increment_log.h \
                Makefile
		$(MPICC) $(VAR) $(CPPFLAGS) $(ACCFLAGS) $(OMPFLAGS) $(OP2_INC) -Iopenacc/ -I. \
		openacc/minimal_acckernels.c -c -o minimal_acckernels.o
		$(MPICPP) $(VAR) $(CPPFLAGS) $(ACCFLAGS) $(OP2_INC) $(OP2_INC) $(HDF5_INC) \
		$(PARMETIS_INC) $(PTSCOTCH_INC) \
		minimal_op.cpp -lm openacc/minimal_acckernels.o -DOPENACC $(CUDA_LIB) -lcudart $(OP2_LIB) -lop2_mpi_cuda \
		$(PARMETIS_LIB) $(PTSCOTCH_LIB) $(HDF5_LIB) -o minimal_mpi_openacc
#
# mpi with CUDA version
#

minimal_mpi_cuda: minimal_op.cpp cuda/minimal_kernels_mpi_cu.o Makefile
		  $(MPICPP) $(MPIFLAGS) minimal_op.cpp -lm cuda/minimal_kernels_mpi_cu.o \
		  $(OP2_INC) $(PARMETIS_INC) $(PTSCOTCH_INC) $(HDF5_INC) \
		  $(OP2_LIB) -lop2_mpi_cuda $(PARMETIS_LIB) $(PTSCOTCH_LIB) \
		  $(HDF5_LIB) $(CUDA_LIB) -lcudart -o minimal_mpi_cuda

cuda/minimal_kernels_mpi_cu.o: cuda/minimal_kernels.cu \
		cuda/increment_log_kernel.cu  increment_log.h \
		Makefile
		nvcc  $(INC) $(NVCCFLAGS) $(OP2_INC) -I $(MPI_INSTALL_PATH)/include -Icuda -I. \
		-c -o cuda/minimal_kernels_mpi_cu.o cuda/minimal_kernels.cu

#
# Hybrid CPU+GPU version
#

minimal_mpi_cuda_hyb: minimal_op.cpp cuda/minimal_hybkernels.o Makefile
		  $(MPICPP) $(MPIFLAGS) $(OMPFLAGS) -DOP_HYBRID_GPU minimal_op.cpp -lm cuda/minimal_hybkernels.o cuda/minimal_hybkernels2.o \
		  $(OP2_INC) $(PARMETIS_INC) $(PTSCOTCH_INC) $(HDF5_INC) \
		  $(OP2_LIB) -lop2_mpi_cuda $(PARMETIS_LIB) $(PTSCOTCH_LIB) \
		  $(HDF5_LIB) $(CUDA_LIB) -lcudart -o minimal_mpi_cuda_hyb

cuda/minimal_hybkernels.o: cuda/minimal_hybkernels.cu \
                cuda/minimal_kernels.cu increment_log.h \
                Makefile
		nvcc -DOP_HYBRID_GPU -DGPUPASS $(INC) $(NVCCFLAGS) $(OP2_INC) -I $(MPI_INSTALL_PATH)/include -Icuda -I. \
                -c -o cuda/minimal_hybkernels.o cuda/minimal_hybkernels.cu
		cat cuda/minimal_hybkernels.cu > cuda/minimal_hybkernels2.cpp
		$(MPICPP) -DOP_HYBRID_GPU $(OMPFLAGS) $(INC) $(OP2_INC) -I $(MPI_INSTALL_PATH)/include -Icuda -I. -Iopenmp \
                -c -o cuda/minimal_hybkernels2.o cuda/minimal_hybkernels2.cpp
		rm cuda/minimal_hybkernels2.cpp

#
# convert ASCI new_gird.dat to HDF5 new_grid.h5
#

convert_mesh_seq: convert_mesh.cpp
	$(MPICPP) $(MPIFLAGS) $(REPRO_MPI_INC) convert_mesh.cpp $(OP2_INC) $(PARMETIS_INC) $(PTSCOTCH_INC) $(HDF5_INC) \
	$(OP2_LIB) -lop2_seq -lop2_hdf5 $(PARMETIS_LIB) $(PTSCOTCH_LIB) $(HDF5_LIB) -o convert_mesh_seq



#
# cleanup
#

clean:
		rm -f minimal_seq minimal_openmp minimal_cuda minimal_mpi minimal_mpi_genseq minimal_mpi_vec minimal_mpi_cuda_hyb minimal_mpi_openmp minimal_mpi_cuda convert_mesh_seq minimal_openacc minimal_mpi_openacc *.o cuda/*.o openacc/*.o openmp4/*.o *.optrpt
