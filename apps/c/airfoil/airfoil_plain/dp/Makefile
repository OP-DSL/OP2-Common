#
# The following environment variables should be predefined:
#
# OP2_COMPILER (gnu,intel,etc)
#
# ... and optionally the following. If not defined, then 
# they must be locatable via PATH variables:
#
# OP2_INSTALL_PATH
# CUDA_INSTALL_PATH
# PARMETIS_INSTALL_PATH
# PTSCOTCH_INSTALL_PATH
# HDF5_INSTALL_PATH
#

include ../../../../make-common.inc

ifeq ($(OP2_COMPILER),gnu)
  CPP		= g++
  CPPFLAGS	= -arch x86_64 -g -fPIC -DUNIX -Wall #-Wextra
  OMPFLAGS	= -fopenmp
  MPICPP	= $(MPICXX_PATH)
  MPIFLAGS	= $(CPPFLAGS)
else
ifeq ($(OP2_COMPILER),intel)
  CPP		= icpc
  CCFLAGS	= -O3 -xHost -DMPICH_IGNORE_CXX_SEEK -restrict -fno-alias -inline-forceinline -qopt-report -parallel -DVECTORIZE #-parallel #-DCOMM_PERF #-DDEBUG #-vec-report
  CPPFLAGS 	= $(CCFLAGS)
  OMPFLAGS	= -qopenmp
  MPICPP	= $(MPICXX_PATH)
  # NVCCFLAGS	= -ccbin=$(MPICPP)
  MPIFLAGS	= $(CPPFLAGS)
else
ifeq ($(OP2_COMPILER),pgi)
  CPP       	= pgc++
  CCFLAGS  	= -O3
  CPPFLAGS 	= $(CCFLAGS)
  OMPFLAGS 	= -mp
  MPICPP   	= $(MPICXX_PATH)
  MPIFLAGS 	= $(CPPFLAGS)
  NVCCFLAGS	= -ccbin=$(MPICPP)
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
  CPP           = clang++
  CCFLAGS       = -O3 #-ffast-math
  CPPFLAGS      = $(CCFLAGS)
  OMPFLAGS      = -I$(OMPTARGET_LIBS)/../include -fopenmp=libomp -Rpass-analysis
  OMPOFFLOAD    = -fopenmp-targets=nvptx64-nvidia-cuda  -ffp-contract=fast -Xcuda-ptxas -v #-Xclang -target-feature -Xclang +ptx35 -fopenmp-nonaliased-maps
  MPICC         = $(MPICC_PATH)
  MPICPP        = $(MPICXX_PATH)
  MPIFLAGS      = $(CPPFLAGS)
  NVCCFLAGS     = -ccbin=$(NVCC_HOST_COMPILER)
else
ifeq ($(OP2_COMPILER),sycl)
  ifdef DEBUG
    CCFLAGS  = -g -O0
  else
    CCFLAGS  = -g -O3
  endif
  CXX	    = g++
  SYCLCXX = compute++
  CXXFLAGS  = $(CCFLAGS)
  MPICXX    = $(MPICPP_PATH)
  MPIFLAGS  = $(CXXFLAGS)
  NVCCFLAGS = -ccbin=$(NVCC_HOST_COMPILER)
  SYCL_FLAGS = -sycl-driver -sycl-target spir -no-serial-memop  -I$(SYCL_INSTALL_PATH)/include  #compute++ SYCL
  #SYCL_FLAGS = -sycl-driver -sycl-target ptx64 -no-serial-memop  -I$(SYCL_INSTALL_PATH)/include  #compute++ SYCL 
  SYCL_LIB   = -L$(SYCL_INSTALL_PATH)/lib -lComputeCpp -L$(CUDA_INSTALL_PATH)/lib64 -lOpenCL
  #SYCL_FLAGS = -mllvm -inline-threshold=1000 -sycl-driver -sycl-target ptx64 -no-serial-memop -I$(SYCL_INSTALL_PATH)/include -I$(CUDA_INSTALL_PATH)/include -std=c++11
  #SYCL_FLAGS = -mllvm -inline-threshold=1000 -I$(SYCL_INSTALL_PATH)/include -I$(CUDA_INSTALL_PATH)/include -std=c++11
else
ifeq ($(OP2_COMPILER),intel-sycl)
  ifdef DEBUG
    CCFLAGS  = -g -O0
  else
    CCFLAGS  = -O3
  endif
  CXX       = g++
  SYCLCXX   = clang++ -O3
  CXXFLAGS  = $(CCFLAGS)
  MPICXX    = $(MPICPP_PATH)
  MPIFLAGS  = $(CXXFLAGS)
  SYCL_LIB   = -L$(SYCL_INSTALL_PATH)/lib -lOpenCL
  NVCCFLAGS = -ccbin=$(NVCC_HOST_COMPILER)
  SYCL_FLAGS = -std=c++11 -fsycl -I$(SYCL_INSTALL_PATH)/include -I$(SYCL_INSTALL_PATH)/include #intel sycl
  #SYCL_FLAGS = -std=c++11 -fsycl #intel sycl
else
ifeq ($(OP2_COMPILER),hipsycl)
  ifdef DEBUG
    CCFLAGS  = -g -O0
  else
    CCFLAGS  = -O3
  endif
  CXX	    = g++
  SYCLCXX = syclcc-clang
  CXXFLAGS  = $(CCFLAGS)
  MPICXX    = $(MPICPP_PATH)
  MPIFLAGS  = $(CXXFLAGS)
  NVCCFLAGS = -ccbin=$(NVCC_HOST_COMPILER)
  SYCL_LIB   = 

else
print:
	@echo "unrecognised value for OP2_COMPILER"
endif
endif
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
ifeq ($(OP2_COMPILER),hipsycl)
  SYCL_FLAGS = --hipsycl-gpu-arch=sm_35
endif
else
ifeq ($(NV_ARCH),Maxwell)
  CODE_GEN_CUDA=-gencode arch=compute_50,code=sm_50
else
ifeq ($(NV_ARCH),Pascal)
  CODE_GEN_CUDA=-gencode arch=compute_60,code=sm_60
ifeq ($(OP2_COMPILER),hipsycl)
  SYCL_FLAGS = --hipsycl-gpu-arch=sm_60
endif
else
ifeq ($(NV_ARCH),Volta)
  CODE_GEN_CUDA=-gencode arch=compute_70,code=sm_70
ifeq ($(OP2_COMPILER),hipsycl)
  SYCL_FLAGS = --hipsycl-gpu-arch=sm_70
endif
endif
endif
endif
endif
endif


NVCCFLAGS += -O3 $(CODE_GEN_CUDA) -m64 -Xptxas=-v -use_fast_math #--Xptxas -dlcm=ca g -G -O0

#VAR		= -DOP_PART_SIZE_1=512 -DOP_PART_SIZE_2=1024 -DOP_PART_SIZE_3=64
#-DOP_BLOCK_SIZE_0=512 -DOP_BLOCK_SIZE_1=64 -DOP_BLOCK_SIZE_2=64 -DOP_BLOCK_SIZE_3=64 -DOP_BLOCK_SIZE_4=64

#
# master to make all versions
#
ALL_TARGETS = clean airfoil_seq airfoil_genseq airfoil_openmp airfoil_mpi airfoil_mpi_genseq airfoil_mpi_openmp airfoil_cuda airfoil_mpi_cuda
ifeq ($(OP2_COMPILER),pgi)
        ALL_TARGETS += #airfoil_openacc airfoil_mpi_openacc
endif
ifeq ($(OP2_COMPILER),intel)
        ALL_TARGETS += airfoil_vec airfoil_mpi_vec
endif

all: $(ALL_TARGETS)


#
# master to make all versions
#


#
# simple header file sequential version
#

airfoil_seq: airfoil.cpp save_soln.h adt_calc.h res_calc.h bres_calc.h Makefile
	     $(CPP) $(CPPFLAGS) airfoil.cpp $(OP2_INC) $(OP2_LIB) -lop2_seq -o airfoil_seq

#
# codegenerated sequential version
#

airfoil_genseq: airfoil_op.cpp seq/airfoil_seqkernels.cpp \
                seq/save_soln_seqkernel.cpp  save_soln.h \
                seq/adt_calc_seqkernel.cpp   adt_calc.h  \
                seq/res_calc_seqkernel.cpp   res_calc.h  \
                seq/bres_calc_seqkernel.cpp  bres_calc.h \
                seq/update_seqkernel.cpp     update.h    \
                Makefile
		$(MPICPP) $(VAR) $(CPPFLAGS) $(OP2_INC) $(OP2_INC) \
                $(PARMETIS_INC) $(PTSCOTCH_INC) -Iseq -I. \
                airfoil_op.cpp -lm seq/airfoil_seqkernels.cpp $(OP2_LIB) -lop2_seq \
                $(PARMETIS_LIB) $(PTSCOTCH_LIB) -o airfoil_genseq

#
# codegenerated vectorised sequential version
#

airfoil_vec: airfoil_op.cpp vec/airfoil_veckernels.cpp \
                vec/save_soln_veckernel.cpp  save_soln.h \
                vec/adt_calc_veckernel.cpp   adt_calc.h  \
                vec/res_calc_veckernel.cpp   res_calc.h  \
                vec/bres_calc_veckernel.cpp  bres_calc.h \
                vec/update_veckernel.cpp     update.h    \
                Makefile
		$(MPICPP) -DVECTORIZE $(VAR) $(OMPFLAGS) $(CPPFLAGS) $(OP2_INC) $(OP2_INC) \
                $(PARMETIS_INC) $(PTSCOTCH_INC) -Ivec -I. \
                airfoil_op.cpp -lm vec/airfoil_veckernels.cpp $(OP2_LIB) -lop2_seq \
                $(PARMETIS_LIB) $(PTSCOTCH_LIB) -o airfoil_vec
#
# x86 version using kernel files generated by op2.py
#

airfoil_openmp:	airfoil_op.cpp openmp/airfoil_kernels.cpp \
		openmp/save_soln_kernel.cpp  save_soln.h \
		openmp/adt_calc_kernel.cpp   adt_calc.h  \
		openmp/res_calc_kernel.cpp   res_calc.h  \
		openmp/bres_calc_kernel.cpp  bres_calc.h \
		openmp/update_kernel.cpp     update.h    \
                Makefile
		$(CPP) $(VAR) $(CPPFLAGS) $(OMPFLAGS) $(OP2_INC) $(OP2_LIB) -Iopenmp -I. \
		airfoil_op.cpp openmp/airfoil_kernels.cpp -lm -lop2_openmp -o airfoil_openmp

#
# CUDA version using kernel files generated by op2.py
#

airfoil_cuda:	airfoil_op.cpp cuda/airfoil_kernels_cu.o Makefile
		$(CPP) $(VAR) $(CPPFLAGS) $(CUDA_INC) $(OP2_INC) $(OP2_LIB) $(CUDA_LIB) \
		airfoil_op.cpp cuda/airfoil_kernels_cu.o -lcudart -lop2_cuda -o airfoil_cuda

cuda/airfoil_kernels_cu.o:	cuda/airfoil_kernels.cu      \
		cuda/save_soln_kernel.cu save_soln.h \
		cuda/adt_calc_kernel.cu  adt_calc.h  \
		cuda/res_calc_kernel.cu  res_calc.h  \
		cuda/bres_calc_kernel.cu bres_calc.h \
		cuda/update_kernel.cu    update.h    \
		Makefile
		nvcc $(VAR) $(INC) $(NVCCFLAGS) $(OP2_INC) \
		-Icuda -I. -c -o cuda/airfoil_kernels_cu.o \
		cuda/airfoil_kernels.cu

#
# codegenerated SYCL version
#

airfoil_sycl: airfoil_op.cpp sycl/airfoil_kernels.cpp \
                sycl/save_soln_kernel.cpp  save_soln.h \
                sycl/adt_calc_kernel.cpp   adt_calc.h  \
                sycl/res_calc_kernel.cpp   res_calc.h  \
                sycl/bres_calc_kernel.cpp  bres_calc.h \
                sycl/update_kernel.cpp     update.h    \
                Makefile
		$(SYCLCXX) $(VAR) $(CPPFLAGS) $(SYCL_FLAGS) $(OP2_INC) $(OP2_INC) \
                -Isycl -I. airfoil_op.cpp -lm sycl/airfoil_kernels.cpp $(OP2_LIB) \
		-lop2_sycl $(SYCL_LIB)  \
                -o airfoil_sycl



#
# mpi with sequential-nodes version
#

airfoil_mpi: airfoil_mpi.cpp save_soln.h adt_calc.h res_calc.h bres_calc.h Makefile
	$(MPICPP) $(MPIFLAGS) $(OP2_INC) $(PARMETIS_INC) $(PTSCOTCH_INC) \
	$(OP2_LIB) airfoil_mpi.cpp -lop2_mpi $(PARMETIS_LIB) $(PTSCOTCH_LIB) -o airfoil_mpi


#
# mpi code generated seq version using kernel files generated by op2.py
#

airfoil_mpi_genseq: airfoil_mpi_op.cpp seq/airfoil_mpi_seqkernels.cpp \
                seq/save_soln_seqkernel.cpp  save_soln.h \
                seq/adt_calc_seqkernel.cpp   adt_calc.h  \
                seq/res_calc_seqkernel.cpp   res_calc.h  \
                seq/bres_calc_seqkernel.cpp  bres_calc.h \
                seq/update_seqkernel.cpp     update.h    \
                Makefile
	        $(MPICPP) $(VAR) $(CPPFLAGS) $(OP2_INC) $(OP2_INC) \
                $(PARMETIS_INC) $(PTSCOTCH_INC) -Iseq -I. \
                airfoil_mpi_op.cpp -lm seq/airfoil_mpi_seqkernels.cpp $(OP2_LIB) -lop2_mpi \
                $(PARMETIS_LIB) $(PTSCOTCH_LIB) -o airfoil_mpi_genseq


#
# mpi vectorized seq version using kernel files generated by op2.py
#

airfoil_mpi_vec: airfoil_mpi_op.cpp vec/airfoil_mpi_veckernels.cpp \
                vec/save_soln_veckernel.cpp  save_soln.h \
                vec/adt_calc_veckernel.cpp   adt_calc.h  \
                vec/res_calc_veckernel.cpp   res_calc.h  \
                vec/bres_calc_veckernel.cpp  bres_calc.h \
                vec/update_veckernel.cpp     update.h    \
                Makefile
		$(MPICPP) -DVECTORIZE $(VAR) $(OMPFLAGS)  $(CPPFLAGS) $(OP2_INC) $(OP2_INC) \
                $(PARMETIS_INC) $(PTSCOTCH_INC) -Ivec -I. \
                airfoil_mpi_op.cpp -lm vec/airfoil_mpi_veckernels.cpp $(OP2_LIB) -lop2_mpi \
                $(PARMETIS_LIB) $(PTSCOTCH_LIB) -o airfoil_mpi_vec


#
# mpi with openmp-nodes version
#

airfoil_mpi_openmp: airfoil_mpi_op.cpp openmp/airfoil_mpi_kernels.cpp \
		    openmp/save_soln_kernel.cpp  save_soln.h \
		    openmp/adt_calc_kernel.cpp   adt_calc.h  \
		    openmp/res_calc_kernel.cpp   res_calc.h  \
		    openmp/bres_calc_kernel.cpp  bres_calc.h \
		    openmp/update_kernel.cpp     update.h    \
		    Makefile
		    $(MPICPP) $(MPIFLAGS) $(OMPFLAGS) airfoil_mpi_op.cpp openmp/airfoil_mpi_kernels.cpp \
		    $(OP2_INC) $(PARMETIS_INC) $(PTSCOTCH_INC) $(HDF5_INC) -Iopenmp -I. \
		    $(OP2_LIB) -lop2_mpi $(PARMETIS_LIB) $(PTSCOTCH_LIB) -o airfoil_mpi_openmp

#
# mpi with CUDA version
#

airfoil_mpi_cuda:	airfoil_mpi_op.cpp cuda/airfoil_kernels_mpi_cu.o Makefile
		  $(MPICPP) $(MPIFLAGS) airfoil_mpi_op.cpp cuda/airfoil_kernels_mpi_cu.o \
		  $(OP2_INC) $(PARMETIS_INC) $(PTSCOTCH_INC) \
		  $(OP2_LIB) -lop2_mpi_cuda $(PARMETIS_LIB) $(PTSCOTCH_LIB) \
		  $(CUDA_LIB) -lcudart -o airfoil_mpi_cuda

cuda/airfoil_kernels_mpi_cu.o:	cuda/airfoil_mpi_kernels.cu      \
		  cuda/save_soln_kernel.cu  save_soln.h \
		  cuda/adt_calc_kernel.cu   adt_calc.h  \
		  cuda/res_calc_kernel.cu   res_calc.h  \
		  cuda/bres_calc_kernel.cu  bres_calc.h \
		  cuda/update_kernel.cu     update.h    \
		  Makefile
		  nvcc  $(INC) $(NVCCFLAGS) $(OP2_INC) -I $(MPI_INSTALL_PATH)/include \
		  -I. -Icuda -c -o cuda/airfoil_kernels_mpi_cu.o cuda/airfoil_mpi_kernels.cu



#
# cleanup
#

clean:
		rm -f airfoil_seq airfoil_vec airfoil_openmp airfoil_cuda airfoil_mpi airfoil_mpi_vec \
		airfoil_mpi_cuda airfoil_mpi_openmp airfoil_genseq airfoil_mpi_genseq airfoil_mpi_cuda_hyb \
		airfoil_sycl *.o cuda/*.o
