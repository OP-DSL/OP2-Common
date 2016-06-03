PREPROCESSOR = -fpp

CPPLINK = -lstdc++
OP2 = $(OP2_INSTALL_PATH)

F_OP2_MOD = $(OP2)/fortran/mod/$(OP2_COMPILER)
F_OP2_LIB = $(OP2)/fortran/lib

HDF5_LIB = -L$(HDF5_INSTALL_PATH)/lib -lhdf5
MPI_LIB = -L$(MPI_INSTALL_PATH)/lib -lmpi
PARMETIS_INC 	= -I$(PARMETIS_INSTALL_PATH) -DHAVE_PARMETIS
PARMETIS_LIB 	= -L$(PARMETIS_INSTALL_PATH) -lparmetis \
            -L$(PARMETIS_INSTALL_PATH) -lmetis

PTSCOTCH_INC 	= -I$(PTSCOTCH_INSTALL_PATH)/include -DHAVE_PTSCOTCH
PTSCOTCH_LIB 	= -L$(PTSCOTCH_INSTALL_PATH)/lib/ -lptscotch \
                  -L$(PTSCOTCH_INSTALL_PATH)/lib/ -lptscotcherr

PART_SIZE_ENV ?= 128

ifeq ($(OP2_COMPILER),gnu)
  CC = gcc
  FC = gfortran
  MPI_LIB = #-L$(MPI_INSTALL_PATH)/lib -lmpi
  MPI_INC = #-I$(MPI_INSTALL_PATH)/include
  MPIF90 = $(FC) -I$(MPI_INC) -L$(MPI_LIB)
  OPT = -O2 -DOP_PART_SIZE_1=$(PART_SIZE_ENV)
  OPENMP = -fopenmp
  CPPLINK = -lstdc++
  ALL_TARGET = airfoil_seq airfoil_openmp
else
ifeq ($(OP2_COMPILER),pgi)
  CC = pgcc
  FC = pgfortran -fast -Mcuda=cuda7.5,cc35,fastmath,ptxinfo
  MPI_LIB = -L$(MPI_INSTALL_PATH)/lib -lmpi
  MPI_INC = -I$(MPI_INSTALL_PATH)/include
  MPIF90 = $(FC) $(MPI_INC) $(MPI_LIB)
  OPT = -DOP_PART_SIZE_1=$(PART_SIZE_ENV) -O2 
  OPENMP = -mp
  CPPLINK = -pgcpplibs
  ALL_TARGET = airfoil_seq airfoil_openmp airfoil_cuda
else
ifeq ($(OP2_COMPILER),intel)
  IC = icc
  FC = ifort
  #OPT = -DOP_PART_SIZE_1=$(PART_SIZE_ENV) -O2 -xSSE4.2 -vec-report
  MPI_LIB = -L$(MPI_INSTALL_PATH)/lib -lmpi
  MPI_INC = -I$(MPI_INSTALL_PATH)/include
  MPIF90 = $(FC) $(MPI_INC) $(MPI_LIB)
  OPENMP = -openmp
  OPT = -DOP_PART_SIZE_1=$(PART_SIZE_ENV) -O3 -xAVX -parallel -qopt-report -fno-alias -inline-forceinline -vec-threshold0 -align array32byte -align rec32byte -DVECTORIZE
  #OPT = -DOP_PART_SIZE_1=$(PART_SIZE_ENV) -O3 -xMIC-AVX512 -parallel -qopt-report=5 -fno-alias -inline-forceinline -vec-threshold0 -align array32byte -align rec32byte -DVECTORIZE
  CPPLINK = -lstdc++
  ALL_TARGET = airfoil_seq airfoil_genseq airfoil_vec airfoil_openmp
else
print:
	echo "unrecognised value for OP2_COMPILER"
endif
endif
endif

FLINK = -L$(F_OP2_LIB)
FMODS = -I$(F_OP2_MOD)
FMPI_MODS = -I$(F_OP2_MOD)/mpi

all: $(ALL_TARGET)

airfoil_seq: airfoil_seqfun.F90 airfoil.F90 constants.F90 input.F90
	$(FC) $(OPT) $(FMODS) -c constants.F90 airfoil_seqfun.F90 input.F90 airfoil.F90
	$(MPIF90) $(OPT) $(FLINK) constants.o airfoil.o input.o airfoil_seqfun.o -o airfoil_seq -lop2_for_seq -lop2_hdf5 $(HDF5_LIB)

airfoil_genseq: airfoil_op.F90 input.F90 save_soln_seqkernel.F90 adt_calc_seqkernel.F90 \
        res_calc_seqkernel.F90 bres_calc_seqkernel.F90 update_seqkernel.F90 constants.F90 airfoil_seqfun.F90
	$(FC) $(OPT) $(FMODS) -c constants.F90 input.F90 airfoil_seqfun.F90 \
        save_soln_seqkernel.F90 update_seqkernel.F90 bres_calc_seqkernel.F90 \
        res_calc_seqkernel.F90 adt_calc_seqkernel.F90 airfoil_op.F90
	$(MPIF90) $(OPT) $(OPENMP) $(FLINK) input.o constants.o airfoil_seqfun.o \
        save_soln_seqkernel.o adt_calc_seqkernel.o res_calc_seqkernel.o \
        bres_calc_seqkernel.o update_seqkernel.o airfoil_op.o -o airfoil_genseq -lop2_for_seq -lop2_hdf5 $(HDF5_LIB)

airfoil_vec: airfoil_op.F90 input.F90 save_soln_veckernel.F90 adt_calc_veckernel.F90 \
        res_calc_veckernel.F90 bres_calc_veckernel.F90 update_veckernel.F90 constants.F90 airfoil_seqfun.F90
	$(FC) $(OPT) $(FMODS) -c constants.F90 input.F90 airfoil_seqfun.F90 \
        save_soln_veckernel.F90 update_veckernel.F90 bres_calc_veckernel.F90 \
        res_calc_veckernel.F90 adt_calc_veckernel.F90 airfoil_op.F90
	$(MPIF90) $(OPT) $(OPENMP) $(FLINK) input.o constants.o airfoil_seqfun.o \
        save_soln_veckernel.o adt_calc_veckernel.o res_calc_veckernel.o \
        bres_calc_veckernel.o update_veckernel.o airfoil_op.o -o airfoil_vec -lop2_for_seq -lop2_hdf5 $(HDF5_LIB)


airfoil_openmp: airfoil_op.F90 save_soln_kernel.F90 adt_calc_kernel.F90 \
	res_calc_kernel.F90 bres_calc_kernel.F90 update_kernel.F90 constants.F90 input.F90 airfoil_seqfun.F90
	$(FC) $(OPT) $(OPENMP) $(FMODS) -c constants.F90 input.F90 airfoil_seqfun.F90 \
	save_soln_kernel.F90 update_kernel.F90 bres_calc_kernel.F90 \
	res_calc_kernel.F90 adt_calc_kernel.F90 airfoil_op.F90
	$(MPIF90) $(OPT) $(OPENMP) $(FLINK) constants.o input.o airfoil_seqfun.o \
	save_soln_kernel.o adt_calc_kernel.o res_calc_kernel.o \
	bres_calc_kernel.o update_kernel.o airfoil_op.o -o airfoil_openmp -lop2_for_openmp -lop2_hdf5 $(HDF5_LIB)

airfoil_cuda: airfoil_op.F90 save_soln_kernel.CUF adt_calc_kernel.CUF input.F90 airfoil_seqfun.F90 \
	res_calc_kernel.CUF bres_calc_kernel.CUF update_kernel.CUF constants.F90 input.F90 cudaConfigurationParams.F95
ifeq ($(OP2_COMPILER),pgi)
	$(FC) $(OPT) $(OPENMP) $(FMODS)/cuda $(FMODS) $(FLINK) -DOP2_WITH_CUDAFOR cudaConfigurationParams.F95 constants.F90 input.F90 airfoil_seqfun.F90 \
	save_soln_kernel.CUF update_kernel.CUF bres_calc_kernel.CUF \
	res_calc_kernel.CUF adt_calc_kernel.CUF airfoil_op.F90 -o airfoil_cuda -lop2_for_cuda -lop2_hdf5 $(HDF5_LIB)
else
	@echo "airfoil_hdf5_cuda cannot be built with $(OP2_COMPILER) compilers .. requires PGI CUDA FORTRAN compiler"
endif



clean:
	rm -f *.o
	rm -f *.mod
	rm -f $(EXEC)
	rm -f *~
	rm -f airfoil_genseq
	rm -f airfoil_seq
	rm -f airfoil_vec
	rm -f airfoil_openmp
	rm -f airfoil_openmp_*
	rm -f airfoil_cuda
	rm -f airfoil_vec
	rm -f airfoil_genseq

