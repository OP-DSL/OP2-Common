PREPROCESSOR = -fpp
PART_SIZE_ENV ?= 256
OP2 = $(OP2_INSTALL_PATH)

F_OP2_MOD = $(OP2)/fortran/mod/$(OP2_COMPILER)
F_OP2_LIB = $(OP2)/fortran/lib

HDF5_LIB = -L$(HDF5_INSTALL_PATH)/lib -lhdf5 -lz -ldl

PARMETIS_INCLUDE_PATH ?= $(PARMETIS_INSTALL_PATH)
PARMETIS_LIB_PATH ?= $(PARMETIS_INSTALL_PATH)
PARMETIS_VER?=4
ifeq ($(PARMETIS_VER),4)
  PARMETIS_INC = -I$(PARMETIS_INCLUDE_PATH)/include #-DHAVE_PARMETIS -DPARMETIS_VER_4
  PARMETIS_LIB = -L$(PARMETIS_LIB_PATH)/lib -lparmetis -lmetis
else
  PARMETIS_INC = -I$(PARMETIS_INCLUDE_PATH)/ #-DHAVE_PARMETIS
  PARMETIS_LIB = -L$(PARMETIS_LIB_PATH)/ -lparmetis -lmetis
endif

PTSCOTCH_INC    = -I$(PTSCOTCH_INSTALL_PATH)/include #-DHAVE_PTSCOTCH
PTSCOTCH_LIB    = -L$(PTSCOTCH_INSTALL_PATH)/lib/ -lptscotch \
                  -L$(PTSCOTCH_INSTALL_PATH)/lib/ -lptscotcherr \
                  -L$(PTSCOTCH_INSTALL_PATH)/lib/ -lscotch \
                  -L$(PTSCOTCH_INSTALL_PATH)/lib/ -lscotcherr


#
# set flags for NVCC compilation and linking
#
ifndef NV_ARCH
  MESSAGE=select an NVIDA device to compile in CUDA, e.g. make NV_ARCH=KEPLER
  NV_ARCH=Kepler
endif
ifeq ($(NV_ARCH),Fermi)
  CODE_GEN_CUDA=cc20
else
ifeq ($(NV_ARCH),Kepler)
  CODE_GEN_CUDA=cc35
else
ifeq ($(NV_ARCH),Pascal)
  CODE_GEN_CUDA=cc60
endif
endif
endif

ifeq ($(OP2_COMPILER),gnu)
  CC = gcc
  FC = gfortran
  MPI_LIB = -L$(MPI_INSTALL_PATH)/lib -lmpi -lmpi_cxx
  MPI_INC = -I$(MPI_INSTALL_PATH)/include
  MPIF90 = $(FC) #$(MPI_INC) $(MPI_LIB)
  OPT = -O2 -DOP_PART_SIZE_1=$(PART_SIZE_ENV)
  OPENMP = -fopenmp
  CPPLINK = -lstdc++
  ALL_TARGETS = airfoil_hdf5_seq airfoil_hdf5_openmp \
                airfoil_hdf5_mpi airfoil_hdf5_mpi_genseq \
                airfoil_hdf5_mpi_openmp
else
ifeq ($(OP2_COMPILER),pgi)
  CC = pgc++
  FC = pgfortran 
  MPI_LIB = -L$(MPI_INSTALL_PATH)/lib -lmpi
  MPI_INC = $(MPI_INSTALL_PATH)/include
  MPIF90 = $(FC) -I$(MPI_INC) -L$(MPI_LIB)
  OPT = -DOP_PART_SIZE_1=$(PART_SIZE_ENV) -O3 -g 
  OPT_CUDA = -DOP_PART_SIZE_1=$(PART_SIZE_ENV) -O3 -Mcuda=$(CODE_GEN_CUDA),fastmath,ptxinfo -DOP2_WITH_CUDAFOR -Minline,reshape 
  OPT_ACC = -O3 -acc -ta=nvidia,$(CODE_GEN_CUDA) -Mcuda -Minfo=accel 
  OPENMP = -mp
  CPPLINK = -pgc++libs -lstdc++
  ALL_TARGETS = airfoil_hdf5_seq airfoil_hdf5_openmp \
                airfoil_hdf5_mpi airfoil_hdf5_mpi_genseq \
                airfoil_hdf5_mpi_openmp airfoil_hdf5_cuda \
                airfoil_hdf5_mpi_cuda airfoil_hdf5_openacc airfoil_hdf5_mpi_openacc
else
ifeq ($(OP2_COMPILER),xl)
  CC = xlc++_r
  #FC = pgfortran -fast -Mcuda=cuda5.0,cc35,fastmath,ptxinfo -Mx,189,0x40 #-Munroll -Minline=save_soln,res_calc,bres_calc,adt_calc -Mipa=fast -Mx,189,0x40
  FC = xlf_r 
  MPI_LIB = -L$(MPI_INSTALL_PATH)/lib -lmpi
  MPI_INC = #$(MPI_INSTALL_PATH)/include
  MPIF90 = mpif90 #$(FC) -I$(MPI_INC) -L$(MPI_LIB)
  OPT = -WF,-DOP_PART_SIZE_1=$(PART_SIZE_ENV) -O3 -qarch=pwr8 -qtune=pwr8 -qhot 
  OPT_CUDA = -WF,-DOP_PART_SIZE_1=$(PART_SIZE_ENV) -O3 -qcuda -qarch=pwr8 -qtune=pwr8 -qhot -qxflag=nrcptpo -WF,-DOP2_WITH_CUDAFOR
  OPT_ACC = -O2 -qarch=pwr8 -qtune=pwr8 -qhot -qsmp=omp -qoffload -WF,OP2_WITH_OMP4
  OPENMP = -qsmp=omp -qthreaded
  OMP_OFFLOAD = -qsmp=omp -qoffload -qreport
  CPPLINK = -lstdc++ -lmpi_cxx -L/opt/ibm/xlC/13.1.5/lib -libmc++
  ALL_TARGETS = airfoil_hdf5_seq airfoil_hdf5_openmp \
                airfoil_hdf5_mpi airfoil_hdf5_mpi_genseq \
                airfoil_hdf5_mpi_openmp airfoil_hdf5_cuda \
                airfoil_hdf5_mpi_cuda
else
ifeq ($(OP2_COMPILER),intel)
  CC = icc
  FC = ifort
  MPI_LIB = -L$(MPI_INSTALL_PATH)/lib -lmpi
  MPI_INC = -I$(MPI_INSTALL_PATH)/include
  MPIF90 =  $(FC) -I$(MPI_INC) -L$(MPI_LIB)
  OPENMP = -qopenmp
  #OPT = -DOP_PART_SIZE_1=$(PART_SIZE_ENV) -DMPICH_IGNORE_CXX_SEEK -restrict -fno-alias -inline-forceinline -qopt-report -parallel
  OPT = -DOP_PART_SIZE_1=$(PART_SIZE_ENV) -O3 -xHost -parallel -qopt-report -fno-alias -fp-model fast=2 -DVECTORIZE #-vec-threshold0 -align array32byte -align rec32byte -DVECTORIZE
  CPPLINK = -lstdc++
  ALL_TARGETS = airfoil_hdf5_seq airfoil_hdf5_vec airfoil_hdf5_openmp \
                airfoil_hdf5_mpi airfoil_hdf5_mpi_genseq airfoil_hdf5_mpi_vec\
                airfoil_hdf5_mpi_openmp
else
ifeq ($(OP2_COMPILER),cray)
  CC = CC
  FC = ftn -em -p $(F_OP2_MOD)
  MPI_LIB = #-L$(MPI_INSTALL_PATH)/lib -lmpi
  MPI_INC = #-I$(MPI_INSTALL_PATH)/include
  MPIF90 = $(FC) -I$(MPI_INC) -L$(MPI_LIB)
  OPT = -O2 -DOP_PART_SIZE_1=$(PART_SIZE_ENV)
  OPENMP = -h omp -lomp
  CPPLINK = -hnopgas_runtime -lpgas-dmapp
  ALL_TARGETS = airfoil_hdf5_seq airfoil_hdf5_openmp \
                airfoil_hdf5_mpi airfoil_hdf5_mpi_genseq \
                airfoil_hdf5_mpi_openmp
else
print:
	echo "unrecognised value for OP2_COMPILER"
endif
endif
endif
endif
endif

FLINK = -L$(F_OP2_LIB)
FMODS = -I$(F_OP2_MOD)
FMPI_MODS = -I$(F_OP2_MOD)/mpi

all: $(ALL_TARGETS)

airfoil_hdf5_seq: constants.F90 airfoil_seqfun.F90 airfoil_hdf5.F90 input.F90
	$(MPIF90) $(OPT) $(FMODS) -c constants.F90 input.F90 airfoil_seqfun.F90 airfoil_hdf5.F90
	$(MPIF90) $(OPT) constants.o input.o airfoil_hdf5.o airfoil_seqfun.o -o airfoil_hdf5_seq \
	$(FLINK) -lop2_for_seq -lop2_hdf5 $(HDF5_LIB) $(MPI_LIB) $(CPPLINK)

airfoil_hdf5_vec: airfoil_hdf5_op.F90 input.F90 save_soln_veckernel.F90 adt_calc_veckernel.F90 \
        res_calc_veckernel.F90 bres_calc_veckernel.F90 update_veckernel.F90 constants.F90 airfoil_seqfun.F90
	$(MPIF90) $(OPT) $(FMODS) -c constants.F90 input.F90 airfoil_seqfun.F90 \
        save_soln_veckernel.F90 update_veckernel.F90 bres_calc_veckernel.F90 \
        res_calc_veckernel.F90 adt_calc_veckernel.F90 airfoil_hdf5_op.F90
	$(MPIF90) $(OPT) input.o constants.o airfoil_seqfun.o \
        save_soln_veckernel.o adt_calc_veckernel.o res_calc_veckernel.o \
        bres_calc_veckernel.o update_veckernel.o airfoil_hdf5_op.o -o airfoil_hdf5_vec \
        $(FLINK) -lop2_for_seq -lop2_hdf5 $(HDF5_LIB) $(MPI_LIB) $(CPPLINK)


airfoil_hdf5_openmp: airfoil_hdf5_op.F90 save_soln_kernel.F90 adt_calc_kernel.F90 input.F90 \
	res_calc_kernel.F90 bres_calc_kernel.F90 update_kernel.F90 constants.F90 airfoil_seqfun.F90
	$(MPIF90) $(OPT) $(OPENMP) $(FMODS) -c constants.F90 input.F90 airfoil_seqfun.F90 \
	save_soln_kernel.F90 update_kernel.F90 bres_calc_kernel.F90 \
	res_calc_kernel.F90 adt_calc_kernel.F90 airfoil_hdf5_op.F90
	$(MPIF90) $(OPT) $(OPENMP) constants.o input.o airfoil_seqfun.o \
	save_soln_kernel.o adt_calc_kernel.o res_calc_kernel.o \
	bres_calc_kernel.o update_kernel.o airfoil_hdf5_op.o -o airfoil_hdf5_openmp \
		$(FLINK) -lop2_for_openmp -lop2_hdf5 $(HDF5_LIB) -lz $(MPI_LIB)

airfoil_hdf5_openmp4: airfoil_hdf5_op.F90 save_soln_omp4kernel.F90 adt_calc_omp4kernel.F90 input.F90 \
	res_calc_omp4kernel.F90 bres_calc_omp4kernel.F90 update_omp4kernel.F90 constants.F90 airfoil_seqfun.F90
	$(MPIF90) $(OPT) $(OMP_OFFLOAD) $(FMODS) -c constants.F90 input.F90 airfoil_seqfun.F90 \
	save_soln_omp4kernel.F90 update_omp4kernel.F90 bres_calc_omp4kernel.F90 \
	res_calc_omp4kernel.F90 adt_calc_omp4kernel.F90 airfoil_hdf5_op.F90
	$(MPIF90) $(OPT) $(OMP_OFFLOAD) constants.o input.o airfoil_seqfun.o \
	save_soln_omp4kernel.o adt_calc_omp4kernel.o res_calc_omp4kernel.o \
	bres_calc_omp4kernel.o update_omp4kernel.o airfoil_hdf5_op.o -o airfoil_hdf5_openmp4 \
		$(FLINK) -lop2_for_openmp4 -lop2_hdf5 $(HDF5_LIB) -lz $(MPI_LIB)

airfoil_hdf5_cuda: airfoil_hdf5_op.F90 save_soln_kernel.CUF adt_calc_kernel.CUF input.F90 airfoil_seqfun.F90 \
        res_calc_kernel.CUF bres_calc_kernel.CUF update_kernel.CUF constants.F90 input.F90 cudaConfigurationParams.F95
	$(FC) $(OPT_CUDA) $(FMODS)/cuda $(FMODS) cudaConfigurationParams.F95 constants.F90 input.F90 airfoil_seqfun.F90 \
		save_soln_kernel.CUF update_kernel.CUF bres_calc_kernel.CUF \
		res_calc_kernel.CUF adt_calc_kernel.CUF airfoil_hdf5_op.F90 -o airfoil_hdf5_cuda \
		$(FLINK) -lop2_for_cuda -lop2_hdf5 $(HDF5_LIB) $(MPI_LIB)

airfoil_hdf5_mpi: constants.F90 airfoil_seqfun.F90 airfoil_hdf5.F90 input.F90
	$(MPIF90) $(OPT) $(FMPI_MODS) -c constants.F90 airfoil_seqfun.F90 input.F90 airfoil_hdf5.F90
	$(MPIF90) $(OPT) $(PARMETIS_INC) $(PTSCOTCH_INC) constants.o input.o airfoil_hdf5.o \
	airfoil_seqfun.o -o airfoil_hdf5_mpi $(FLINK) -lop2_for_mpi $(PARMETIS_LIB) $(PTSCOTCH_LIB) $(HDF5_LIB) -lz $(MPI_LIB) $(CPPLINK)

airfoil_hdf5_mpi_genseq: airfoil_hdf5_op.F90 input.F90 save_soln_seqkernel.F90 adt_calc_seqkernel.F90 \
	res_calc_seqkernel.F90 bres_calc_seqkernel.F90 update_seqkernel.F90 constants.F90 airfoil_seqfun.F90
	$(MPIF90) $(OPT) $(OPENMP) $(FMPI_MODS) -c constants.F90 input.F90 airfoil_seqfun.F90 \
	save_soln_seqkernel.F90 update_seqkernel.F90 bres_calc_seqkernel.F90 \
	res_calc_seqkernel.F90 adt_calc_seqkernel.F90 airfoil_hdf5_op.F90
	$(MPIF90) $(OPT) $(OPENMP) $(PARMETIS_INC) $(PTSCOTCH_INC) input.o constants.o airfoil_seqfun.o \
	save_soln_seqkernel.o adt_calc_seqkernel.o res_calc_seqkernel.o \
	bres_calc_seqkernel.o update_seqkernel.o airfoil_hdf5_op.o -o airfoil_hdf5_mpi_genseq \
	$(FLINK) -lop2_for_mpi $(PARMETIS_LIB) $(PTSCOTCH_LIB) $(HDF5_LIB) -lz $(MPI_LIB) $(CPPLINK)

airfoil_hdf5_mpi_vec: airfoil_hdf5_op.F90 input.F90 save_soln_veckernel.F90 adt_calc_veckernel.F90 \
        res_calc_veckernel.F90 bres_calc_veckernel.F90 update_veckernel.F90 constants.F90 airfoil_seqfun.F90
	$(MPIF90) $(OPT) $(FMPI_MODS) -c constants.F90 input.F90 airfoil_seqfun.F90 \
        save_soln_veckernel.F90 update_veckernel.F90 bres_calc_veckernel.F90 \
        res_calc_veckernel.F90 adt_calc_veckernel.F90 airfoil_hdf5_op.F90
	$(MPIF90) $(OPT) $(PARMETIS_INC) $(PTSCOTCH_INC) input.o constants.o airfoil_seqfun.o \
        save_soln_veckernel.o adt_calc_veckernel.o res_calc_veckernel.o \
        bres_calc_veckernel.o update_veckernel.o airfoil_hdf5_op.o -o airfoil_hdf5_mpi_vec \
        $(FLINK) -lop2_for_mpi $(PARMETIS_LIB) $(PTSCOTCH_LIB) $(HDF5_LIB) -lz $(MPI_LIB) $(CPPLINK)

airfoil_hdf5_mpi_openmp: airfoil_hdf5_op.F90 input.F90 save_soln_kernel.F90 adt_calc_kernel.F90 \
	res_calc_kernel.F90 bres_calc_kernel.F90 update_kernel.F90 constants.F90 airfoil_seqfun.F90
	$(MPIF90) $(OPT) $(OPENMP) $(FMPI_MODS) -c constants.F90 input.F90 airfoil_seqfun.F90 \
	save_soln_kernel.F90 update_kernel.F90 bres_calc_kernel.F90 \
	res_calc_kernel.F90 adt_calc_kernel.F90 airfoil_hdf5_op.F90
	$(MPIF90) $(OPT) $(OPENMP) $(PARMETIS_INC) $(PTSCOTCH_INC) input.o constants.o airfoil_seqfun.o \
	save_soln_kernel.o adt_calc_kernel.o res_calc_kernel.o \
	bres_calc_kernel.o update_kernel.o airfoil_hdf5_op.o -o airfoil_hdf5_mpi_openmp \
	$(FLINK) -lop2_for_mpi $(PARMETIS_LIB) $(PTSCOTCH_LIB) $(HDF5_LIB) -lz $(MPI_LIB) $(CPPLINK)

airfoil_hdf5_mpi_cuda: airfoil_hdf5_op.F90 save_soln_kernel.CUF adt_calc_kernel.CUF input.F90 airfoil_seqfun.F90 \
	res_calc_kernel.CUF bres_calc_kernel.CUF update_kernel.CUF constants.F90 input.F90 cudaConfigurationParams.F95
	$(MPIF90) $(OPT_CUDA) $(FMPI_MODS)/cuda $(FMPI_MODS) $(PARMETIS_INC) $(PTSCOTCH_INC) \
		cudaConfigurationParams.F95 constants.F90 input.F90 airfoil_seqfun.F90 save_soln_kernel.CUF \
		update_kernel.CUF bres_calc_kernel.CUF res_calc_kernel.CUF adt_calc_kernel.CUF \
		airfoil_hdf5_op.F90 -o airfoil_hdf5_mpi_cuda $(FLINK) -lop2_for_mpi_cuda \
		$(PARMETIS_LIB) $(PTSCOTCH_LIB) $(HDF5_LIB) -lz $(MPI_LIB) $(CPPLINK)

airfoil_hdf5_openacc: airfoil_hdf5_op.F90 save_soln_acckernel.F90 adt_calc_acckernel.F90 input.F90 airfoil_seqfun.F90 \
	res_calc_acckernel.F90 bres_calc_acckernel.F90 update_acckernel.F90 constants.F90 input.F90
	$(MPIF90) $(OPT_ACC) $(FMPI_MODS)/cuda $(FMPI_MODS) \
		constants.F90 input.F90 airfoil_seqfun.F90 save_soln_acckernel.F90 \
		update_acckernel.F90 bres_calc_acckernel.F90 res_calc_acckernel.F90 adt_calc_acckernel.F90 \
		airfoil_hdf5_op.F90 -o airfoil_hdf5_openacc $(FLINK) -lop2_for_cuda -lop2_hdf5 \
		$(HDF5_LIB) -lz $(MPI_LIB) $(CPPLINK)

airfoil_hdf5_mpi_openacc: airfoil_hdf5_op.F90 save_soln_acckernel.F90 adt_calc_acckernel.F90 input.F90 airfoil_seqfun.F90 \
	res_calc_acckernel.F90 bres_calc_acckernel.F90 update_acckernel.F90 constants.F90 input.F90
	$(MPIF90) $(OPT_ACC) $(FMPI_MODS)/cuda $(FMPI_MODS) \
		constants.F90 input.F90 airfoil_seqfun.F90 save_soln_acckernel.F90 \
		update_acckernel.F90 bres_calc_acckernel.F90 res_calc_acckernel.F90 adt_calc_acckernel.F90 \
		airfoil_hdf5_op.F90 -o airfoil_hdf5_mpi_openacc $(FLINK) -lop2_for_mpi_cuda  \
		$(PARMETIS_LIB) $(PTSCOTCH_LIB) $(HDF5_LIB) -lz $(MPI_LIB) $(CPPLINK)




clean:
	rm -f *.o
	rm -f *.mod
	rm -f $(EXEC)
	rm -f *~
	rm -f airfoil_hdf5_seq
	rm -f airfoil_hdf5_mpi
	rm -f airfoil_hdf5_openacc
	rm -f airfoil_hdf5_openmp airfoil_hdf5_mpi_openmp airfoil_hdf5_openmp_* airfoil_hdf5_mpi_openmp_*
	rm -f airfoil_hdf5_mpi_genseq
	rm -f airfoil_hdf5_mpi_vec
	rm -f airfoil_hdf5_cuda airfoil_hdf5_mpi_cuda
