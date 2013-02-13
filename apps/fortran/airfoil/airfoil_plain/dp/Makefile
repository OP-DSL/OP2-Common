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


ifeq ($(OP2_COMPILER),gnu)

else
ifeq ($(OP2_COMPILER),pgi)
IC = pgcc
FC = pgfortran -Mcuda=cuda4.2
OPT = -DOP_PART_SIZE_1=$(PART_SIZE_ENV) -O2
OPENMP = -mp
else
ifeq ($(OP2_COMPILER),intel)
IC = icc
FC = /opt/intel/Compiler/11.1/064/bin/intel64/ifort
OPENMP = -openmp
OPT = -DOP_PART_SIZE_1=$(PART_SIZE_ENV) -O2 -xSSE4.2 -vec-report
else
print:
	echo "unrecognised value for OP2_COMPILER"
endif
endif
endif

FLINK = -L$(F_OP2_LIB)
FMODS = -I$(F_OP2_MOD)

all: clean airfoil_seq airfoil_openmp_$(PART_SIZE_ENV)

airfoil_seq: airfoil_seqfun.F90 airfoil.F90 constants.F90
	$(FC) $(OPT) $(FMODS) -c constants.F90 airfoil_seqfun.F90 airfoil.F90
	$(FC) $(OPT) $(FLINK) constants.o airfoil.o airfoil_seqfun.o -o airfoil_seq -lop2_for_seq

airfoil_openmp_$(PART_SIZE_ENV): airfoil_op.F90 save_soln_kernel.F90 adt_calc_kernel.F90 \
	res_calc_kernel.F90 bres_calc_kernel.F90 update_kernel.F90 constants.F90 airfoil_seqfun.F90
	$(FC) $(OPT) $(OPENMP) $(FMODS) -c constants.F90 \
	save_soln_kernel.F90 update_kernel.F90 bres_calc_kernel.F90 \
	res_calc_kernel.F90 adt_calc_kernel.F90 airfoil_op.F90 airfoil_seqfun.F90
	$(FC) $(OPT) $(OPENMP) $(FLINK) constants.o airfoil_seqfun.o \
	save_soln_kernel.o adt_calc_kernel.o res_calc_kernel.o \
	bres_calc_kernel.o update_kernel.o airfoil_op.o -o airfoil_openmp_$(PART_SIZE_ENV) -lop2_for_openmp

clean:
	rm -f *.o
	rm -f *.mod
	rm -f $(EXEC)
	rm -f *~
	rm -f airfoil_seq
	rm -f airfoil_openmp_$(PART_SIZE_ENV)
