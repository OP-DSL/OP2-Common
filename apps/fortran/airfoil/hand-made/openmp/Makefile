CC = gcc
FC = gfortran

OPENMP = -fopenmp

#IC = icc
#FC = ifort

#OPENMP = -openmp
#DEBUG = -debug full
PREPROCESSOR = -cpp

EXEC = airfoil_openmp
CPPLINK = -lstdc++
FLINK = -L$(F_OP2_LIB)

F_OP2_MOD = $(OP2)/fortran/mod
F_OP2_LIB = $(OP2)/fortran/lib

all: debug airfoil_openmp airfoil_link

debug: debug.c
	$(CC) $(DEBUG) -c debug.c

airfoil_openmp: airfoil_debug.f90 OP2Profiling.f90 input.f90 constvars.f90 airfoil_seqfun.f90 save_soln_openmp.f90 adt_calc_openmp.f90 res_calc_openmp.f90 bres_calc_openmp.f90 update_openmp.f90 airfoil.f90
	$(FC) $(DEBUG) $(OPENMP) $(PREPROCESSOR) -I$(F_OP2_MOD) -c airfoil_debug.f90 OP2Profiling.f90 input.f90 constvars.f90 airfoil_seqfun.f90 save_soln_openmp.f90 adt_calc_openmp.f90 res_calc_openmp.f90 bres_calc_openmp.f90 update_openmp.f90 airfoil.f90

airfoil_link: airfoil_debug.o OP2Profiling.o input.o constvars.o airfoil_seqfun.o save_soln_openmp.o adt_calc_openmp.o res_calc_openmp.o bres_calc_openmp.o update_openmp.o airfoil.o
	$(FC) $(DEBUG) $(FLINK) $(OPENMP) $(PREPROCESSOR) $(CPPLINK) airfoil_debug.o OP2Profiling.o debug.o input.o constvars.o airfoil_seqfun.o save_soln_openmp.o adt_calc_openmp.o res_calc_openmp.o bres_calc_openmp.o update_openmp.o airfoil.o -lop2_for_rt_support -lop2_for_openmp -o $(EXEC)

clean:
	rm -f *.o
	rm -f *.mod
	rm -f airfoil_seq