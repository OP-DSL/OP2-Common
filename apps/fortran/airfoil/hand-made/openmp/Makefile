#
# The following environment variables are required:
#
# OP2_COMPILER      - Compiler toolchain: gnu (gcc, gfortran) or pgi (icc, pgfortran)
# OP2               - OP2 base directory

F_OP2 = $(OP2)/fortran
F_OP2_INC = $(F_OP2)/include
F_OP2_SRC = $(F_OP2)/src
F_OP2_OBJ = $(F_OP2)/obj
F_OP2_LIB = $(F_OP2)/lib
F_OP2_MOD = $(F_OP2)/mod

#
# compiler settings
#

ifeq ($(OP2_COMPILER),gnu)
  CC		= gcc
  CFLAGS = -g -O3 -std=c99 -Wall -Werror -fPIC -pedantic -pipe
  FC = gfortran
  FFLAGS = -J$(F_OP2_MOD) -g -O4 -Wall -Werror -pedantic -pipe
  OPENMP = -fopenmp
else
ifeq ($(OP2_COMPILER),pgi)
  F_OP2_MOD = $(F_OP2)/mod/intel
  CC		= icc
  CFLAGS	= -g -O3 -std=c99 -Wall -Werror -vec-report -xSSE4.2
  FC      = ifort
  FFLAGS  = -module $(F_OP2_MOD) -g -O3 -vec-report -xSSE4.2
  DEBUG = -debug full
  OPENMP = -openmp
else
print:
	@echo "unrecognised value for OP2_COMPILER"
endif
endif

PREPROCESSOR = -cpp

EXEC = airfoil_openmp
CPPLINK = -lstdc++
FLINK = -L$(F_OP2_LIB)

.PHONY: clean

all: debug airfoil_openmp airfoil_link

debug: debug.c
	$(CC) $(DEBUG) -c debug.c

airfoil_openmp: airfoil_debug.f90 OP2Profiling.f90 input.f90 constvars.f90 airfoil_seqfun.f90 save_soln_openmp.f90 adt_calc_openmp.f90 res_calc_openmp.f90 bres_calc_openmp.f90 update_openmp.f90 airfoil.f90
	$(FC) $(DEBUG) $(OPENMP) $(PREPROCESSOR) -I$(F_OP2_MOD) -c airfoil_debug.f90 OP2Profiling.f90 input.f90 constvars.f90 airfoil_seqfun.f90 save_soln_openmp.f90 adt_calc_openmp.f90 res_calc_openmp.f90 bres_calc_openmp.f90 update_openmp.f90 airfoil.f90

airfoil_link: airfoil_debug.o OP2Profiling.o input.o constvars.o airfoil_seqfun.o save_soln_openmp.o adt_calc_openmp.o res_calc_openmp.o bres_calc_openmp.o update_openmp.o airfoil.o
	$(FC) $(DEBUG) $(FLINK) $(OPENMP) $(PREPROCESSOR) $(CPPLINK) airfoil_debug.o OP2Profiling.o debug.o input.o constvars.o airfoil_seqfun.o save_soln_openmp.o adt_calc_openmp.o res_calc_openmp.o bres_calc_openmp.o update_openmp.o airfoil.o -lop2_for_openmp_rt_support -lop2_for_openmp -o $(EXEC)

clean:
	-rm -f *.o
	-rm -f *.mod
	-rm -f airfoil_seq
	-rm -f *~
