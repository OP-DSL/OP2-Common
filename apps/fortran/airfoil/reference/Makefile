#
# The following environment variables are required:
#
# CUDA_INSTALL_PATH - CUDA base directory
# OP2_COMPILER      - Compiler toolchain: gnu (gcc, gfortran) or pgi (icc, pgfortran)
# OP2               - OP2 base directory
#
# Note: the CUDA library can only be built with the PGI toolchain

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
  FFLAGS = -J$(F_OP2_MOD) -O4 -Wall -Werror -pedantic -pipe
else
ifeq ($(OP2_COMPILER),pgi)
  F_OP2_MOD = $(F_OP2)/mod/intel
  CC		= icc
  CFLAGS	= -g -O3 -std=c99 -Wall -Werror -vec-report -xSSE4.2
  FC      = ifort
  FFLAGS  = -module $(F_OP2_MOD) -g -O3 -vec-report -xSSE4.2
else
print:
	@echo "unrecognised value for OP2_COMPILER"
endif
endif

.PHONY: clean

all: c_extlib airfoil airfoil_link

c_extlib: debug.c
	$(CC) $(CFLAGS) -c debug.c

airfoil: $(F_OP2_MOD)/op2_fortran_declarations.mod $(F_OP2_MOD)/op2_fortran_reference.mod  airfoil_debug.f90 airfoil.f90 airfoil_seqfun.f90  constvars.f90  input.f90  output.f90
	$(FC) $(FFLAGS) -c constvars.f90 airfoil_debug.f90 airfoil_seqfun.f90 input.f90 output.f90 airfoil.f90


airfoil_link: debug.o airfoil_debug.o airfoil.o  airfoil_seqfun.o  constvars.o input.f90 output.o
	$(FC) $(FFLAGS) -L$(F_OP2_LIB) debug.o constvars.o airfoil_seqfun.o airfoil_debug.o input.o output.o airfoil.o -lop2_for_reference -o airfoil_seq

clean:
	-rm -f *.o
	-rm -f *.mod
	-rm -f airfoil_seq
