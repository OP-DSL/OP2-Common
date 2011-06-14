
#CC = icc -debug full
#FC = ifort -debug full

CC = gcc
FC = gfortran

FDEBUG = -Wtabs -g -O0 #  -Wall -pedantic
CDEBUG = -g -O0  -Wall -pedantic

F_OP2_LIB = $(OP2)/fortran/lib
F_OP2_MOD = $(OP2)/fortran/mod

# IC = 
# FC = ifort -debug full

.PHONY: clean

all: c_extlib airfoil airfoil_link

c_extlib: debug.c
	$(CC) $(CDEBUG) -c debug.c

airfoil: $(F_OP2_MOD)/op2_fortran_declarations.mod $(F_OP2_MOD)/op2_fortran_reference.mod airfoil.f90  airfoil_seqfun.f90  constvars.f90  input.f90  output.f90
	$(FC) $(FDEBUG) -fintrinsic-modules-path $(F_OP2_MOD) -c constvars.f90 airfoil_seqfun.f90 input.f90 output.f90 airfoil.f90 


airfoil_link: debug.o airfoil.o  airfoil_seqfun.o  constvars.o input.f90 output.o
	$(FC) $(FDEBUG) -L$(F_OP2_LIB) debug.o constvars.o airfoil_seqfun.o input.o output.o airfoil.o -lop2_for_reference -o airfoil_seq

clean:
	-rm -f *.o
	-rm -f *.mod
	-rm -f airfoil_seq