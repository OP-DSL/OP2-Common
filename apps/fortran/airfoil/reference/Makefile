#CC = icc -debug full
#FC = ifort -debug full

OP2_C_CODE = $(OP2)/c/
OP2_INC = $(OP2_C_CODE)/include
OP2_SRC = $(OP2_C_CODE)/src

CC = gcc 
FC = gfortran

STD = -std=c99
FDEBUG = -Wtabs -g -O0 #  -Wall -pedantic
CDEBUG = -g -O0  -Wall -pedantic

# IC = 
# FC = ifort -debug full

all: c_corelib c_extlib c_reflib airfoil_seq link

c_corelib: $(OP2_SRC)/op_lib_core.cpp $(OP2_INCLUDE)/op_lib_core.h
	$(CC) $(STD) $(CDEBUG) -c op_lib_core.cpp -o 

c_extlib: debug.c op_fortran_wrappers.c
	$(CC) $(CDEBUG) -c debug.c op_fortran_wrappers.c

c_reflib: op_seq.h op_seq.c
	$(CC) $(CDEBUG) -c op_seq.c

airfoil_seq:  constvars.f90 op2_fortran.f90 airfoil_seqfun.f90 input.f90 output.f90 airfoil.f90
	$(FC) $(FDEBUG) -c constvars.f90 op2_fortran.f90 airfoil_seqfun.f90 input.f90 output.f90 airfoil.f90

link: debug.o constvars.o airfoil.o airfoil_seqfun.o op2_fortran.o input.o output.o  op_seq.o
	$(FC) $(FDEBUG) op_lib_core.o op_seq.o op_fortran_wrappers.o debug.o constvars.o op2_fortran.o airfoil_seqfun.o input.o output.o airfoil.o -o airfoil_seq
	
clean:
	rm -f *.o
	rm -f *.mod
	rm -f airfoil_seq