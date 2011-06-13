
# requires definition of environment variable OP2, pointing to base OP2 directory

C_OP2 = $(OP2)/c
C_OP2_INC = $(C_OP2)/include
C_OP2_SRC = $(C_OP2)/src

F_OP2 = $(OP2)/fortran
F_OP2_INC = $(F_OP2)/include
F_OP2_SRC = $(F_OP2)/src
F_OP2_OBJ = $(F_OP2)/obj
F_OP2_LIB = $(F_OP2)/lib

F_OP2_USER = $(F_OP2_SRC)/user
F_OP2_BACKEND = $(F_OP2_SRC)/backend

CC = gcc
FCC = gfortran

CFLAGS = -Wall
FFLAGS = -Wall

all: c_corelib f_corelib f_reference f_rt_support

c_corelib: $(C_OP2_SRC)/op_lib_core.cpp $(C_OP2_INC)/op_lib_core.h
	$(CC) -I$(C_OP2_INC) -c $(C_OP2_SRC)/op_lib_core.cpp -o $(F_OP2_OBJ)/op_lib_core.o

f_corelib: $(F_OP2_SRC)/backend/op2_for_declarations.f90 $(F_OP2_SRC)/backend/op2_for_C_wrappers.c
	$(CC) -I$(C_OP2_INC) -c $(F_OP2_SRC)/backend/op2_for_C_wrappers.c -o $(F_OBJ)/op2_for_C_wrappers.o
	$(FCC) -c $(F_OP2_SRC)/backend/op2_for_declarations.f90 -o $(F_OBJ)/op2_for_declarations.o

c_rt_support: $(C_OP2_SRC)/op_rt_support.cpp $(C_OP2_INC)/op_rt_support.h
	$(CC) -c $(C_OP2_SRC)/op_rt_support.cpp -o $(F_OP2_LIB)/op_rt_support.o

f_rt_support: c_rt_support $(F_OP2_BACKEND)/op2_for_rt_support.f90 $(F_OP2_BACKEND)/op2_for_rt_wrappers.c
	$(FCC) -c $(F_OP2_BACKEND)/op2_for_rt_support.f90 -o $(F_OP2_OBJ)/op2_for_rt_support.o
	ar -r $(F_OP2_LIB)/lib_op2_f_rt_support.a \
	$(F_OP2_LIB)/op_rt_support.o \
	$(F_OP2_OBJ)/op2_for_rt_support.o

# does not depend on reference.h because we don't need the op_par_loop templated implementation, but our own implementation
f_reference: $(C_OP2_INC)/op_lib_core.h $(C_OP2_SRC)/op_lib_core.cpp \
						 $(C_OP2_SRC)/op_reference.cpp \
						 $(F_OP2_SRC)/backend/op2_for_C_wrappers.c \
						 $(F_OP2_SRC)/backend/op2_for_declarations.f90 \
						 $(F_OP2_SRC)/backend/op2_for_reference.f90 \
						 $(F_OP2_SRC)/backend/op2_C_reference.c
	$(CC) -I$(C_OP2_INC) -c $(C_OP2_SRC)/op_lib_core.cpp -o $(F_OP2_OBJ)/op_lib_core.o
	$(CC) -I$(C_OP2_INC) -c $(C_OP2_SRC)/op_reference.cpp -o $(F_OP2_OBJ)/op_reference.o
	$(CC) -I$(C_OP2_INC) -c $(F_OP2_SRC)/backend/op2_for_C_wrappers.c -o $(F_OP2_OBJ)/op2_for_C_wrappers.o
	$(FCC) -c $(F_OP2_SRC)/backend/op2_for_declarations.f90 -o $(F_OP2_SRC)/backend/op2_for_declarations.o
	$(FCC) -c $(F_OP2_SRC)/backend/op2_for_reference.f90 -o $(F_OP2_SRC)/backend/op2_for_reference.o
	$(CC) -I$(C_OP2_INC) -c $(C_OP2_SRC)/op2_C_reference.c -o $(F_OP2_OBJ)/op2_C_reference.o
	ar -r $(F_OP2_LIB)/libop2_f_reference.a \
				$(F_OP2_OBJ)/op_lib_core.o \
				$(F_OP2_OBJ)/op_reference.o \
				$(F_OP2_OBJ)/op2_for_C_wrappers.o \
				$(F_OP2_SRC)/backend/op2_for_declarations.o \
				$(F_OP2_OBJ)/op2_C_reference.o

c_cuda_wrappers: $(C_OP2_SRC)/cuda/op_cuda_decl.cpp
	$(CC) -c $(C_OP2_SRC)/cuda/op_cuda_decl.cpp -o $(F_OBJ)/op_cuda_decl.o

f_cuda: c_core_lib c_cuda_wrappers f_rt_support
	ar -r $(F_OP2_LIB)/libop2_f_cuda.a \
	$(F_OP2_OBJ)/op_lib_core.o \
	$(F_OP2_OBJ)/op2_for_C_wrappers.o \
	$(F_OBJ)/op2_for_declarations.o
