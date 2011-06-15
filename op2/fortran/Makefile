
# requires definition of environment variable OP2, pointing to base OP2 directory

C_OP2 = $(OP2)/c
C_OP2_INC = $(C_OP2)/include
C_OP2_SRC = $(C_OP2)/src

F_OP2 = $(OP2)/fortran
F_OP2_INC = $(F_OP2)/include
F_OP2_SRC = $(F_OP2)/src
F_OP2_OBJ = $(F_OP2)/obj
F_OP2_LIB = $(F_OP2)/lib
F_OP2_MOD = $(F_OP2)/mod

F_OP2_USER = $(F_OP2_SRC)/user
F_OP2_BACKEND = $(F_OP2_SRC)/backend 

CUDA_INC = $(CUDA_INSTALL_PATH)/include

CC = gcc -g
FCC = gfortran -g

CFLAGS = -Wall
FFLAGS = -J$(F_OP2_MOD)


.PHONY: clean

all: f_reference f_cuda

# does not depend on reference.h because we don't need the op_par_loop templated implementation, but our own implementation
f_reference: $(C_OP2_INC)/op_lib_core.h $(C_OP2_SRC)/op_lib_core.c \
						 $(C_OP2_SRC)/op_reference.c \
						 $(F_OP2_SRC)/backend/op2_for_C_wrappers.c \
						 $(F_OP2_SRC)/backend/op2_for_declarations.f90 \
						 $(F_OP2_SRC)/backend/op2_for_reference.f90 \
						 $(F_OP2_SRC)/backend/op2_C_reference.c
	$(CC) -std=c99 -I$(C_OP2_INC) -c $(C_OP2_SRC)/op_lib_core.c -o $(F_OP2_OBJ)/op_lib_core.o
	$(CC) -std=c99 -I$(C_OP2_INC) -c $(C_OP2_SRC)/op_reference.c -o $(F_OP2_OBJ)/op_reference.o
	$(CC) -I$(C_OP2_INC) -c $(F_OP2_SRC)/backend/op2_for_C_wrappers.c -o $(F_OP2_OBJ)/op2_for_C_wrappers.o
	$(FCC) $(FFLAGS) -c $(F_OP2_SRC)/backend/op2_for_declarations.f90 -o $(F_OP2_OBJ)/op2_for_declarations.o
	$(FCC) $(FFLAGS) -c $(F_OP2_SRC)/backend/op2_for_reference.f90 -o $(F_OP2_OBJ)/op2_for_reference.o
	$(CC) -I$(C_OP2_INC) -c $(F_OP2_SRC)/backend/op2_C_reference.c -o $(F_OP2_OBJ)/op2_C_reference.o
	ar -r $(F_OP2_LIB)/libop2_for_reference.a \
				$(F_OP2_OBJ)/op_lib_core.o \
				$(F_OP2_OBJ)/op_reference.o \
				$(F_OP2_OBJ)/op2_for_reference.o \
				$(F_OP2_OBJ)/op2_for_C_wrappers.o \
				$(F_OP2_OBJ)/op2_for_declarations.o \
				$(F_OP2_OBJ)/op2_C_reference.o

f_cuda: $(C_OP2_INC)/op_lib_core.h $(C_OP2_SRC)/op_lib_core.c \
				$(C_OP2_INC)/op_cuda_rt_support.h $(C_OP2_SRC)/cuda/op_cuda_rt_support.c \
				$(C_OP2_SRC)/cuda/op_cuda_decl.c
	$(CC) $(FLAGS) -I$(CUDA_INSTALL_PATH)/include  -I$(CUDA_INC) -c $(C_OP2_SRC)/cuda/op_cuda_decl.c -o $(F_OP2_OBJ)/cuda/op_cuda_decl.o
	$(CC) $(FLAGS) -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_INC) -c $(C_OP2_SRC)/cuda/op_cuda_rt_support.c -o $(F_OP2_OBJ)/cuda/op_cuda_rt_support.o
	$(CC) -I$(C_OP2_INC) -c $(F_OP2_SRC)/backend/op2_for_C_wrappers.c -o $(F_OP2_OBJ)/op2_for_C_wrappers.o
	$(FCC) $(FFLAGS) -c $(F_OP2_SRC)/backend/op2_for_declarations.f90 -o $(F_OP2_OBJ)/op2_for_declarations.o
	ar -r $(F_OP2_LIB)/libop2_for_cuda.a \
	$(F_OP2_OBJ)/$(F_OP2_OBJ)/cuda/op_cuda_decl.o \
	$(F_OP2_OBJ)/cuda/op_cuda_rt_support.o \
	$(F_OP2_OBJ)/op2_for_C_wrappers.o \
	$(F_OP2_OBJ)/op2_for_declarations.o

f_rt_support: $(C_OP2_INC)/op_rt_support.h $(C_OP2_INC)/op_rt_support.c \
							$(F_OP2_SRC)/op2_for_rt_wrappers.c $(F_OP2_SRC)/op2_for_rt_support.f90
	$(CC) $(FLAGS) -I(C_OP2_INC) -c $(C_OP2_INC)/op_rt_support.c -o $(F_OP2_OBJ)/op_rt_support.o
	$(CC) $(FLAGS) -I(C_OP2_INC) -c $(F_OP2_SRC)/op2_for_rt_wrappers.c -o $(F_OP2_OBJ)/op2_for_rt_wrappers.o
	$(FCC) $(FFLAGS) -c $(F_OP2_SRC)/op2_for_rt_support.f90 -o $(F_OP2_OBJ)/op2_for_rt_support.o
	ar -r $(F_OP2_LIB)/libop2_for_rt_support.a \
		$(F_OP2_OBJ)/op_rt_support.o \
		$(F_OP2_OBJ)/op2_for_rt_wrappers.o \
		$(F_OP2_OBJ)/op2_for_rt_support.o

clean:
	-rm $(F_OP2_OBJ)/*.o
	-rm $(F_OP2_LIB)/*.a
	-rm $(F_OP2_SRC)/*~
	-rm $(F_OP2_MOD)/*.mod
	-rm $(F_OP2)/*~
