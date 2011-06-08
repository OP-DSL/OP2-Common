



OP2 = $(OP2_INST_DIR)
SRC = $(OP2)/src
INC = $(OP2)/include
LIB = $(OP2)/lib
OBJ = $(OP2)/obj
CUDA_INC = $(CUDA)/include

CC = gcc
CPP = g++
NVCC = nvcc

FLAGS = -Wall

.PHONY: clean

all: core core_nvcc reference reference_nvcc runtime runtime_nvcc cuda cuda_nvcc

core: $(INC)/op_lib_core.h $(SRC)/op_lib_core.cpp
	$(CPP) $(FLAGS) -I$(INC) -c $(SRC)/op_lib_core.cpp -o $(OBJ)/op_lib_core.o
	ar -r $(LIB)/libop2_core.a $(OBJ)/op_lib_core.o

core_nvcc: $(INC)/op_lib_core.h $(SRC)/op_lib_core.cpp
	$(NVCC) -I$(INC) -c $(SRC)/op_lib_core.cpp -o $(OBJ)/op_lib_core_nvcc.o
	ar -r $(LIB)/libop2_core_nvcc.a $(OBJ)/op_lib_core_nvcc.o

reference: $(INC)/op_reference.h $(SRC)/op_reference.cpp $(INC)/op_lib_core.h $(SRC)/op_lib_core.cpp
	$(CPP) $(FLAGS) -I$(INC) -c $(SRC)/op_reference.cpp -o $(OBJ)/op_reference.o
	ar -r $(LIB)/libop2_reference.a $(OBJ)/op_reference.o
	
reference_nvcc: $(INC)/op_reference.h $(SRC)/op_reference.cpp $(INC)/op_lib_core.h $(SRC)/op_lib_core.cpp
	$(NVCC) --compiler-options $(FLAGS) -I$(INC) -c $(SRC)/op_reference.cpp -o $(OBJ)/op_reference.o
	ar -r $(LIB)/libop2_reference_nvcc.a $(OBJ)/op_reference.o

runtime: $(INC)/op_rt_support.h $(SRC)/op_rt_support.cpp $(INC)/op_lib_core.h $(SRC)/op_lib_core.cpp
	$(CPP) $(FLAGS) -I$(INC) -c $(SRC)/op_rt_support.cpp -o $(OBJ)/op_rt_support.o
	ar -r $(LIB)/libop2_rt_support.a $(OBJ)/op_rt_support.o

runtime_nvcc: $(INC)/op_rt_support.h $(SRC)/op_rt_support.cpp $(INC)/op_lib_core.h $(SRC)/op_lib_core.cpp
	$(NVCC) --compiler-options $(FLAGS) -I$(INC) -c $(SRC)/op_rt_support.cpp -o $(OBJ)/op_rt_support.o
	ar -r $(LIB)/libop2_rt_support_nvcc.a $(OBJ)/op_rt_support.o

cuda: $(INC)/op_cuda_rt_support.h $(SRC)/cuda/op_cuda_decl.cpp $(SRC)/cuda/op_cuda_rt_support.cpp $(OBJ)/op_lib_core.o
	$(CPP) -I$(CUDA_INSTALL_PATH)/include  -I$(INC) -c $(SRC)/cuda/op_cuda_decl.cpp -o $(OBJ)/cuda/op_cuda_decl.o
	$(CPP) -I$(CUDA_INSTALL_PATH)/include -I$(INC) -c $(SRC)/cuda/op_cuda_rt_support.cpp -o $(OBJ)/cuda/op_cuda_rt_support.o
	ar -r $(LIB)/libop2_cuda.a $(OBJ)/cuda/op_cuda_rt_support.o $(OBJ)/cuda/op_cuda_decl.o $(OBJ)/op_lib_core.o

cuda_nvcc: $(INC)/op_cuda_rt_support.h $(SRC)/cuda/op_cuda_decl.cu $(SRC)/cuda/op_cuda_rt_support.cu $(OBJ)/op_lib_core_nvcc.o
	$(NVCC) -I$(INC) -c $(SRC)/cuda/op_cuda_decl.cu -o $(OBJ)/cuda/op_cuda_decl_nvcc.o
	$(NVCC) -I$(INC) -c $(SRC)/cuda/op_cuda_rt_support.cu -o $(OBJ)/cuda/op_cuda_rt_support.o
	ar -r $(LIB)/libop2_cuda_nvcc.a $(OBJ)/cuda/op_cuda_rt_support.o $(OBJ)/cuda/op_cuda_decl_nvcc.o $(OBJ)/op_lib_core_nvcc.o

clean:
	\rm -if $(OBJ)/*.o
	\rm -if $(OBJ)/cuda/*.o
	\rm -if $(OBJ)/openmp/*.o
	\rm -if $(LIB)/*.a
	\rm -if ./*~