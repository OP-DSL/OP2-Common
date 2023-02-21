MPICC       = mpicc -g
OPTFLAGS = -Ofast -lm

BIN =  airfoil_step6

all: $(BIN)

airfoil_step6: airfoil_step6.cpp Makefile
	$(MPICC) $(CCFLAGS) -I$(OP2_INSTALL_PATH)/include $(OPTFLAGS) -o $@ airfoil_step6.cpp -L$(OP2_INSTALL_PATH)/lib/ -lop2_mpi -L$(HDF5_INSTALL_PATH)/lib -lhdf5


clean:
	$(RM) $(BIN)
