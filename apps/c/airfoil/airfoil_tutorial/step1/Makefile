CC       = g++
OPTFLAGS = -Ofast -lm

BIN =  airfoil_step1

all: $(BIN)

airfoil_step1: airfoil_step1.cpp Makefile
	$(CC) $(CCFLAGS) -I$(OP2_INSTALL_PATH)/include $(OPTFLAGS) -o $@ airfoil_step1.cpp -L$(OP2_INSTALL_PATH)/lib/ -lop2_seq


clean:
	$(RM) $(BIN)
