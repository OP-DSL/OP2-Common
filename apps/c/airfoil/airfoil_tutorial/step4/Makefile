CC       = g++ -g
OPTFLAGS = -Ofast -lm

BIN =  airfoil_step4

all: $(BIN)

airfoil_step4: airfoil_step4.cpp Makefile
	$(CC) $(CCFLAGS) -I$(OP2_INSTALL_PATH)/include $(OPTFLAGS) -o $@ airfoil_step4.cpp -L$(OP2_INSTALL_PATH)/lib/ -lop2_seq


clean:
	$(RM) $(BIN)
