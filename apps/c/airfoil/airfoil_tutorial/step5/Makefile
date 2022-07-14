CC       = g++ -g
OPTFLAGS = -Ofast -lm

BIN =  airfoil_step5

all: $(BIN)

airfoil_step5: airfoil_step5.cpp Makefile
	$(CC) $(CCFLAGS) -I$(OP2_INSTALL_PATH)/include $(OPTFLAGS) -o $@ airfoil_step5.cpp -L$(OP2_INSTALL_PATH)/lib/ -lop2_seq


clean:
	$(RM) $(BIN)
