CC       = g++
OPTFLAGS = -Ofast -lm

BIN =  airfoil_orig

all: $(BIN)

airfoil_orig: airfoil_orig.cpp Makefile
	$(CC) $(CCFLAGS) $(OPTFLAGS) -o $@ airfoil_orig.cpp

clean:
	$(RM) $(BIN)
