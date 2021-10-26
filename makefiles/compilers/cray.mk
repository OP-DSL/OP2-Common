  CCFLAGS  = -O3
  CXX      = CC
  CXXFLAGS = $(CCFLAGS)
  MPICXX   = CC
  MPIFLAGS = $(CXXFLAGS) #-fsanitize=signed-integer-overflow,unsigned-integer-overflow
