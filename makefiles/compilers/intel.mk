  ifdef DEBUG
    CCFLAGS = -O0 -g -pg
  else
    CCFLAGS = -O3 -xHost
  endif
  CXX       = icpc
  CXXFLAGS  = $(CCFLAGS)
  MPICXX    = $(MPICXX_PATH)
  MPIFLAGS  = $(CXXFLAGS)
