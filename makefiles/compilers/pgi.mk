  ifdef DEBUG
    CCFLAGS = -g -O0
  else
    CCFLAGS = -O3
  endif
  CXX	    = pgc++
  CXXFLAGS  = $(CCFLAGS)
  MPICXX    = $(MPICPP_PATH)
  MPIFLAGS  = $(CXXFLAGS)
  # NVCXXFLAGS += -ccbin=$(MPICXX)
