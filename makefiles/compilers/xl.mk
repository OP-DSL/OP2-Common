  ifdef DEBUG
    CCFLAGS = -O0 -g
  else
    CCFLAGS = -qarch=pwr8 -qtune=pwr8 -O3 -qhot
  endif
  CXX       = xlc++
  CXXFLAGS  = $(CCFLAGS)
  MPICXX    = $(MPICXX_PATH)
  MPIFLAGS  = $(CXXFLAGS)
  OMP4FLAGS = -qsmp=omp -qoffload
