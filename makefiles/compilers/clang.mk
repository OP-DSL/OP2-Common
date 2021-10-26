ifdef DEBUG
  CCFLAGS  = -x c++ -O0 -I$(OMPTARGET_LIBS)/../include
else
  CCFLAGS  = -x c++ -O3 -I$(OMPTARGET_LIBS)/../include
endif
CXX	    = clang++
CXXFLAGS  = $(CCFLAGS)
MPICXX    = $(MPICPP_PATH)
MPIFLAGS  = $(CXXFLAGS)
NVCXXFLAGS = -ccbin=$(NVCXX_HOST_COMPILER)
OMP4FLAGS = -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda
