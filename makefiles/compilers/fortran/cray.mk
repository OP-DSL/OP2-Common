# Compiler executables and flags
FC = ftn

MPIFC = ftn

BASE_FFLAGS =

ifndef DEBUG
  BASE_FFLAGS += -O3
else
  BASE_FFLAGS += -g -Og
endif

FFLAGS ?= $(BASE_FFLAGS)

F_MOD_OUT_OPT ?= -em -J

# Available OpenMP features
OMP_FFLAGS ?= -fopenmp
F_HAS_OMP ?= true

OMP_OFFLOAD_FFLAGS ?=
F_HAS_OMP_OFFLOAD ?= false

# Available CUDA features
CUDA_FFLAGS ?=
F_HAS_CUDA ?= false
