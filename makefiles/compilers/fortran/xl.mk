# Compiler executables and flags
FC = xlf_r

BASE_FFLAGS =

ifndef DEBUG
  BASE_FFLAGS += -O3
else
  BASE_FFLAGS += -g -O0
endif

FFLAGS ?= $(BASE_FFLAGS)

# Available OpenMP features
OMP_FFLAGS ?= -qsmp=omp
F_HAS_OMP ?= true

OMP_OFFLOAD_FFLAGS ?=
F_HAS_OMP_OFFLOAD ?= false

# Available CUDA features
CUDA_FFLAGS ?=
F_HAS_CUDA ?= false
