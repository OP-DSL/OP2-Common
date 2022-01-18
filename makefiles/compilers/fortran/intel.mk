# Compiler executables and flags
FC := ifort

BASE_FFLAGS := -warn all

ifndef DEBUG
  BASE_FFLAGS += -O3
else
  BASE_FFLAGS += -g -O0
endif

FFLAGS ?= $(BASE_FFLAGS)
F_MOD_OUT_OPT ?= -module #
F_HAS_PARALLEL_BUILDS ?= true

# Available OpenMP features
OMP_FFLAGS ?= -qopenmp
F_HAS_OMP ?= true

OMP_OFFLOAD_FFLAGS ?=
F_HAS_OMP_OFFLOAD ?= false

# Available CUDA features
CUDA_FFLAGS ?=
F_HAS_CUDA ?= false

