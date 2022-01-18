# Compiler executables and flags
FC := gfortran

BASE_FFLAGS := -Wall -pedantic -ffixed-line-length-none -ffree-line-length-none -fcray-pointer

ifndef DEBUG
  BASE_FFLAGS += -O3
else
  BASE_FFLAGS += -g -Og
endif

FFLAGS ?= $(BASE_FFLAGS)
F_MOD_OUT_OPT ?= -J
F_HAS_PARALLEL_BUILDS ?= true

# Available OpenMP features
OMP_FFLAGS ?= -fopenmp
F_HAS_OMP ?= true

OMP_OFFLOAD_FFLAGS ?=
F_HAS_OMP_OFFLOAD ?= false

# Available CUDA features
CUDA_FFLAGS ?=
F_HAS_CUDA ?= false
