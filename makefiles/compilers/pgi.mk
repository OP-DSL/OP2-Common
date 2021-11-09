# TODO: PGI_CUDA_FORT_FLAGS? F_HAS_CUDA? -pgc++libs?

# Compiler executables and flags
CC ?= pgcc
CXX ?= pgc++
FC ?= pgfortran

BASE_CPPFLAGS = -MD
BASE_FFLAGS =

ifndef DEBUG
  BASE_CPPFLAGS += -O3
  BASE_FFLAGS += -O3
else
  BASE_CPPFLAGS += -g -O0
  BASE_FFLAGS += -g -O0
endif

CFLAGS ?= -c99 $(BASE_CPPFLAGS)
CXXFLAGS ?= $(BASE_CPPFLAGS)
FFLAGS ?= $(BASE_FFLAGS)

CXXFLAGS ?= -lc++

F_MOD_OUT_OPT ?= -module #

# Available OpenMP features
OMP_CPPFLAGS ?= -mp
OMP_FFLAGS ?= -mp

CPP_HAS_OMP ?= true
F_HAS_OMP ?= true

OMP_OFFLOAD_CPPFLAGS ?=
OMP_OFFLOAD_FFLAGS ?=

CPP_HAS_OMP_OFFLOAD ?= false
F_HAS_OMP_OFFLOAD ?= false
