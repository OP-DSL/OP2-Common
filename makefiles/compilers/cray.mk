# Compiler executables and flags
CC = cc
CXX = CC
FC = ftn

MPICC = cc
MPICXX = CC
MPIFC = ftn

BASE_CPPFLAGS = -MMD -MP -Wall -Wextra -pedantic
BASE_FFLAGS =

ifndef DEBUG
  BASE_CPPFLAGS += -O3
  BASE_FFLAGS += -O3
else
  BASE_CPPFLAGS += -g -Og
  BASE_FFLAGS += -g -Og
endif

CFLAGS ?= -std=c99 $(BASE_CPPFLAGS)
CXXFLAGS ?= $(BASE_CPPFLAGS)
FFLAGS ?= $(BASE_FFLAGS)

F_MOD_OUT_OPT ?= -em -J

# Available OpenMP features
OMP_CPPFLAGS ?=
OMP_FFLAGS ?=

# The compiler enables OpenMP by default for -O1 and up
CPP_HAS_OMP ?= true
F_HAS_OMP ?= true

OMP_OFFLOAD_CPPFLAGS ?=
OMP_OFFLOAD_FFLAGS ?=

CPP_HAS_OMP_OFFLOAD ?= false
F_HAS_OMP_OFFLOAD ?= false
