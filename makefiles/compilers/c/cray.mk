# Compiler executables and flags
CC := cc
CXX := CC

MPICC := cc
MPICXX := CC

BASE_CPPFLAGS := -MMD -MP -Wall -Wextra -pedantic

ifndef DEBUG
  BASE_CPPFLAGS += -O3
else
  BASE_CPPFLAGS += -g -Og
endif

CFLAGS ?= -std=c99 $(BASE_CPPFLAGS)
CXXFLAGS ?= $(BASE_CPPFLAGS)

CXXLINK ?= -lstdc++

# Available OpenMP features
OMP_CPPFLAGS ?= -fopenmp
CPP_HAS_OMP ?= true

OMP_OFFLOAD_CPPFLAGS ?=
CPP_HAS_OMP_OFFLOAD ?= false
