# Compiler executables and flags
CC ?= nvcc
CXX ?= nvc++

BASE_CPPFLAGS = -MD

ifndef DEBUG
  BASE_CPPFLAGS += -O3
else
  BASE_CPPFLAGS += -g -O0
endif

CFLAGS ?= -c99 $(BASE_CPPFLAGS)
CXXFLAGS ?= $(BASE_CPPFLAGS)

CXXLINK ?= -lstdc++

# Available OpenMP features
OMP_CPPFLAGS ?= -mp
CPP_HAS_OMP ?= true

OMP_OFFLOAD_CPPFLAGS ?=
CPP_HAS_OMP_OFFLOAD ?= false
