# Compiler executables and flags
CC = xlc_r
CXX = xlc++_r

BASE_CPPFLAGS = -MMD -MP -Wall -pedantic

ifndef DEBUG
  BASE_CPPFLAGS += -O3
else
  BASE_CPPFLAGS += -g -O0
endif

CFLAGS ?= -std=c99 $(BASE_CPPFLAGS)
CXXFLAGS ?= $(BASE_CPPFLAGS)

CXXLINK ?= -lc++

# Available OpenMP features
OMP_CPPFLAGS ?= -qsmp=omp
CPP_HAS_OMP ?= true

OMP_OFFLOAD_CPPFLAGS ?=
CPP_HAS_OMP_OFFLOAD ?= false
