# Compiler executables and flags
CC = icc
CXX = icpc

BASE_CPPFLAGS = -MMD -MP -Wall

ifndef DEBUG
  BASE_CPPFLAGS += -O3
else
  BASE_CPPFLAGS += -g -O0
endif

CFLAGS ?= -std=c99 $(BASE_CPPFLAGS)
CXXFLAGS ?= $(BASE_CPPFLAGS)

CXXLINK ?= -lstdc++ -lirc -lsvml

# Available OpenMP features
OMP_CPPFLAGS ?= -qopenmp
CPP_HAS_OMP ?= true

OMP_OFFLOAD_CPPFLAGS ?=
CPP_HAS_OMP_OFFLOAD ?= false
