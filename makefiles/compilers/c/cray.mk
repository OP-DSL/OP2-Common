# Compiler executables and flags
CONFIG_CC := cc
CONFIG_CXX := CC

CONFIG_MPICC := cc
CONFIG_MPICXX := CC

BASE_CPPFLAGS := -MMD -MP -Wall -Wextra -pedantic

ifndef DEBUG
  BASE_CPPFLAGS += -O3
else
  BASE_CPPFLAGS += -g -Og
endif

CONFIG_CFLAGS ?= -std=c99 $(BASE_CPPFLAGS)
CONFIG_CXXFLAGS ?= $(BASE_CPPFLAGS)

CONFIG_CXXLINK ?= -lstdc++

# Available OpenMP features
CONFIG_OMP_CPPFLAGS ?= -fopenmp
CONFIG_CPP_HAS_OMP ?= true

# CONFIG_OMP_OFFLOAD_CPPFLAGS ?=
CONFIG_CPP_HAS_OMP_OFFLOAD ?= false