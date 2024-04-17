# Compiler executables and flags
CONFIG_CC := gcc
CONFIG_CXX := g++

BASE_CXXFLAGS := -MMD -MP -Wall -Wextra -pedantic

ifndef DEBUG
  BASE_CXXFLAGS += -O3

  ifeq ($(TARGET_HOST),true)
    BASE_CXXFLAGS += -march=native
  endif
else
  BASE_CXXFLAGS += -g -Og
endif

CONFIG_CFLAGS ?= -std=c99 $(BASE_CXXFLAGS) $(EXTRA_CFLAGS)
CONFIG_CXXFLAGS ?= $(BASE_CXXFLAGS) $(EXTRA_CXXFLAGS)

CONFIG_CXXLINK ?= -lstdc++

# Available OpenMP features
CONFIG_OMP_CXXFLAGS ?= -fopenmp
CONFIG_CPP_HAS_OMP ?= true

# CONFIG_OMP_OFFLOAD_CXXFLAGS ?=
CONFIG_CPP_HAS_OMP_OFFLOAD ?= false
