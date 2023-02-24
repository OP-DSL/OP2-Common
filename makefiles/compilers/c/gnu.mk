# Compiler executables and flags
CONFIG_CC := gcc
CONFIG_CXX := g++

BASE_CPPFLAGS := -MMD -MP -Wall -Wextra -pedantic

ifndef DEBUG
  BASE_CPPFLAGS += -O3
else
  BASE_CPPFLAGS += -g -Og
endif

CONFIG_CFLAGS ?= -std=c99 $(BASE_CPPFLAGS)
CONFIG_CXXFLAGS ?= -std=gnu++17 $(BASE_CPPFLAGS)

CONFIG_CXXLINK ?= -lstdc++

# Available OpenMP features
CONFIG_OMP_CPPFLAGS ?= -fopenmp
CONFIG_CPP_HAS_OMP ?= true

# CONFIG_OMP_OFFLOAD_CPPFLAGS ?=
CONFIG_CPP_HAS_OMP_OFFLOAD ?= false
