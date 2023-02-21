# Compiler executables and flags
CONFIG_CC := icc
CONFIG_CXX := icpc

BASE_CPPFLAGS := -MMD -MP -Wall

ifndef DEBUG
  BASE_CPPFLAGS += -O3
else
  BASE_CPPFLAGS += -g -O0
endif

CONFIG_CFLAGS ?= -std=c99 $(BASE_CPPFLAGS)
CONFIG_CXXFLAGS ?= $(BASE_CPPFLAGS)

CONFIG_CXXLINK ?= -lstdc++ -lirc -lsvml

# Available OpenMP features
CONFIG_OMP_CPPFLAGS ?= -qopenmp
CONFIG_CPP_HAS_OMP ?= true

# CONFIG_OMP_OFFLOAD_CPPFLAGS ?=
CONFIG_CPP_HAS_OMP_OFFLOAD ?= false
