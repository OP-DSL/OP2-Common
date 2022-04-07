# Compiler executables and flags
CONFIG_CC := xlc_r
CONFIG_CXX := xlc++_r

BASE_CPPFLAGS := -MMD -MP -Wall -pedantic

ifndef DEBUG
  BASE_CPPFLAGS += -O3
else
  BASE_CPPFLAGS += -g -O0
endif

CONFIG_CFLAGS ?= -std=c99 $(BASE_CPPFLAGS)
CONFIG_CXXFLAGS ?= $(BASE_CPPFLAGS)

CONFIG_CXXLINK ?= -lc++

# Available OpenMP features
CONFIG_OMP_CPPFLAGS ?= -qsmp=omp
CONFIG_CPP_HAS_OMP ?= true

# CONFIG_OMP_OFFLOAD_CPPFLAGS ?=
CONFIG_CPP_HAS_OMP_OFFLOAD ?= false
