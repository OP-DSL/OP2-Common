# Compiler executables and flags
CONFIG_CC := xlc_r
CONFIG_CXX := xlc++_r

BASE_CXXFLAGS := -MMD -MP -Wall -pedantic

ifndef DEBUG
  ifeq ($(TARGET_HOST),true)
    BASE_CXXFLAGS += -O4
  else
    BASE_CXXFLAGS += -O3
  endif
else
  BASE_CXXFLAGS += -g -O0
endif

CONFIG_CFLAGS ?= -std=c99 $(BASE_CXXFLAGS) $(EXTRA_CFLAGS)
CONFIG_CXXFLAGS ?= $(BASE_CXXFLAGS) $(EXTRA_CXXFLAGS)

CONFIG_CXXLINK ?= -lc++

# Available OpenMP features
CONFIG_OMP_CXXFLAGS ?= -qsmp=omp
CONFIG_CPP_HAS_OMP ?= true

# CONFIG_OMP_OFFLOAD_CXXFLAGS ?=
CONFIG_CPP_HAS_OMP_OFFLOAD ?= false
