# Compiler executables and flags
CONFIG_CC := gcc
CONFIG_CXX := g++

BASE_CXXFLAGS := -MMD -MP -Wall -Wextra -pedantic #-fsanitize=address -fsanitize=undefined

ifndef DEBUG
  BASE_CXXFLAGS += -g -O3

  ifeq ($(TARGET_HOST),true)
    BASE_CXXFLAGS += -march=native
  endif
else
  BASE_CXXFLAGS += -g -Og #-fsanitize=undefined -fsanitize=address
endif

CONFIG_CFLAGS ?= -std=c99 $(BASE_CXXFLAGS) $(EXTRA_CFLAGS)
CONFIG_CXXFLAGS ?= $(BASE_CXXFLAGS) $(EXTRA_CXXFLAGS) -std=c++17

CONFIG_CXXLINK ?= -lstdc++

# Available OpenMP features
CONFIG_OMP_CXXFLAGS ?= -fopenmp
CONFIG_CPP_HAS_OMP ?= true

# Use Homebrew libomp on macOS if the compiler has no OpenMP of its own
OMP_TEST = echo 'int main(){}' | $(CONFIG_CXX) $(1) -x c++ - -o /dev/null 2> /dev/null && echo ok

ifeq ($(shell uname -s),Darwin)
  ifeq ($(CONFIG_CPP_HAS_OMP),true)
    ifneq ($(shell $(call OMP_TEST,$(CONFIG_OMP_CXXFLAGS))),ok)
      LIBOMP_PREFIX := $(shell brew --prefix --installed libomp 2> /dev/null)
      LIBOMP_CXXFLAGS := -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include -L$(LIBOMP_PREFIX)/lib -lomp

      ifneq ($(and $(LIBOMP_PREFIX),$(shell $(call OMP_TEST,$(LIBOMP_CXXFLAGS)))),)
        CONFIG_OMP_CXXFLAGS := $(LIBOMP_CXXFLAGS)
      else
        CONFIG_CPP_HAS_OMP := false
      endif
    endif
  endif
endif

# CONFIG_OMP_OFFLOAD_CXXFLAGS ?=
CONFIG_CPP_HAS_OMP_OFFLOAD ?= false
