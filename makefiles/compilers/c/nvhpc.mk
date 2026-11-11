# Compiler executables and flags
CONFIG_CC := nvc
CONFIG_CXX := nvc++

ifndef DEBUG
  BASE_CXXFLAGS += -O3

  ifeq ($(TARGET_HOST),true)
    BASE_CXXFLAGS += -fast
  endif
else
  BASE_CXXFLAGS += -g -O0
endif

CONFIG_CFLAGS ?= -c99 $(BASE_CXXFLAGS) $(EXTRA_CFLAGS)
CONFIG_CXXFLAGS ?= $(BASE_CXXFLAGS) $(EXTRA_CXXFLAGS) -std=c++20

CONFIG_CXXLINK ?= -lstdc++

# Available OpenMP features
CONFIG_OMP_CXXFLAGS ?= -mp
CONFIG_CPP_HAS_OMP ?= true
