# Compiler executables and flags
CONFIG_CC := icc
CONFIG_CXX := icpc

BASE_CXXFLAGS := -MMD -MP -Wall  -D_GLIBCXX_USE_CXX11_ABI=1 -D__PURE_INTEL_C99_HEADERS__ -no-multibyte-chars -D__is_assignable=__is_trivially_assignable -std=c++17

ifndef DEBUG
  BASE_CXXFLAGS += -g -O3

  ifeq ($(TARGET_HOST),true)
    BASE_CXXFLAGS += -xhost
  endif
else
  BASE_CXXFLAGS += -g -O0
endif

CONFIG_CFLAGS ?= -std=c99 $(BASE_CXXFLAGS) $(EXTRA_CXXFLAGS)
CONFIG_CXXFLAGS ?= $(BASE_CXXFLAGS) $(EXTRA_CXXFLAGS)

CONFIG_CXXLINK ?= -lstdc++ -lirc -lsvml

# Available OpenMP features
CONFIG_OMP_CXXFLAGS ?= -qopenmp
CONFIG_CPP_HAS_OMP ?= true

# CONFIG_OMP_OFFLOAD_CXXFLAGS ?=
CONFIG_CPP_HAS_OMP_OFFLOAD ?= false
