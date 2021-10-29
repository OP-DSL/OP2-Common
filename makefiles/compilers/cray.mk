CC ?= cc
CXX ?= CC
FC ?= ftn

BASE_CPPFLAGS =
BASE_FFLAGS =

ifndef DEBUG
  BASE_CPPFLAGS += -O3
  BASE_FFLAGS += -O3
else
  BASE_CPPFLAGS += -g -O0
  BASE_FFLAGS += -g -O0
endif

CFLAGS ?= -h std=c99 $(BASE_CPPFLAGS)
CXXFLAGS ?= $(BASE_CPPFLAGS)
FFLAGS ?= $(BASE_FFLAGS)

F_MOD_OUT_OPT ?= -em -J


OMP_CPPFLAGS ?=
OMP_FFLAGS ?=

# The compiler enable OpenMP by default for -O1 and up
CPP_HAS_OMP ?= true
F_HAS_OMP ?= true

OMP_OFFLOAD_CPPFLAGS ?=
OMP_OFFLOAD_FFLAGS ?=

CPP_HAS_OMP_OFFLOAD ?= false
F_HAS_OMP_OFFLOAD ?= false
