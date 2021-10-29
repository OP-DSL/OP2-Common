CC ?= icc
CXX ?= icpc
FC ?= ifort

BASE_CPPFLAGS = -MMD -MP -Wall
BASE_FFLAGS = -warn all

ifndef DEBUG
  BASE_CPPFLAGS += -O3
  BASE_FFLAGS += -O3
else
  BASE_CPPFLAGS += -g -O0
  BASE_FFLAGS += -g -O0
endif

CFLAGS ?= -std=c99 $(BASE_CPPFLAGS)
CXXFLAGS ?= $(BASE_CPPFLAGS)
FFLAGS ?= $(BASE_FFLAGS)

F_MOD_OUT_OPT ?= -module #


OMP_CPPFLAGS ?= -qopenmp
OMP_FFLAGS ?= -qopenmp

CPP_HAS_OMP ?= true
F_HAS_OMP ?= true

OMP_OFFLOAD_CPPFLAGS ?=
OMP_OFFLOAD_FFLAGS ?=

CPP_HAS_OMP_OFFLOAD ?= false
F_HAS_OMP_OFFLOAD ?= false
