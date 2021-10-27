CC ?= gcc
CXX ?= g++
FC ?= gfortran

BASE_CPPFLAGS = -MMD -MP -Wall -Wextra -pedantic
BASE_FFLAGS = -Wall -pedantic -ffixed-line-length-none -ffree-line-length-none

ifndef DEBUG
	BASE_CPPFLAGS += -O3
	BASE_FFLAGS += -O3
else
	BASE_CPPFLAGS += -g -Og
	BASE_FFLAGS += -g -Og
endif

CFLAGS ?= -std=c99 $(BASE_CPPFLAGS)
CXXFLAGS ?= $(BASE_CPPFLAGS)
FFLAGS ?= $(BASE_FFLAGS)

F_MOD_OUT_OPT ?= -J


OMP_CPPFLAGS ?= -fopenmp
OMP_FFLAGS ?= -fopenmp

CPP_HAS_OMP ?= true
F_HAS_OMP ?= true

OMP_OFFLOAD_CPPFLAGS ?=
OMP_OFFLOAD_FFLAGS ?=

CPP_HAS_OMP_OFFLOAD ?= false
F_HAS_OMP_OFFLOAD ?= false
