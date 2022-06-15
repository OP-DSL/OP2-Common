SHELL = /bin/sh
.SUFFIXES:

.DEFAULT_GOAL := all

# Helper function to upper-case a string
define UPPERCASE =
$(shell echo "$(1)" | tr "[:lower:]" "[:upper:]")
endef

# Helper variables for comma and space substitution
COMMA := ,
SPACE := $() $()

ESC_RED != echo "\033[31m"
ESC_GREEN != echo "\033[32m"
ESC_DEFCOL != echo "\033[39m"

ESC_BOLD != echo "\033[1m"
ESC_RESET != echo "\033[0m"

TEXT_FOUND := $(ESC_GREEN)FOUND$(ESC_DEFCOL)
TEXT_NOTFOUND := $(ESC_RED)NOT FOUND$(ESC_DEFCOL)

# Get the makefiles directory (where this file is)
MAKEFILES_DIR != dirname $(realpath \
  $(word $(words $(MAKEFILE_LIST)), $(MAKEFILE_LIST)))

ROOT_DIR != realpath $(MAKEFILES_DIR)/../

define info_bold =
$(info $(ESC_BOLD)$(1)$(ESC_RESET))
endef

ifeq ($(MAKECMDGOALS),config)
  # Include profile
  ifdef OP2_PROFILE
    include $(MAKEFILES_DIR)/profiles/$(OP2_PROFILE).mk
  endif

  CONFIG_OP2_BUILD_DIR ?= $(ROOT_DIR)/op2

  CONFIG_OP2_INC ?= -I$(ROOT_DIR)/op2/include
  CONFIG_OP2_LIB ?= -L$(CONFIG_OP2_BUILD_DIR)/lib
  CONFIG_OP2_MOD ?= -I$(CONFIG_OP2_BUILD_DIR)/mod
  CONFIG_OP2_MOD_CUDA ?= $(CONFIG_OP2_MOD)/cuda

  CONFIG_AR := ar rcs

  # Dependencies
  DEPS_DIR := $(MAKEFILES_DIR)/dependencies
  DEP_BUILD_LOG := $(DEPS_DIR)/tests/.build.log

  ifndef MAKE_DETECT_DEBUG
    DEP_DETECT_EXTRA += 2> /dev/null
  endif

  # Compiler definitions
  include $(MAKEFILES_DIR)/compilers.mk

  $(info Looking for compilers and dependencies:)
  $(info )

  ifeq ($(CONFIG_HAVE_C),true)
    $(call info_bold,> C/C++ compilers $(TEXT_FOUND) ($(CONFIG_CXX)); looking for HDF5 (seq))
    include $(DEPS_DIR)/hdf5_seq.mk
  else
    $(call info_bold,> C/C++ compilers $(TEXT_NOTFOUND); skipping search for HDF5 (seq))
  endif

  $(info )

  ifeq ($(CONFIG_HAVE_C_CUDA),true)
    $(call info_bold,> C/C++ CUDA compiler $(TEXT_FOUND) ($(CONFIG_NVCC)); looking for the CUDA libraries)
    include $(DEPS_DIR)/cuda.mk
  else
    $(call info_bold,> C/C++ CUDA compiler $(TEXT_NOTFOUND); skipping search for CUDA libraries)
  endif

  $(info )

  ifeq ($(CONFIG_HAVE_MPI_C),true)
    $(call info_bold,> MPI C/C++ compilers $(TEXT_FOUND) ($(CONFIG_MPICXX)); \
      looking for HDF5 (parallel)$(COMMA) PT-Scotch and ParMETIS)

    include $(DEPS_DIR)/hdf5_par.mk

    include $(DEPS_DIR)/ptscotch.mk
    include $(DEPS_DIR)/parmetis.mk
    include $(DEPS_DIR)/kahip.mk
  else
    $(call info_bold,> MPI C/C++ compilers $(TEXT_NOTFOUND); \
      skipping search for HDF5 (parallel)$(COMMA) PT-Scotch and ParMETIS)
  endif

  $(info )
  $(shell rm -f $(DEP_BUILD_LOG))

  CONFIG_VARS := $(sort $(filter CONFIG_%,$(.VARIABLES)))

  $(file > $(MAKEFILES_DIR)/.config.mk,# Generated at $(shell date -R))
  $(file >> $(MAKEFILES_DIR)/.config.mk,)

  $(foreach var,$(CONFIG_VARS),$(file >> $(MAKEFILES_DIR)/.config.mk,\
    $(patsubst CONFIG_%,%,$(var)) := $($(var))#))

  $(info Config written to $(MAKEFILES_DIR)/.config.mk)
endif

.PHONY: config print_config

config:
	@echo > /dev/null

print_config:
	@echo > /dev/null

clean_config:
	rm -f $(MAKEFILES_DIR)/.config.mk

ifeq ($(wildcard $(MAKEFILES_DIR)/.config.mk),)
  $(error $(MAKEFILES_DIR)/.config.mk not found, run "make config" first)
else
  $(info Reading config from $(MAKEFILES_DIR)/.config.mk)
endif

$(info )

include $(MAKEFILES_DIR)/.config.mk

ifneq ($(MAKECMDGOALS),clean)
  # Evaluates to X_LIB if HAVE_X and X_LIB is defined
  # otherwise evaluates to "implicit" if HAVE_X is defined but not X_LIB
  # otherwise evaluates to "not found"
  I_STR = $(if $(HAVE_$(1)),$(if $($(1)_LIB),$($(1)_LIB),implicit),not found)

  $(info Compilers:)
  $(info .   C: $(if $(HAVE_C),$(CC),not found))
  $(info .   C++: $(if $(HAVE_C),$(CXX),not found))
  $(info .   CUDA: $(if $(HAVE_C_CUDA),$(NVCC),not found))
  $(info .   Fortran: $(if $(HAVE_F),$(FC),not found))
  $(info )
  $(info MPI compilers:)
  $(info .   C: $(if $(HAVE_MPI_C),$(MPICC),not found))
  $(info .   C++: $(if $(HAVE_MPI_C),$(MPICXX),not found))
  $(info .   Fortran: $(if $(HAVE_MPI_F),$(MPIFC),not found))
  $(info )
  $(info CUDA libraries: $(call I_STR,CUDA))
  $(info )
  $(info HDF5 I/O:)
  $(info .   Sequential: $(call I_STR,HDF5_SEQ))
  $(info .   Parallel: $(call I_STR,HDF5_PAR))
  $(info )
  $(info MPI partitioners:)
  $(info .   PT-Scotch: $(call I_STR,PTSCOTCH))
  $(info .   ParMETIS: $(call I_STR,PARMETIS))
  $(info .   KaHIP: $(call I_STR,KAHIP))
  $(info )
  $(info Compilation flags:)
  $(info .   C: $(CFLAGS))
  $(info .   C++: $(CXXFLAGS))
  $(info .   CUDA: $(NVCCFLAGS))
  $(info .   Fortran: $(FFLAGS))
  $(info )
endif

OP2_LIBS_SINGLE_NODE := seq cuda openmp openmp4
OP2_FOR_LIBS_SINGLE_NODE := $(foreach lib,$(OP2_LIBS_SINGLE_NODE),f_$(lib))

OP2_LIBS_MPI := mpi mpi_cuda
OP2_FOR_LIBS_MPI := $(foreach lib,$(OP2_LIBS_MPI),f_$(lib))

OP2_LIBS := hdf5 $(OP2_LIBS_SINGLE_NODE) $(OP2_LIBS_MPI)
OP2_FOR_LIBS := f_hdf5 $(OP2_FOR_LIBS_SINGLE_NODE) $(OP2_FOR_LIBS_MPI)
