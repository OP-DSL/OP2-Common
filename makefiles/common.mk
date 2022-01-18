SHELL = /bin/sh
.SUFFIXES:

.DEFAULT_GOAL := all

# Helper function to upper-case a string
define UPPERCASE =
$(shell echo "$(1)" | tr "[:lower:]" "[:upper:]")
endef

# Helper variables for comma and space substitution
COMMA := ,
SPACE :=
SPACE +=

# Get the makefiles directory (where this file is)
MAKEFILES_DIR != dirname $(realpath \
	$(word $(words $(MAKEFILE_LIST)), $(MAKEFILE_LIST)))

ROOT_DIR != realpath $(MAKEFILES_DIR)/../

# Include profile
ifdef OP2_PROFILE
  include $(MAKEFILES_DIR)/profiles/$(OP2_PROFILE).mk
endif

OP2_BUILD_DIR ?= $(ROOT_DIR)/op2

OP2_INC ?= -I$(ROOT_DIR)/op2/include
OP2_LIB ?= -L$(OP2_BUILD_DIR)/lib
OP2_MOD ?= -I$(OP2_BUILD_DIR)/mod
OP2_MOD_CUDA ?= $(OP2_MOD)/cuda

OP2_LIBS_SINGLE_NODE := seq cuda openmp openmp4
OP2_FOR_LIBS_SINGLE_NODE := $(foreach lib,$(OP2_LIBS_SINGLE_NODE),f_$(lib))

OP2_LIBS_MPI := mpi mpi_cuda
OP2_FOR_LIBS_MPI := $(foreach lib,$(OP2_LIBS_MPI),f_$(lib))

OP2_LIBS := hdf5 $(OP2_LIBS_SINGLE_NODE) $(OP2_LIBS_MPI)
OP2_FOR_LIBS := f_hdf5 $(OP2_FOR_LIBS_SINGLE_NODE) $(OP2_FOR_LIBS_MPI)

AR := ar rcs

# Dependencies
DEPS_DIR := $(MAKEFILES_DIR)/dependencies

ifneq ($(MAKECMDGOALS),clean)
  # Compiler definitions
  include $(MAKEFILES_DIR)/compilers.mk

  ifeq ($(HAVE_C),true)
    include $(DEPS_DIR)/hdf5_seq.mk
  endif

  ifeq ($(HAVE_C_CUDA),true)
    include $(DEPS_DIR)/cuda.mk
  endif

  ifeq ($(HAVE_MPI_C),true)
    include $(DEPS_DIR)/hdf5_par.mk

    include $(DEPS_DIR)/ptscotch.mk
    include $(DEPS_DIR)/parmetis.mk
  endif
endif

.PHONY: detect
detect:
	@echo > /dev/null

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
  $(info )
  $(info Compilation flags:)
  $(info .   C: $(CFLAGS))
  $(info .   C++: $(CXXFLAGS))
  $(info .   CUDA: $(NVCCFLAGS))
  $(info .   Fortran: $(FFLAGS))
  $(info )
endif

# Generate helper variables OP2_LIB_SEQ, OP2_LIB_MPI_CUDA, ...
define OP2_LIB_template =
OP2_LIB_$(call UPPERCASE,$(1)) := $(OP2_LIB) -lop2_$(1) $(2)
OP2_LIB_FOR_$(call UPPERCASE,$(1)) := $(OP2_LIB) -lop2_for_$(1) $(3)
endef

OP2_LIB_EXTRA_MPI += $(PARMETIS_LIB) $(PTSCOTCH_LIB)
OP2_LIB_FOR_EXTRA_MPI += $(PARMETIS_LIB) $(PTSCOTCH_LIB)

ifeq ($(OP2_LIBS_WITH_HDF5),true)
  OP2_LIB_EXTRA += -lop2_hdf5 $(HDF5_SEQ_LIB)
  OP2_LIB_EXTRA_MPI += $(HDF5_PAR_LIB)

  OP2_LIB_FOR_EXTRA += -lop2_for_hdf5 $(HDF5_SEQ_LIB)
  OP2_LIB_FOR_EXTRA_MPI += $(HDF5_PAR_LIB)
endif

$(foreach lib,$(OP2_LIBS_SINGLE_NODE),$(eval $(call OP2_LIB_template,$(lib),\
	$(OP2_LIB_EXTRA),$(OP2_LIB_FOR_EXTRA))))

$(foreach lib,$(OP2_LIBS_MPI),$(eval $(call OP2_LIB_template,$(lib),\
	$(OP2_LIB_EXTRA_MPI),$(OP2_LIB_FOR_EXTRA_MPI))))

OP2_LIB_CUDA += $(CUDA_LIB)
OP2_LIB_MPI_CUDA += $(CUDA_LIB)
