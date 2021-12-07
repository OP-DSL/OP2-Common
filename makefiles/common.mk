SHELL = /bin/sh
.SUFFIXES:

# Helper function to upper-case a string
define UPPERCASE =
$(shell echo "$(1)" | tr "[:lower:]" "[:upper:]")
endef

# Get the makefiles directory (where this file is)
MAKEFILES_DIR != dirname $(realpath \
	$(word $(words $(MAKEFILE_LIST)), $(MAKEFILE_LIST)))

ROOT_DIR != realpath $(MAKEFILES_DIR)/../

# Include profile #! PRE section
ifdef OP2_PROFILE
  OP2_PROFILE_FILE = $(MAKEFILES_DIR)/profiles/$(OP2_PROFILE).mk

  $(shell sed -n '/^#!\s*PRE/,$${p;/^#!\s*POST/q}' $(OP2_PROFILE_FILE) > \
      $(MAKEFILES_DIR)/.profile.pre.mk)

  $(shell sed -n '/^#!\s*POST/,$${p;/^#!\s*PRE/q}' $(OP2_PROFILE_FILE) > \
      $(MAKEFILES_DIR)/.profile.post.mk)

  include $(MAKEFILES_DIR)/.profile.pre.mk
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

ifdef OP2_COMPILER
  OP2_C_COMPILER ?= $(OP2_COMPILER)
  OP2_F_COMPILER ?= $(OP2_COMPILER)
  OP2_C_CUDA_COMPILER ?= nvhpc
endif

ifdef OP2_C_COMPILER
  include $(MAKEFILES_DIR)/compilers/c/$(OP2_C_COMPILER).mk
else
  $(warning OP2_C_COMPILER undefined: define or use OP2_COMPILER or OP2_PROFILE)
endif

ifdef OP2_F_COMPILER
  include $(MAKEFILES_DIR)/compilers/fortran/$(OP2_F_COMPILER).mk
else
  $(warning OP2_F_COMPILER undefined: define or use OP2_COMPILER or OP2_PROFILE)
endif

ifdef OP2_C_CUDA_COMPILER
  include $(MAKEFILES_DIR)/compilers/c_cuda/$(OP2_C_CUDA_COMPILER).mk
else
  $(warning OP2_C_CUDA_COMPILER undefined: define or use OP2_COMPILER or OP2_PROFILE)
endif

ifeq ($(F_HAS_CUDA),true)
  CUDA_FFLAGS += -DOP2_WITH_CUDAFOR
endif

ifdef MPI_INSTALL_PATH
  MPI_BIN ?= $(MPI_INSTALL_PATH)/bin/
endif

MPICC ?= $(MPI_BIN)mpicc
MPICXX ?= $(MPI_BIN)mpicxx
MPIFC ?= $(MPI_BIN)mpif90

# Anti MPI C++ binding measures
CFLAGS += -DOMPI_SKIP_MPICXX -DMPICH_IGNORE_CXX_SEEK -DMPIPP_H
CXXFLAGS += -DOMPI_SKIP_MPICXX -DMPICH_IGNORE_CXX_SEEK -DMPIPP_H

# Dependencies
include $(MAKEFILES_DIR)/dependencies.mk

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

# Include profile #! POST section
ifdef OP2_PROFILE_FILE
  include $(MAKEFILES_DIR)/.profile.post.mk
endif
