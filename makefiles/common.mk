SHELL = /bin/sh
.SUFFIXES:

# Helper function to upper-case a string
define UPPERCASE =
$(shell echo "$(1)" | tr "[:lower:]" "[:upper:]")
endef

# Get the makefiles directory (where this file is)
MAKEFILES_DIR := $(shell dirname $(realpath \
				 $(word $(words $(MAKEFILE_LIST)), $(MAKEFILE_LIST))))

ROOT_DIR := $(shell realpath $(MAKEFILES_DIR)/../)

# Include profile #! PRE section
ifdef OP2_PROFILE
  OP2_PROFILE_FILE = $(MAKEFILES_DIR)/profiles/$(OP2_PROFILE).mk

  $(shell awk '/#!\s+PRE/, /(#!\s+POST|END)/' $(OP2_PROFILE_FILE) > \
      $(MAKEFILES_DIR)/.profile.pre.mk)

  $(shell awk '/#!\s+POST/, /(#!\s+PRE|END)/' $(OP2_PROFILE_FILE) > \
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

ifdef CUDA_INSTALL_PATH
  CUDA_INC ?= -I$(CUDA_INSTALL_PATH)/include
  CUDA_LIB ?= -L$(CUDA_INSTALL_PATH)/lib64 \
	      -L$(CUDA_INSTALL_PATH)/lib \
	      -lcudart
endif

ifdef MPI_INSTALL_PATH
  MPI_BIN ?= $(MPI_INSTALL_PATH)/bin/
endif

MPICC ?= $(MPI_BIN)mpicc
MPICXX ?= $(MPI_BIN)mpicxx
MPIFC ?= $(MPI_BIN)mpif90

CFLAGS += -DOMPI_SKIP_MPICXX -DMPICH_IGNORE_CXX_SEEK
CXXFLAGS += -DOMPI_SKIP_MPICXX -DMPICH_IGNORE_CXX_SEEK

PARMETIS_INC ?= -DHAVE_PARMETIS -DPARMETIS_VER_4
ifdef PARMETIS_INSTALL_PATH
  PARMETIS_INC := -I$(PARMETIS_INSTALL_PATH)/include $(PARMETIS_INC)
  PARMETIS_LIB ?= -L$(PARMETIS_INSTALL_PATH)/lib -lparmetis -lmetis
endif

PTSCOTCH_INC ?= -DHAVE_PTSCOTCH
ifdef PTSCOTCH_INSTALL_PATH
  PTSCOTCH_INC := -I$(PTSCOTCH_INSTALL_PATH)/include $(PTSCOTCH_INC)
  PTSCOTCH_LIB ?= -L$(PTSCOTCH_INSTALL_PATH)/lib -lptscotch -lscotch -lptscotcherr
endif

HDF5_INC ?=
ifdef HDF5_INSTALL_PATH
  HDF5_INC := -I$(HDF5_INSTALL_PATH)/include $(HDF5_INC)
  HDF5_LIB ?= -L$(HDF5_INSTALL_PATH)/lib -l:libhdf5.a -ldl -lm -lz
endif

# Generate helper variables OP2_LIB_SEQ, OP2_LIB_MPI_CUDA, ...
define OP2_LIB_template =
OP2_LIB_$(call UPPERCASE,$(1)) := $(OP2_LIB) -lop2_$(1) $(2)
OP2_LIB_FOR_$(call UPPERCASE,$(1)) := $(OP2_LIB) -lop2_for_$(1) $(3)
endef

OP2_LIB_EXTRA +=
OP2_LIB_EXTRA_MPI += $(PARMETIS_LIB) $(PTSCOTCH_LIB)

OP2_LIB_FOR_EXTRA += $(OP2_LIB_EXTRA)
OP2_LIB_FOR_EXTRA_MPI += $(OP2_LIB_EXTRA_MPI)

ifeq ($(OP2_LIBS_WITH_HDF5),true)
  OP2_LIB_EXTRA += -lop2_hdf5 $(HDF5_LIB)
  OP2_LIB_EXTRA_MPI += -lop2_hdf5 $(HDF5_LIB)

  OP2_LIB_FOR_EXTRA += -lop2_for_hdf5 $(HDF5_LIB)
  OP2_LIB_FOR_EXTRA_MPI += -lop2_for_hdf5 $(HDF5_LIB)
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
