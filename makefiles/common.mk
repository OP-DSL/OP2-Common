SHELL = /bin/sh
.SUFFIXES:

# Get the makefiles directory (where this file is)
MAKEFILES_DIR := $(shell dirname $(realpath \
				 $(word $(words $(MAKEFILE_LIST)), $(MAKEFILE_LIST))))

ROOT_DIR := $(shell realpath $(MAKEFILES_DIR)/../)
OP2_BUILD_DIR ?= $(ROOT_DIR)/op2

OP2_INC ?= -I$(ROOT_DIR)/op2/include
OP2_LIB ?= -L$(OP2_BUILD_DIR)/lib

AR = ar rcs

ifdef CUDA_INSTALL_PATH
  include $(MAKEFILES_DIR)/nvcc.mk

  CUDA_INC ?= -I$(CUDA_INSTALL_PATH)/include
  CUDA_LIB ?= -lcudart
endif

ifdef MPI_INSTALL_PATH
  MPICC ?= $(MPI_INSTALL_PATH)/bin/mpicc
  MPICXX ?= $(MPI_INSTALL_PATH)/bin/mpic++
  MPIFC ?= $(MPI_INSTALL_PATH)/bin/mpifort

  MPI_INC ?= -I$(MPI_INSTALL_PATH)/include
  MPI_LIB ?= -L$(MPI_INSTALL_PATH)/lib -lmpi
endif

ifdef PARMETIS_INSTALL_PATH
  PARMETIS_INC ?= -I$(PARMETIS_INSTALL_PATH)/include -DHAVE_PARMETIS -DPARMETIS_VER_4
  PARMETIS_LIB ?= -L$(PARMETIS_INSTALL_PATH)/lib -lparmetis -lmetis
endif

ifdef PTSCOTCH_INSTALL_PATH
  PTSCOTCH_INC += -I$(PTSCOTCH_INSTALL_PATH)/include -DHAVE_PTSCOTCH
  PTSCOTCH_LIB += -L$(PTSCOTCH_INSTALL_PATH)/lib -lscotch -lptscotch -lptscotcherr
endif

ifdef HDF5_INSTALL_PATH
  HDF5_INC += -I$(HDF5_INSTALL_PATH)/include
  HDF5_LIB += -L$(HDF5_INSTALL_PATH)/lib -l:libhdf5.a -ldl -lm
endif

include $(MAKEFILES_DIR)/compilers/$(OP2_COMPILER).mk
