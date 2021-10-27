SHELL = /bin/sh
.SUFFIXES:

# Get the makefiles directory (where this file is)
MAKEFILES_DIR := $(shell dirname $(realpath \
				 $(word $(words $(MAKEFILE_LIST)), $(MAKEFILE_LIST))))

ROOT_DIR := $(shell realpath $(MAKEFILES_DIR)/../)
OP2_BUILD_DIR ?= $(ROOT_DIR)/op2

AR = ar rcs
INC += -Iinclude

ifdef CUDA_INSTALL_PATH
  include $(MAKEFILES_DIR)/nvcc.mk

  INC += -I$(CUDA_INSTALL_PATH)/include
  LIB += -lcudart
endif

ifdef MPI_INSTALL_PATH
  MPICC ?= $(MPI_INSTALL_PATH)/bin/mpicc
  MPICXX ?= $(MPI_INSTALL_PATH)/bin/mpic++
  MPIFC ?= $(MPI_INSTALL_PATH)/bin/mpifort

  INC += -I$(MPI_INSTALL_PATH)/include
  LIB += -L$(MPI_INSTALL_PATH)/lib -lmpi
endif

ifdef PARMETIS_INSTALL_PATH
  INC += -I$(PARMETIS_INSTALL_PATH)/include -DHAVE_PARMETIS -DPARMETIS_VER_4
  LIB += -L$(PARMETIS_INSTALL_PATH)/lib -lparmetis -lmetis
endif

ifdef PTSCOTCH_INSTALL_PATH
  INC += -I$(PTSCOTCH_INSTALL_PATH)/include -DHAVE_PTSCOTCH
  LIB += -L$(PTSCOTCH_INSTALL_PATH)/lib -lscotch -lptscotch -lptscotcherr
endif

ifdef HDF5_INSTALL_PATH
  INC += -I$(HDF5_INSTALL_PATH)/include
  LIB += -I$(HDF5_INSTALL_PATH)/lib -lhdf5 -lz -ldl -lm
endif

include $(MAKEFILES_DIR)/compilers/$(OP2_COMPILER).mk
