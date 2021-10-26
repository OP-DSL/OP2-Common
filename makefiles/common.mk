SHELL = /bin/sh
.SUFFIXES:

# Get the makefiles directory (where this file is)
MAKEFILES_DIR := $(shell dirname $(realpath \
				 $(word $(words $(MAKEFILE_LIST)), $(MAKEFILE_LIST))))

include $(MAKEFILES_DIR)/compilers/mpi.mk
include $(MAKEFILES_DIR)/compilers/nvcc.mk
include $(MAKEFILES_DIR)/compilers/$(OP2_COMPILER).mk

INC = -Iinclude

ifdef CUDA_INSTALL_PATH
  INC += -I$(CUDA_INSTALL_PATH)/include
endif

ifdef MPI_INSTALL_PATH
  MPI_INC = -I$(MPI_INSTALL_PATH)/include
endif

PARMETIS_VER=4
ifdef PARMETIS_INSTALL_PATH
  INC += -I$(PARMETIS_INSTALL_PATH)/include
  INC += -DHAVE_PARMETIS
endif

ifeq ($(PARMETIS_VER),4)
  INC += -DPARMETIS_VER_4
endif

ifdef PTSCOTCH_INSTALL_PATH
  INC += -I$(PTSCOTCH_INSTALL_PATH)/include
  INC += -DHAVE_PTSCOTCH
endif

ifdef HDF5_INSTALL_PATH
  INC += -I$(HDF5_INSTALL_PATH)/include -I$(HDF5_INSTALL_PATH)/gnu/9.1/include
endif

AR = ar rcs
