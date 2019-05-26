
# Locate MPI compilers:
ifdef MPI_INSTALL_PATH
  ifneq ("","$(wildcard $(MPI_INSTALL_PATH)/bin/mpic++)")
    MPICPP_PATH = $(MPI_INSTALL_PATH)/bin/mpic++
  else
  ifneq ("","$(wildcard $(MPI_INSTALL_PATH)/intel64/bin/mpic++)")
    MPICPP_PATH = $(MPI_INSTALL_PATH)/intel64/bin/mpic++
  else
    MPICPP_PATH = mpic++
  endif
  endif

  ifneq ("","$(wildcard $(MPI_INSTALL_PATH)/bin/mpicxx)")
    MPICXX_PATH = $(MPI_INSTALL_PATH)/bin/mpicxx
  else
  ifneq ("","$(wildcard $(MPI_INSTALL_PATH)/intel64/bin/mpicxx)")
    MPICXX_PATH = $(MPI_INSTALL_PATH)/intel64/bin/mpicxx
  else
    MPICXX_PATH = mpicxx
  endif
  endif

  ifneq ("","$(wildcard $(MPI_INSTALL_PATH)/bin/mpicc)")
    MPICC_PATH = $(MPI_INSTALL_PATH)/bin/mpicc
  else
  ifneq ("","$(wildcard $(MPI_INSTALL_PATH)/intel64/bin/mpicc)")
    MPICC_PATH = $(MPI_INSTALL_PATH)/intel64/bin/mpicc
  else
    MPICC_PATH = mpicc
  endif
  endif
else
  MPICXX_PATH = mpicxx
  MPICC_PATH  = mpicc
endif

# OP2 paths
ifdef OP2_INSTALL_PATH
  OP2_INC = -I$(OP2_INSTALL_PATH)/c/include
  OP2_LIB = -L$(OP2_INSTALL_PATH)/c/lib
endif

# CUDA paths
ifdef CUDA_INSTALL_PATH
  CUDA_INC = -I$(CUDA_INSTALL_PATH)/include
  CUDA_LIB = -L$(CUDA_INSTALL_PATH)/lib64
endif

# HDF5 paths
ifdef HDF5_INSTALL_PATH
  HDF5_INC := -I$(HDF5_INSTALL_PATH)/include
  HDF5_LIB := -L$(HDF5_INSTALL_PATH)/lib
endif
HDF5_LIB += -lhdf5 -lz

#
# partitioning software for MPI versions
#
# ParMETIS
PARMETIS_VER=4
ifdef PARMETIS_INSTALL_PATH
  PARMETIS_INC = -I$(PARMETIS_INSTALL_PATH)/include
  PARMETIS_LIB = -L$(PARMETIS_INSTALL_PATH)/lib
endif
PARMETIS_INC += -DHAVE_PARMETIS
PARMETIS_LIB += -lparmetis -lmetis
ifeq ($(PARMETIS_VER),4)
  PARMETIS_INC += -DPARMETIS_VER_4
endif

# PT-Scotch
ifdef PTSCOTCH_INSTALL_PATH
  PTSCOTCH_INC = -I$(PTSCOTCH_INSTALL_PATH)/include
  PTSCOTCH_LIB = -L$(PTSCOTCH_INSTALL_PATH)/lib
endif
PTSCOTCH_INC += -DHAVE_PTSCOTCH
PTSCOTCH_LIB += -lptscotch -lscotch  -lptscotcherr
