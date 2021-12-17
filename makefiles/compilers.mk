ifdef OP2_COMPILER
  OP2_C_COMPILER ?= $(OP2_COMPILER)
  OP2_F_COMPILER ?= $(OP2_COMPILER)
  OP2_C_CUDA_COMPILER ?= nvhpc
endif

ifdef OP2_C_COMPILER
  include $(MAKEFILES_DIR)/compilers/c/$(OP2_C_COMPILER).mk
endif

ifdef OP2_C_CUDA_COMPILER
  include $(MAKEFILES_DIR)/compilers/c_cuda/$(OP2_C_CUDA_COMPILER).mk
endif

ifdef OP2_F_COMPILER
  include $(MAKEFILES_DIR)/compilers/fortran/$(OP2_F_COMPILER).mk
endif

ifeq ($(F_HAS_CUDA),true)
  CUDA_FFLAGS += -DOP2_WITH_CUDAFOR
endif

# Test if the compilers are present
ifneq ($(and $(shell which $(CC)),$(shell which $(CXX))),)
  HAVE_C := true
endif

ifneq ($(shell which $(NVCC)),)
  HAVE_C_CUDA := true
endif

ifneq ($(shell which $(FC)),)
  HAVE_F := true
endif

# Check for the MPI compilers
ifdef MPI_INSTALL_PATH
  MPI_BIN ?= $(MPI_INSTALL_PATH)/bin/
endif

MPICC ?= $(MPI_BIN)mpicc
MPICXX ?= $(MPI_BIN)mpicxx
MPIFC ?= $(MPI_BIN)mpif90

ifneq ($(and $(shell which $(CC)),$(shell which $(CXX))),)
  HAVE_MPI_C := true

  # Anti MPI C++ binding measures
  CFLAGS += -DOMPI_SKIP_MPICXX -DMPICH_IGNORE_CXX_SEEK -DMPIPP_H
  CXXFLAGS += -DOMPI_SKIP_MPICXX -DMPICH_IGNORE_CXX_SEEK -DMPIPP_H
endif

ifneq ($(shell which $(MPIFC)),)
  HAVE_MPI_F := true
endif
