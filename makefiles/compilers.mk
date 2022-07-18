ifdef OP2_COMPILER
  OP2_C_COMPILER ?= $(OP2_COMPILER)
  OP2_F_COMPILER ?= $(OP2_COMPILER)
  OP2_C_CUDA_COMPILER ?= nvhpc
endif

# Process CUDA_GEN and NV_ARCH until CUDA_GEN is a whitespace separated list of
# numerical CUDA architectures
CUDA_GEN := $(subst $(COMMA),$(SPACE),$(CUDA_GEN))

CUDA_GEN_Fermi   := 20
CUDA_GEN_Kepler  := 35
CUDA_GEN_Maxwell := 50
CUDA_GEN_Pascal  := 60
CUDA_GEN_Volta   := 70
CUDA_GEN_Ampere  := 80

NV_ARCH := $(subst $(COMMA),$(SPACE),$(NV_ARCH))
$(foreach arch,$(NV_ARCH),$(eval CUDA_GEN += $(CUDA_GEN_$(arch))))

TARGET_HOST ?= true

# Include the relevant compiler makefiles
ifdef OP2_C_COMPILER
  include $(MAKEFILES_DIR)/compilers/c/$(OP2_C_COMPILER).mk
endif

ifdef OP2_C_CUDA_COMPILER
  include $(MAKEFILES_DIR)/compilers/c_cuda/$(OP2_C_CUDA_COMPILER).mk
endif

ifdef OP2_F_COMPILER
  include $(MAKEFILES_DIR)/compilers/fortran/$(OP2_F_COMPILER).mk
endif

ifeq ($(CONFIG_F_HAS_CUDA),true)
  CONFIG_CUDA_FFLAGS += -DOP2_WITH_CUDAFOR
endif

# Test if the compilers are present
ifneq ($(and $(shell which $(CONFIG_CC) 2> /dev/null),$(shell which $(CONFIG_CXX) 2> /dev/null)),)
  CONFIG_CC != which $(CONFIG_CC)
  CONFIG_CXX != which $(CONFIG_CXX)
  CONFIG_HAVE_C := true
endif

ifneq ($(shell which $(CONFIG_NVCC) 2> /dev/null),)
  CONFIG_NVCC != which $(CONFIG_NVCC)
  CONFIG_HAVE_C_CUDA := true
endif

ifneq ($(shell which $(CONFIG_FC) 2> /dev/null),)
  CONFIG_FC != which $(CONFIG_FC)
  CONFIG_HAVE_F := true
endif

# Check for the MPI compilers
ifdef MPI_INSTALL_PATH
  MPI_BIN ?= $(MPI_INSTALL_PATH)/bin/
endif

CONFIG_MPICC ?= $(MPI_BIN)mpicc
CONFIG_MPICXX ?= $(MPI_BIN)mpicxx
CONFIG_MPIFC ?= $(MPI_BIN)mpif90

ifneq ($(and $(shell which $(CONFIG_MPICC) 2>/dev/null),$(shell which $(CONFIG_MPICXX) 2> /dev/null)),)
  CONFIG_MPICC != which $(CONFIG_MPICC)
  CONFIG_MPICXX != which $(CONFIG_MPICXX)

  CONFIG_HAVE_MPI_C := true

  # Anti MPI C++ binding measures
  CONFIG_CFLAGS += -DOMPI_SKIP_MPICXX -DMPICH_IGNORE_CXX_SEEK -DMPIPP_H
  CONFIG_CXXFLAGS += -DOMPI_SKIP_MPICXX -DMPICH_IGNORE_CXX_SEEK -DMPIPP_H
endif

ifneq ($(shell which $(CONFIG_MPIFC) 2> /dev/null),)
  CONFIG_MPIFC != which $(CONFIG_MPIFC)
  CONFIG_HAVE_MPI_F := true
endif
