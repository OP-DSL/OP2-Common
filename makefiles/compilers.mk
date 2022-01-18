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

ifeq ($(F_HAS_CUDA),true)
  CUDA_FFLAGS += -DOP2_WITH_CUDAFOR
endif

# Test if the compilers are present
ifneq ($(and $(shell which $(CC) 2> /dev/null),$(shell which $(CXX) 2> /dev/null)),)
  HAVE_C := true
endif

ifneq ($(shell which $(NVCC) 2> /dev/null),)
  HAVE_C_CUDA := true
endif

ifneq ($(shell which $(FC) 2> /dev/null),)
  HAVE_F := true
endif

# Check for the MPI compilers
ifdef MPI_INSTALL_PATH
  MPI_BIN ?= $(MPI_INSTALL_PATH)/bin/
endif

MPICC ?= $(MPI_BIN)mpicc
MPICXX ?= $(MPI_BIN)mpicxx
MPIFC ?= $(MPI_BIN)mpif90

ifneq ($(and $(shell which $(MPICC) 2>/dev/null),$(shell which $(MPICXX) 2> /dev/null)),)
  HAVE_MPI_C := true

  # Anti MPI C++ binding measures
  CFLAGS += -DOMPI_SKIP_MPICXX -DMPICH_IGNORE_CXX_SEEK -DMPIPP_H
  CXXFLAGS += -DOMPI_SKIP_MPICXX -DMPICH_IGNORE_CXX_SEEK -DMPIPP_H
endif

ifneq ($(shell which $(MPIFC) 2> /dev/null),)
  HAVE_MPI_F := true
endif
