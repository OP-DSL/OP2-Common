# Library dependencies
CUDA_INC ?=
CUDA_LIB ?= -lculibos -lcudart_static -lpthread -lrt -ldl

PARMETIS_INC ?= -DHAVE_PARMETIS -DPARMETIS_VER_4
PARMETIS_LIB ?= -lparmetis -lmetis

PTSCOTCH_INC ?= -DHAVE_PTSCOTCH
PTSCOTCH_LIB ?= -lptscotch -lscotch -lptscotcherr

HDF5_SEQ_INC ?=
HDF5_SEQ_LIB ?= -lhdf5 -ldl -lm -lz

HDF5_PAR_INC ?=
HDF5_PAR_LIB ?= $(HDF5_SEQ_LIB)

# Manually specified dependency paths
ifdef CUDA_INSTALL_PATH
  CUDA_INC := -I$(CUDA_INSTALL_PATH)/include $(CUDA_INC)
  CUDA_LIB := -L$(CUDA_INSTALL_PATH)/lib64 -L$(CUDA_INSTALL_PATH)/lib $(CUDA_LIB)
endif

ifdef PARMETIS_INSTALL_PATH
  PARMETIS_INC := -I$(PARMETIS_INSTALL_PATH)/include $(PARMETIS_INC)
  PARMETIS_LIB := -L$(PARMETIS_INSTALL_PATH)/lib $(PARMETIS_LIB)
endif

ifdef PTSCOTCH_INSTALL_PATH
  PTSCOTCH_INC := -I$(PTSCOTCH_INSTALL_PATH)/include $(PTSCOTCH_INC)
  PTSCOTCH_LIB := -L$(PTSCOTCH_INSTALL_PATH)/lib $(PTSCOTCH_LIB)
endif

ifdef HDF5_INSTALL_PATH
  HDF5_IS_PAR != grep "^\s*\#define\s*H5_HAVE_PARALLEL\s*1" \
	              $(HDF5_INSTALL_PATH)/include/H5pubconf.h

  ifneq ($(HDF5_IS_PAR),)
    HDF5_PAR_INSTALL_PATH = $(HDF5_INSTALL_PATH)
  else
    HDF5_SEQ_INSTALL_PATH = $(HDF5_INSTALL_PATH)
  endif
endif

ifdef HDF5_SEQ_INSTALL_PATH
  HDF5_SEQ_INC := -I$(HDF5_SEQ_INSTALL_PATH)/include $(HDF5_SEQ_INC)
  HDF5_SEQ_LIB := -L$(HDF5_SEQ_INSTALL_PATH)/lib -Wl,-rpath,$(HDF5_SEQ_INSTALL_PATH)/lib \
		  $(HDF5_SEQ_LIB)
endif

ifdef HDF5_PAR_INSTALL_PATH
  HDF5_PAR_INC := -I$(HDF5_PAR_INSTALL_PATH)/include $(HDF5_PAR_INC)
  HDF5_PAR_LIB := -L$(HDF5_PAR_INSTALL_PATH)/lib -Wl,-rpath,$(HDF5_PAR_INSTALL_PATH)/lib \
		  $(HDF5_PAR_LIB)
endif

# Try to detect variant of implicit HDF5
ifeq ($(and $(HDF5_SEQ_INSTALL_PATH),$(HDF5_PAR_INSTALL_PATH)),)
  $(shell $(CXX) $(CXXFLAGS) $(MAKEFILES_DIR)/test_hdf5.cpp $(HDF5_PAR_LIB) \
	  -o $(MAKEFILES_DIR)/test_hdf5 2> /dev/null)

  ifeq  ($(.SHELLSTATUS),0)
    RESULT != $(MAKEFILES_DIR)/test_hdf5
    $(shell rm -f $(MAKEFILES_DIR)/test_hdf5)

    $(info HDF5 implicit check result: $(RESULT))

    ifeq ($(RESULT),0)
      HDF5_SEQ_INSTALL_PATH := <implicit>
    else ifeq ($(RESULT),0))
      HDF5_PAR_INSTALL_PATH := <implicit>
    endif
  endif
endif
