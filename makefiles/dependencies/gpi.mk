GPI_DEF ?= -DHAVE_GPI
ifdef GPI_INSTALL_PATH
  #GPI_INC_PATH := -I$(GPI_INSTALL_PATH)/src/include
  #GPI_LIB_PATH := -L$(GPI_INSTALL_PATH)/src/.libs
  GPI_INC_PATH := -I$(GPI_INSTALL_PATH)/include
  GPI_LIB_PATH := -L$(GPI_INSTALL_PATH)/lib64
  GPI_LIB_PATH += -L$(GPI_INSTALL_PATH)/lib
endif

# Change to MPI-compiled GASPI [DONE?]
GPI_TEST = $(CONFIG_MPICXX) $(GPI_INC_PATH) \
		$(DEPS_DIR)/tests/gpi_mpi.cpp $(GPI_LIB_PATH) $(GPI_LINK) \
		-o $(DEPS_DIR)/tests/gpi_mpi

#$(file > $(DEP_BULD_LOG),$(GPI_TEST))
$(shell $(GPI_TEST) >> $(DEP_BUILD_LOG) 2>&1)

ifneq ($(.SHELLSTATUS),0)
  GPI_LINK ?= -Wl,-rpath,$(GPI_INSTALL_PATH)/lib64 -Wl,-rpath,$(GPI_INSTALL_PATH)/lib -lGPI2 -libverbs -lm -lpthread 

  $(file >> $(DEP_BUILD_LOG),$(GPI_TEST))
  $(shell $(GPI_TEST) >> $(DEP_BUILD_LOG) 2>&1)
endif

ifeq ($(.SHELLSTATUS),0)
  $(shell rm -f $(DEPS_DIR)/tests/gpi_mpi)
  $(call info_bold,  > GPI2 library $(TEXT_FOUND) )

  CONFIG_HAVE_GPI := true
  CONFIG_GPI_INC := $(strip $(GPI_INC_PATH) $(GPI_DEF))
  CONFIG_GPI_LIB := $(strip $(GPI_LIB_PATH) $(GPI_LINK))
else
  $(call info_bold,  > GPI-2 library $(TEXT_NOTFOUND):)
  $(info $(file < $(DEP_BUILD_LOG)))
  $(info )
endif
