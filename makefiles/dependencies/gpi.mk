ifdef GPI2_INSTALL_PATH
  #GPI2_INC_PATH := -I$(GPI2_INSTALL_PATH)/src/include
  #GPI2_LIB_PATH := -L$(GPI2_INSTALL_PATH)/src/.libs
  GPI2_INC_PATH := -I$(GPI2_INSTALL_PATH)/include
  GPI2_LIB_PATH := -L$(GPI2_INSTALL_PATH)/lib64
endif

GPI2_TEST = $(CONFIG_CXX) $(GPI2_INC_PATH) \
		$(DEPS_DIR)/tests/gpi.cpp $(GPI2_LIB_PATH) $(GPI2_LINK) \
		-o $(DEPS_DIR)/tests/gpi

$(file > $(DEP_BULD_LOG),$(GPI2_TEST))
$(shell $(GPI2_TEST) >> $(DEP_BUILD_LOG) 2>&1)

ifneq ($(.SHELLSTATUS),0)
  GPI2_LINK ?= -Wl,-rpath,$(GPI2_INSTALL_PATH)/lib64 -lGPI2 -libverbs -lm -lpthread

  $(file >> $(DEP_BUILD_LOG),$(GPI2_TEST))
  $(shell $(GPI2_TEST) >> $(DEP_BUILD_LOG) 2>&1)
endif

ifeq ($(.SHELLSTATUS),0)
  $(shell rm -f $(DEPS_DIR)/tests/gpi)
  $(call info_bold,  > GPI2 library $(TEXT_FOUND) )

  CONFIG_HAVE_GPI2 := true
  CONFIG_GPI2_INC := $(strip $(GPI2_INC_PATH))
  CONFIG_GPI2_LIB := $(strip $(GPI2_LIB_PATH) $(GPI2_LINK_FLAGS))
else
  $(call info_bold,  > GPI2 library $(TEXT_NOTFOUND):)
  $(info $(file < $(DEP_BUILD_LOG)))
  $(info )
endif
