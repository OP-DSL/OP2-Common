ifdef HIP_INSTALL_PATH
  HIP_INC_PATH := -I$(HIP_INSTALL_PATH)/include
  HIP_LIB_PATH := -L$(HIP_INSTALL_PATH)/lib64 -L$(HIP_INSTALL_PATH)/lib
endif

HIP_TEST = $(CONFIG_HIP) $(HIP_INC_PATH) \
                    $(DEPS_DIR)/tests/hip.cpp $(HIP_LIB_PATH) $(HIP_LINK) \
                    -o $(DEPS_DIR)/tests/hip

$(file > $(DEP_BUILD_LOG),$(HIP_TEST))
$(shell $(HIP_TEST) >> $(DEP_BUILD_LOG) 2>&1)

ifneq ($(.SHELLSTATUS),0)
  HIP_LINK ?= #-lculibos -lpthread -lrt -ldl

  $(file >> $(DEP_BUILD_LOG),$(HIP_TEST))
  $(shell $(HIP_TEST) >> $(DEP_BUILD_LOG) 2>&1)
endif

ifeq ($(.SHELLSTATUS),0)
  $(shell rm -f $(DEPS_DIR)/tests/hip)

  $(call info_bold,  > HIP libraries $(TEXT_FOUND) (link flags: $(or $(HIP_LINK), none)))

  CONFIG_HAVE_HIP := true

  CONFIG_HIP_INC := $(strip $(HIP_INC_PATH) $(HIP_DEF))
  CONFIG_HIP_LIB := $(strip $(HIP_LIB_PATH) $(HIP_LINK))
else
  $(call info_bold,  > HIP libraries $(TEXT_NOTFOUND):)
  $(info $(file < $(DEP_BUILD_LOG)))
  $(info )
endif
