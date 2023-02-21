KAHIP_DEF ?= -DHAVE_KAHIP

ifdef KAHIP_INSTALL_PATH
  KAHIP_INC_PATH := -I$(KAHIP_INSTALL_PATH)/include
  KAHIP_LIB_PATH := -L$(KAHIP_INSTALL_PATH)/lib
endif

KAHIP_TEST = $(CONFIG_MPICXX) $(KAHIP_INC_PATH) \
                    $(DEPS_DIR)/tests/kahip.cpp $(KAHIP_LIB_PATH) $(KAHIP_LINK) \
                    -o $(DEPS_DIR)/tests/kahip

$(file > $(DEP_BUILD_LOG),$(KAHIP_TEST))
$(shell $(KAHIP_TEST) >> $(DEP_BUILD_LOG) 2>&1)

ifneq ($(.SHELLSTATUS),0)
  KAHIP_LINK ?= -lparhip_interface

  $(file >> $(DEP_BUILD_LOG),$(KAHIP_TEST))
  $(shell $(KAHIP_TEST) >> $(DEP_BUILD_LOG) 2>&1)
endif

ifeq ($(.SHELLSTATUS),0)
  $(shell rm -f $(DEPS_DIR)/tests/kahip)

  $(call info_bold,  > KaHIP libraries $(TEXT_FOUND) (link flags: $(or $(KAHIP_LINK),none)))

  CONFIG_HAVE_KAHIP := true

  CONFIG_KAHIP_INC := $(strip $(KAHIP_INC_PATH) $(KAHIP_DEF))
  CONFIG_KAHIP_LIB := $(strip $(KAHIP_LIB_PATH) $(KAHIP_LINK))
else
  $(call info_bold,  > KaHIP libraries $(TEXT_NOTFOUND):)
  $(info $(file < $(DEP_BUILD_LOG)))
  $(info )
endif