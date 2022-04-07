PTSCOTCH_DEF ?= -DHAVE_PTSCOTCH

ifdef PTSCOTCH_INSTALL_PATH
  PTSCOTCH_INC_PATH := -I$(PTSCOTCH_INSTALL_PATH)/include
  PTSCOTCH_LIB_PATH := -L$(PTSCOTCH_INSTALL_PATH)/lib
endif

PTSCOTCH_TEST = $(CONFIG_MPICXX) $(PTSCOTCH_INC_PATH) \
                    $(DEPS_DIR)/tests/ptscotch.cpp $(PTSCOTCH_LIB_PATH) $(PTSCOTCH_LINK) \
                    -o $(DEPS_DIR)/tests/ptscotch

$(file > $(DEP_BUILD_LOG),$(PTSCOTCH_TEST))
$(shell $(PTSCOTCH_TEST) >> $(DEP_BUILD_LOG) 2>&1)

ifneq ($(.SHELLSTATUS),0)
  PTSCOTCH_LINK ?= -lptscotch -lscotch -lptscotcherr

  $(file >> $(DEP_BUILD_LOG),$(PTSCOTCH_TEST))
  $(shell $(PTSCOTCH_TEST) >> $(DEP_BUILD_LOG) 2>&1)
endif

ifeq ($(.SHELLSTATUS),0)
  $(shell rm -f $(DEPS_DIR)/tests/ptscotch)

  $(call info_bold,  > PT-Scotch libraries $(TEXT_FOUND) (link flags: $(or $(PTSCOTCH_LINK),none)))

  CONFIG_HAVE_PTSCOTCH := true

  CONFIG_PTSCOTCH_INC := $(strip $(PTSCOTCH_INC_PATH) $(PTSCOTCH_DEF))
  CONFIG_PTSCOTCH_LIB := $(strip $(PTSCOTCH_LIB_PATH) $(PTSCOTCH_LINK))
else
  $(call info_bold,  > PT-Scotch libraries $(TEXT_NOTFOUND):)
  $(info $(file < $(DEP_BUILD_LOG)))
  $(info )
endif
