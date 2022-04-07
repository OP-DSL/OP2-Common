PARMETIS_DEF ?= -DHAVE_PARMETIS -DPARMETIS_VER_4

ifdef PARMETIS_INSTALL_PATH
  PARMETIS_INC_PATH := -I$(PARMETIS_INSTALL_PATH)/include
  PARMETIS_LIB_PATH := -L$(PARMETIS_INSTALL_PATH)/lib
endif

PARMETIS_TEST = $(CONFIG_MPICXX) $(PARMETIS_INC_PATH) \
                    $(DEPS_DIR)/tests/parmetis.cpp $(PARMETIS_LIB_PATH) $(PARMETIS_LINK) \
                    -o $(DEPS_DIR)/tests/parmetis

$(file > $(DEP_BUILD_LOG),$(PARMETIS_TEST))
$(shell $(PARMETIS_TEST) >> $(DEP_BUILD_LOG) 2>&1)

ifneq ($(.SHELLSTATUS),0)
  PARMETIS_LINK ?= -lparmetis -lmetis

  $(file >> $(DEP_BUILD_LOG),$(PARMETIS_TEST))
  $(shell $(PARMETIS_TEST) >> $(DEP_BUILD_LOG) 2>&1)
endif

ifeq ($(.SHELLSTATUS),0)
  $(shell rm -f $(DEPS_DIR)/tests/parmetis)

  $(call info_bold,  > ParMETIS libraries $(TEXT_FOUND) (link flags: $(or $(PARMETIS_LINK),none)))

  CONFIG_HAVE_PARMETIS := true

  CONFIG_PARMETIS_INC := $(strip $(PARMETIS_INC_PATH) $(PARMETIS_DEF))
  CONFIG_PARMETIS_LIB := $(strip $(PARMETIS_LIB_PATH) $(PARMETIS_LINK))
else
  $(call info_bold,  > ParMETIS libraries $(TEXT_NOTFOUND):)
  $(info $(file < $(DEP_BUILD_LOG)))
  $(info )
endif
