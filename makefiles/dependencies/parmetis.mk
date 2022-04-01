PARMETIS_DEF ?= -DHAVE_PARMETIS -DPARMETIS_VER_4

ifdef PARMETIS_INSTALL_PATH
  PARMETIS_INC_PATH := -I$(PARMETIS_INSTALL_PATH)/include
  PARMETIS_LIB_PATH := -L$(PARMETIS_INSTALL_PATH)/lib
endif

PARMETIS_TEST = $(CONFIG_MPICXX) $(PARMETIS_INC_PATH) \
                    $(DEPS_DIR)/tests/parmetis.cpp $(PARMETIS_LIB_PATH) $(PARMETIS_LINK) \
                    -o $(DEPS_DIR)/tests/parmetis

$(info ## Looking for the ParMETIS libraries...)

$(info ### Testing presence of implicitly linked libraries)
$(info .   $(PARMETIS_TEST))
$(shell $(PARMETIS_TEST))

ifneq ($(.SHELLSTATUS),0)
  PARMETIS_LINK ?= -lparmetis -lmetis

  $(info ### Testing presence of explicitly linked libraries)
  $(info .   $(PARMETIS_TEST))
  $(shell $(PARMETIS_TEST))
endif

ifeq ($(.SHELLSTATUS),0)
  $(shell rm -f $(DEPS_DIR)/tests/parmetis)

  $(info ## ParMETIS libraries $(TEXT_FOUND))

  CONFIG_HAVE_PARMETIS := true

  CONFIG_PARMETIS_INC := $(strip $(PARMETIS_INC_PATH) $(PARMETIS_DEF))
  CONFIG_PARMETIS_LIB := $(strip $(PARMETIS_LIB_PATH) $(PARMETIS_LINK))
else
  $(info ## ParMETIS libraries $(TEXT_NOTFOUND))
endif

$(info )
