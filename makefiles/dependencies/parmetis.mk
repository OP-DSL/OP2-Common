PARMETIS_DEF ?= -DHAVE_PARMETIS -DPARMETIS_VER_4
PARMETIS_LINK ?= -lparmetis -lmetis

ifdef PARMETIS_INSTALL_PATH
  PARMETIS_INC_PATH := -I$(PARMETIS_INSTALL_PATH)/include
  PARMETIS_LIB_PATH := -L$(PARMETIS_INSTALL_PATH)/lib
endif

PARMETIS_TEST = $(MPICXX) $(PARMETIS_INC_PATH) \
                    $(DEPS_DIR)/tests/parmetis.cpp $(PARMETIS_LIB_PATH) $(PARMETIS_LINK) \
                    -o $(DEPS_DIR)/tests/parmetis 2> /dev/null

$(shell $(PARMETIS_TEST))

ifneq ($(.SHELLSTATUS),0)
  PARMETIS_LINK :=
  $(shell $(PARMETIS_TEST))
endif

ifeq ($(.SHELLSTATUS),0)
  $(shell rm -f $(DEPS_DIR)/tests/parmetis)

  HAVE_PARMETIS := true

  PARMETIS_INC := $(PARMETIS_INC_PATH) $(PARMETIS_DEF)
  PARMETIS_LIB := $(PARMETIS_LIB_PATH) $(PARMETIS_LINK)
endif
