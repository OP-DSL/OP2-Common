PTSCOTCH_DEF ?= -DHAVE_PTSCOTCH

ifdef PTSCOTCH_INSTALL_PATH
  PTSCOTCH_INC_PATH := -I$(PTSCOTCH_INSTALL_PATH)/include
  PTSCOTCH_LIB_PATH := -L$(PTSCOTCH_INSTALL_PATH)/lib
endif

PTSCOTCH_TEST = $(MPICXX) $(PTSCOTCH_INC_PATH) \
                    $(DEPS_DIR)/tests/ptscotch.cpp $(PTSCOTCH_LIB_PATH) $(PTSCOTCH_LINK) \
                    -o $(DEPS_DIR)/tests/ptscotch 2> /dev/null

$(shell $(PTSCOTCH_TEST))

ifneq ($(.SHELLSTATUS),0)
  PTSCOTCH_LINK ?= -lptscotch -lscotch -lptscotcherr
  $(shell $(PTSCOTCH_TEST))
endif

ifeq ($(.SHELLSTATUS),0)
  $(shell rm -f $(DEPS_DIR)/tests/ptscotch)

  HAVE_PTSCOTCH := true

  PTSCOTCH_INC := $(strip $(PTSCOTCH_INC_PATH) $(PTSCOTCH_DEF))
  PTSCOTCH_LIB := $(strip $(PTSCOTCH_LIB_PATH) $(PTSCOTCH_LINK))
endif
