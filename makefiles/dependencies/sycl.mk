ifdef SYCL_INSTALL_PATH
  SYCL_INC_PATH := -I$(SYCL_INSTALL_PATH)/include
  SYCL_LIB_PATH := -L$(SYCL_INSTALL_PATH)/lib
endif

SYCL_TEST = $(CONFIG_CXX) $(CONFIG_SYCLCCFLAGS) $(SYCL_INC_PATH) \
                    $(DEPS_DIR)/tests/sycl.cpp $(SYCL_LIB_PATH) $(SYCL_LINK) \
                    -o $(DEPS_DIR)/tests/sycl

$(file > $(DEP_BUILD_LOG),$(SYCL_TEST))
$(shell $(SYCL_TEST) >> $(DEP_BUILD_LOG) 2>&1)

ifeq ($(.SHELLSTATUS),0)
  $(shell rm -f $(DEPS_DIR)/tests/sycl)

  $(call info_bold,  > SYCL libraries $(TEXT_FOUND) (link flags: $(or $(SYCL_LINK), none)))

  CONFIG_HAVE_SYCL := true

  CONFIG_SYCLCC_INC := $(strip $(SYCL_INC_PATH) $(SYCL_DEF))
  CONFIG_SYCLCC_LIB := $(strip $(SYCL_LIB_PATH) $(SYCL_LINK))
else
  $(call info_bold,  > SYCL libraries $(TEXT_NOTFOUND):)
  $(info $(file < $(DEP_BUILD_LOG)))
  $(info )
endif
