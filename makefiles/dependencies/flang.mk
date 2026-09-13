# Optional LLVM Flang parser libraries used to build op2-flang-scan

ifdef FLANG_INSTALL_PATH
  LLVM_INSTALL_PATH ?= $(FLANG_INSTALL_PATH)
endif

CONFIG_FLANG_SCAN_CMAKE ?= cmake
ifneq ($(shell which $(CONFIG_FLANG_SCAN_CMAKE) 2>/dev/null),)
  CONFIG_FLANG_SCAN_CMAKE := $(shell which $(CONFIG_FLANG_SCAN_CMAKE))
  CONFIG_HAVE_CMAKE := true
endif

# True if $(1) looks like an LLVM prefix that includes Flang parser headers and the LLVM CMake
define flang_prefix_ok
$(and $(wildcard $(1)/include/flang/Parser/parsing.h),\
      $(wildcard $(1)/lib/cmake/llvm/LLVMConfig.cmake),\
      $(or $(wildcard $(1)/lib/libFortranParser.a),\
           $(wildcard $(1)/lib/libFortranParser.so)))
endef

FLANG_PREFIX_CANDIDATES :=
ifdef LLVM_INSTALL_PATH
  FLANG_PREFIX_CANDIDATES += $(LLVM_INSTALL_PATH)
endif
ifdef HOME
  FLANG_PREFIX_CANDIDATES += $(HOME)/.local/llvm
endif

LLVM_CONFIG_PREFIX := $(shell llvm-config --prefix 2>/dev/null || \
  llvm-config-20 --prefix 2>/dev/null || \
  llvm-config-19 --prefix 2>/dev/null || \
  llvm-config-18 --prefix 2>/dev/null)
ifneq ($(LLVM_CONFIG_PREFIX),)
  FLANG_PREFIX_CANDIDATES += $(LLVM_CONFIG_PREFIX)
endif

FLANG_PREFIX_CANDIDATES += $(shell ls -d /usr/lib/llvm-[0-9]* 2>/dev/null | sort -V -r)
FLANG_PREFIX_CANDIDATES += /usr

FLANG_DETECTED_PREFIX :=
$(foreach p,$(FLANG_PREFIX_CANDIDATES),\
  $(if $(FLANG_DETECTED_PREFIX),,\
    $(if $(call flang_prefix_ok,$(p)),\
      $(eval FLANG_DETECTED_PREFIX := $(p)))))

ifeq ($(CONFIG_HAVE_CMAKE),true)
  ifneq ($(strip $(FLANG_DETECTED_PREFIX)),)
    CONFIG_HAVE_FLANG := true
    CONFIG_LLVM_INSTALL_PATH := $(FLANG_DETECTED_PREFIX)

    ifneq ($(shell which ninja 2>/dev/null),)
      CONFIG_FLANG_SCAN_CMAKE_GENERATOR := Ninja
    else
      CONFIG_FLANG_SCAN_CMAKE_GENERATOR := Unix Makefiles
    endif

    $(call info_bold,> LLVM Flang $(TEXT_FOUND) ($(CONFIG_LLVM_INSTALL_PATH)))
  else
    $(call info_bold,> LLVM Flang $(TEXT_NOTFOUND))
    $(info .   Optional: set LLVM_INSTALL_PATH to an LLVM prefix that includes)
    $(info .   Flang parser headers (include/flang/Parser/parsing.h) and the)
    $(info .   LLVM/MLIR CMake packages. See docs/getting_started.rst.)
  endif
else
  $(call info_bold,> LLVM Flang skipped (cmake $(TEXT_NOTFOUND)))
  $(info .   CMake >= 3.26 is required to build op2-flang-scan)
endif
