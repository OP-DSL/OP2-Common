APP_EXTRA_TRANSLATOR_FLAGS ?= -I $(OP2_INSTALL_PATH)/include

include ../../../makefiles/common.mk

APP_NAME := dat_increments
APP_SRC := dat_increments.cpp

VARIANT_FILTER_OUT := mpi_%

include ../../../makefiles/c_app.mk

APP_NAME := dat_increments_soa

APP_EXTRA_TRANSLATOR_FLAGS += --force_soa

include ../../../makefiles/c_app.mk
