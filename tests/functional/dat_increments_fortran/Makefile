APP_EXTRA_TRANSLATOR_FLAGS ?= -I $(OP2_INSTALL_PATH)/include

include ../../../makefiles/common.mk

APP_NAME := dat_increments
APP_SRC := dat_increments.F90

VARIANT_FILTER_OUT := mpi_% openmp

include ../../../makefiles/f_app.mk

APP_NAME := dat_increments_soa

APP_EXTRA_TRANSLATOR_FLAGS += --force_soa

include ../../../makefiles/f_app.mk
