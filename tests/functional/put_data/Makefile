APP_EXTRA_TRANSLATOR_FLAGS ?= -I $(OP2_INSTALL_PATH)/include

include ../../../makefiles/common.mk

APP_NAME := put_data_tests
APP_SRC := put_data_tests.cpp

VARIANT_FILTER_OUT := mpi_%

include ../../../makefiles/c_app.mk

APP_NAME := put_data_tests_par

VARIANT_FILTER := mpi_%

include ../../../makefiles/c_app.mk



APP_NAME := put_data_tests_soa
APP_SRC := put_data_tests.cpp

VARIANT_FILTER_OUT := mpi_%
APP_EXTRA_TRANSLATOR_FLAGS += --force_soa

include ../../../makefiles/c_app.mk

APP_NAME := put_data_tests_soa_par

VARIANT_FILTER := mpi_%

include ../../../makefiles/c_app.mk
