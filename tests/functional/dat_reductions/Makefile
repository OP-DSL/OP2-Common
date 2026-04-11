APP_EXTRA_TRANSLATOR_FLAGS ?= -I $(OP2_INSTALL_PATH)/include

include ../../../makefiles/common.mk

APP_NAME := reduc_tests
APP_SRC := reduc_tests.cpp

VARIANT_FILTER_OUT := mpi_%

include ../../../makefiles/c_app.mk

APP_NAME := reduc_tests_par

VARIANT_FILTER := mpi_%

include ../../../makefiles/c_app.mk



APP_NAME := reduc_tests_soa
APP_SRC := reduc_tests.cpp

VARIANT_FILTER_OUT := mpi_%
APP_EXTRA_TRANSLATOR_FLAGS += --force_soa

include ../../../makefiles/c_app.mk

APP_NAME := reduc_tests_soa_par

VARIANT_FILTER := mpi_%

include ../../../makefiles/c_app.mk
