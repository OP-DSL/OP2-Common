APP_EXTRA_TRANSLATOR_FLAGS ?= -I $(OP2_INSTALL_PATH)/include

include ../../../makefiles/common.mk

APP_NAME := reduc_tests
APP_SRC := reduc_kernels.F90 reduc_tests.F90

VARIANT_FILTER_OUT := mpi_%

include ../../../makefiles/f_app.mk

APP_NAME := reduc_tests_par
APP_EXTRA_FLAGS := -DUSE_MPI

VARIANT_FILTER := mpi_%

include ../../../makefiles/f_app.mk

APP_NAME := reduc_tests_soa
APP_EXTRA_FLAGS :=

VARIANT_FILTER_OUT := mpi_%
APP_EXTRA_TRANSLATOR_FLAGS += --force_soa

include ../../../makefiles/f_app.mk

APP_NAME := reduc_tests_soa_par
APP_EXTRA_FLAGS := -DUSE_MPI

VARIANT_FILTER := mpi_%

include ../../../makefiles/f_app.mk
