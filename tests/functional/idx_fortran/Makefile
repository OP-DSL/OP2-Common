APP_EXTRA_TRANSLATOR_FLAGS ?= -I $(OP2_INSTALL_PATH)/include

include ../../../makefiles/common.mk

APP_NAME := idx_tests
APP_SRC := idx_kernels.F90 idx_tests.F90

VARIANT_FILTER_OUT := mpi_%

include ../../../makefiles/f_app.mk

APP_NAME := idx_tests_par
APP_EXTRA_FLAGS := -DUSE_MPI

VARIANT_FILTER := mpi_%

include ../../../makefiles/f_app.mk



APP_NAME := idx_tests_soa
APP_EXTRA_FLAGS :=
APP_SRC := idx_kernels.F90 idx_tests.F90

VARIANT_FILTER_OUT := mpi_%
APP_EXTRA_TRANSLATOR_FLAGS += --force_soa

include ../../../makefiles/f_app.mk

APP_NAME := idx_tests_soa_par
APP_EXTRA_FLAGS := -DUSE_MPI

VARIANT_FILTER := mpi_%

include ../../../makefiles/f_app.mk
