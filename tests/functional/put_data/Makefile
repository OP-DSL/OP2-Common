APP_EXTRA_TRANSLATOR_FLAGS ?= -I $(OP2_INSTALL_PATH)/include

include ../../../makefiles/common.mk

# op_mpi_put_data is only provided by the MPI library, so only the MPI
# variants of this test are built.
APP_NAME := put_data_tests_par
APP_SRC := put_data_tests.cpp

VARIANT_FILTER := mpi_%

include ../../../makefiles/c_app.mk

APP_NAME := put_data_tests_soa_par
APP_EXTRA_TRANSLATOR_FLAGS += --force_soa

VARIANT_FILTER := mpi_%

include ../../../makefiles/c_app.mk
