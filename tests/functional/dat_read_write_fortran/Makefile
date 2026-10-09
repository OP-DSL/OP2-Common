APP_EXTRA_TRANSLATOR_FLAGS ?= -I $(OP2_INSTALL_PATH)/include

# The OpenMP SIMD path loses updates when lanes share an indirect RW target.
NO_OPENMP := openmp mpi_openmp

include ../../../makefiles/common.mk

APP_NAME := rw_tests
APP_SRC := rw_kernels.F90 rw_tests.F90

VARIANT_FILTER_OUT := mpi_% $(NO_OPENMP)

include ../../../makefiles/f_app.mk

APP_NAME := rw_tests_par
APP_EXTRA_FLAGS := -DUSE_MPI

VARIANT_FILTER := mpi_%
VARIANT_FILTER_OUT := $(NO_OPENMP)

include ../../../makefiles/f_app.mk

APP_NAME := rw_tests_soa
APP_EXTRA_FLAGS :=

VARIANT_FILTER_OUT := mpi_% $(NO_OPENMP)
APP_EXTRA_TRANSLATOR_FLAGS += --force_soa

include ../../../makefiles/f_app.mk

APP_NAME := rw_tests_soa_par
APP_EXTRA_FLAGS := -DUSE_MPI

VARIANT_FILTER := mpi_%
VARIANT_FILTER_OUT := $(NO_OPENMP)

include ../../../makefiles/f_app.mk
