include ../../../makefiles/common.mk

APP_NAME := airfoil_plain
APP_SRC := airfoil_constants.F90 airfoil_kernels.F90 airfoil.F90
APP_EXTRA_FLAGS := -DPLAIN

VARIANT_FILTER_OUT := $(APP_NAME)_mpi_%

include ../../../makefiles/f_app.mk

APP_NAME := airfoil_arg_ptrs
APP_EXTRA_FLAGS := -DARG_PTRS

VARIANT_FILTER_OUT := $(APP_NAME)_mpi_%

include ../../../makefiles/f_app.mk

APP_NAME := airfoil_hdf5
APP_EXTRA_FLAGS := -DHDF5

OP2_LIBS_WITH_HDF5 := true

include ../../../makefiles/f_app.mk
