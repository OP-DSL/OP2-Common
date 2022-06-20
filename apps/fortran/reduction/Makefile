include ../../../makefiles/common.mk

APP_NAME := reduction
APP_SRC := reduction.F90

VARIANT_FILTER_OUT := mpi_%

include ../../../makefiles/f_app.mk

APP_NAME := reduction_hdf5
APP_EXTRA_FLAGS := -DHDF5

OP2_LIBS_WITH_HDF5 := true

include ../../../makefiles/f_app.mk
