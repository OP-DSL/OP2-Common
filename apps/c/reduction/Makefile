include ../../../makefiles/common.mk

APP_NAME := reduction
APP_SRC := reduction.cpp

VARIANT_FILTER_OUT := mpi_%

include ../../../makefiles/c_app.mk

APP_NAME := reduction_par
APP_SRC := reduction_mpi.cpp

VARIANT_FILTER := mpi_%

include ../../../makefiles/c_app.mk
