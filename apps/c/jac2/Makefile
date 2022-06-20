include ../../../makefiles/common.mk

APP_NAME := jac
APP_SRC := jac.cpp

VARIANT_FILTER_OUT := mpi_%

include ../../../makefiles/c_app.mk

APP_NAME := jac_par
APP_SRC := jac_mpi.cpp

VARIANT_FILTER := mpi_%

include ../../../makefiles/c_app.mk
