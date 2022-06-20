include ../../../../makefiles/common.mk

APP_NAME := aero
APP_SRC := aero.cpp

VARIANT_FILTER_OUT := mpi_%

include ../../../../makefiles/c_app.mk

APP_NAME := aero_par
APP_SRC := aero_mpi.cpp

VARIANT_FILTER := mpi_%

include ../../../../makefiles/c_app.mk
