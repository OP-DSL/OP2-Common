include ../../../../../makefiles/common.mk

APP_NAME := airfoil
APP_SRC := airfoil.cpp

VARIANT_FILTER_OUT := mpi_%

include ../../../../../makefiles/c_app.mk

APP_NAME := airfoil_par
APP_SRC := airfoil_mpi.cpp

VARIANT_FILTER := mpi_%

include ../../../../../makefiles/c_app.mk
