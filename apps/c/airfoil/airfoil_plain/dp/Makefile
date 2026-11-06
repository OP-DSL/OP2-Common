include ../../../../../makefiles/common.mk

APP_NAME := airfoil
APP_SRC := airfoil.cpp

# No legacy GPU variants: update's op_arg_info is rejected there.
VARIANT_FILTER_OUT := mpi_% cuda hip

include ../../../../../makefiles/c_app.mk

APP_NAME := airfoil_par
APP_SRC := airfoil_mpi.cpp

VARIANT_FILTER := mpi_%
VARIANT_FILTER_OUT := mpi_cuda mpi_hip

include ../../../../../makefiles/c_app.mk
