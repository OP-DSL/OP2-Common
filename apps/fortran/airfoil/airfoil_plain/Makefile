include ../../../../makefiles/common.mk

PART_SIZE_ENV ?= 128
FFLAGS += -DOP_PART_SIZE_1=$(PART_SIZE_ENV)

all: airfoil_seq

airfoil_seq: constants.F90 airfoil.F90 airfoil_seqfun.F90 input.F90
	$(FC) $(FFLAGS) $(OP2_MOD) $^ $(OP2_LIB_FOR_SEQ) -o $@
