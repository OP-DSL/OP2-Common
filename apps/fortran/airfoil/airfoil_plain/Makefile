include ../../../../makefiles/common.mk

PART_SIZE_ENV ?= 128
FFLAGS += -DOP_PART_SIZE_1=$(PART_SIZE_ENV)

all: airfoil_seq airfoil_genseq

clean:
	-rm -f airfoil_seq
	-rm -f *.mod

GENERATED = \
	

airfoil_seq: constants.F90 airfoil_seqfun.F90 input.F90 airfoil.F90
	$(FC) $(FFLAGS) $(OP2_MOD) $^ $(OP2_LIB_FOR_SEQ) -o $@

airfoil_genseq: constants.F90 input.F90 airfoil_seqfun.F90 save_soln_seqkernel.F90 update_seqkernel.F90 bres_calc_seqkernel.F90 res_calc_seqkernel.F90 adt_calc_seqkernel.F90 airfoil_op.F90
	$(FC) $(FFLAGS) $(OP2_MOD) $^ $(OP2_LIB_FOR_SEQ) -o $@
