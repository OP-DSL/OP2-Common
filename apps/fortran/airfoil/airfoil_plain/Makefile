include ../../../../makefiles/common.mk

APP_NAME = airfoil

PART_SIZE_ENV ?= 128
FFLAGS += -DOP_PART_SIZE_1=$(PART_SIZE_ENV)

TARGETS = $(APP_NAME)_seq $(APP_NAME)_genseq $(APP_NAME)_openmp $(APP_NAME)_cuda

all: $(TARGETS)

KERNELS = $(patsubst %.inc,%,$(wildcard *.inc))
KERNEL_SOURCES = $(wildcard *.inc) $(wildcard *.inc2)

GEN_KERNELS = $(foreach kernel,$(KERNELS),$(kernel)_kernel.F90)
GEN_KERNELS_SEQ = $(foreach kernel,$(KERNELS),$(kernel)_seqkernel.F90)
GEN_KERNELS_CUDA = $(foreach kernel,$(KERNELS),$(kernel)_kernel.CUF)

GENERATED = \
	$(APP_NAME)_op.F90 \
	$(GEN_KERNELS) \
	$(GEN_KERNELS_SEQ) \
	$(GEN_KERNELS_CUDA)

$(GENERATED)&: $(APP_NAME).F90 $(KERNEL_SOURCES)
	$(ROOT_DIR)/translator/fortran/op2_fortran.py $<

$(APP_NAME)_seq: constants.F90 $(APP_NAME)_seqfun.F90 input.F90 $(APP_NAME).F90
	$(FC) $(FFLAGS) $(OP2_MOD) $^ $(OP2_LIB_FOR_SEQ) -o $@

$(APP_NAME)_genseq: constants.F90 $(GEN_KERNELS_SEQ) $(APP_NAME)_seqfun.F90 input.F90 $(APP_NAME)_op.F90
	$(FC) $(FFLAGS) $(OP2_MOD) $^ $(OP2_LIB_FOR_SEQ) -o $@

$(APP_NAME)_openmp: constants.F90 $(GEN_KERNELS) $(APP_NAME)_seqfun.F90 input.F90 $(APP_NAME)_op.F90
	$(FC) $(FFLAGS) $(OMP_FFLAGS) $(OP2_MOD) $^ $(OP2_LIB_FOR_OPENMP) -o $@

$(APP_NAME)_cuda: constants.F90 $(GEN_KERNELS_CUDA) $(APP_NAME)_seqfun.F90 input.F90 $(APP_NAME)_op.F90
	$(FC) $(FFLAGS) $(CUDA_FFLAGS) $(OP2_MOD) $^ $(OP2_LIB_FOR_CUDA) -o $@

clean:
	-rm -f $(TARGETS)
	-rm -f $(GENERATED)
	-rm -f *.mod
