PART_SIZE_ENV ?= 128
FFLAGS += -DOP_PART_SIZE_1=$(PART_SIZE_ENV)

ALL_APP_VARIANTS := seq genseq vec openmp openmp4
ALL_APP_VARIANTS := $(ALL_APP_VARIANTS) $(foreach variant,$(ALL_APP_VARIANTS),mpi_$(variant))

BUILDABLE_APP_VARIANTS := seq genseq

ifeq ($(F_HAS_OMP),true)
	BUILDABLE_APP_VARIANTS += vec openmp
endif

ifeq ($(F_HAS_OMP_OFFLOAD),true)
	BUILDABLE_APP_VARIANTS += openmp4
endif

ifneq ($(shell which $(MPIFC) 2> /dev/null),)
  BUILDABLE_APP_VARIANTS += $(foreach variant,$(BUILDABLE_APP_VARIANTS),mpi_$(variant))
endif

VARIANT_FILTER ?= %
VARIANT_FILTER_OUT ?=

BUILDABLE_APP_VARIANTS := $(filter-out $(VARIANT_FILTER_OUT),\
						  $(filter $(VARIANT_FILTER),$(BUILDABLE_APP_VARIANTS)))

ALL_APP_VARIANTS := $(foreach variant,$(ALL_APP_VARIANTS),$(APP_NAME)_$(variant))
BUILDABLE_APP_VARIANTS := $(foreach variant,$(BUILDABLE_APP_VARIANTS),$(APP_NAME)_$(variant))

KERNELS = $(patsubst %.inc,%,$(wildcard *.inc))
KERNEL_SOURCES = $(wildcard *.inc) $(wildcard *.inc2)

GEN_KERNELS = $(foreach kernel,$(KERNELS),$(kernel)_kernel.F90)
GEN_KERNELS_SEQ = $(foreach kernel,$(KERNELS),$(kernel)_seqkernel.F90)
GEN_KERNELS_VEC = $(foreach kernel,$(KERNELS),$(kernel)_veckernel.F90)
GEN_KERNELS_OMP4 = $(foreach kernel,$(KERNELS),$(kernel)_omp4kernel.F90)
GEN_KERNELS_CUDA = $(foreach kernel,$(KERNELS),$(kernel)_kernel.CUF)

GENERATED = \
	$(APP_NAME)_op.F90 \
	$(GEN_KERNELS) \
	$(GEN_KERNELS_SEQ) \
	$(GEN_KERNELS_VEC) \
	$(GEN_KERNELS_OMP4) \
	$(GEN_KERNELS_CUDA)

.PHONY: all generate clean

all: $(BUILDABLE_APP_VARIANTS)

clean:
	-rm -f $(ALL_APP_VARIANTS)
	-rm -f $(GENERATED)
	-rm -f *.mod
	-rm -rf mod

generate:
	[ ! -f $(APP_NAME).F90 ] || $(ROOT_DIR)/translator/fortran/op2_fortran.py $(APP_NAME).F90

mod/%:
	@mkdir -p $@

$(APP_NAME)_seq: constants.F90 $(APP_NAME)_seqfun.F90 input.F90 $(APP_NAME).F90 | mod/seq
	$(FC) $(FFLAGS) $(F_MOD_OUT_OPT)$| $(OP2_MOD) $^ $(OP2_LIB_FOR_SEQ) $(CXXLINK) -o $@

$(APP_NAME)_mpi_seq: constants.F90 $(APP_NAME)_seqfun.F90 input.F90 $(APP_NAME).F90 | mod/mpi_seq
	$(MPIFC) $(FFLAGS) $(F_MOD_OUT_OPT)$| $(OP2_MOD) $^ $(OP2_LIB_FOR_MPI) $(CXXLINK) -o $@

$(APP_NAME)_genseq: constants.F90 $(GEN_KERNELS_SEQ) $(APP_NAME)_seqfun.F90 input.F90 $(APP_NAME)_op.F90 | mod/genseq
	$(FC) $(FFLAGS) $(F_MOD_OUT_OPT)$| $(OP2_MOD) $^ $(OP2_LIB_FOR_SEQ) $(CXXLINK) -o $@

$(APP_NAME)_mpi_genseq: constants.F90 $(GEN_KERNELS_SEQ) $(APP_NAME)_seqfun.F90 input.F90 $(APP_NAME)_op.F90 | mod/mpi_genseq
	$(MPIFC) $(FFLAGS) $(F_MOD_OUT_OPT)$| $(OP2_MOD) $^ $(OP2_LIB_FOR_MPI) $(CXXLINK) -o $@

$(APP_NAME)_vec: constants.F90 $(GEN_KERNELS_VEC) $(APP_NAME)_seqfun.F90 input.F90 $(APP_NAME)_op.F90 | mod/vec
	$(FC) $(FFLAGS) $(OMP_FFLAGS) -DVECTORIZE $(F_MOD_OUT_OPT)$| $(OP2_MOD) $^ $(OP2_LIB_FOR_SEQ) $(CXXLINK) -o $@

$(APP_NAME)_mpi_vec: constants.F90 $(GEN_KERNELS_VEC) $(APP_NAME)_seqfun.F90 input.F90 $(APP_NAME)_op.F90 | mod/mpi_vec
	$(MPIFC) $(FFLAGS) $(OMP_FFLAGS) -DVECTORIZE $(F_MOD_OUT_OPT)$| $(OP2_MOD) $^ $(OP2_LIB_FOR_MPI) $(CXXLINK) -o $@

$(APP_NAME)_openmp: constants.F90 $(GEN_KERNELS) $(APP_NAME)_seqfun.F90 input.F90 $(APP_NAME)_op.F90 | mod/openmp
	$(FC) $(FFLAGS) $(OMP_FFLAGS) $(F_MOD_OUT_OPT)$| $(OP2_MOD) $^ $(OP2_LIB_FOR_OPENMP) $(CXXLINK) -o $@

$(APP_NAME)_mpi_openmp: constants.F90 $(GEN_KERNELS) $(APP_NAME)_seqfun.F90 input.F90 $(APP_NAME)_op.F90 | mod/mpi_openmp
	$(MPIFC) $(FFLAGS) $(OMP_FFLAGS) $(F_MOD_OUT_OPT)$| $(OP2_MOD) $^ $(OP2_LIB_FOR_MPI) $(CXXLINK) -o $@

$(APP_NAME)_openmp4: constants.F90 $(GEN_KERNELS_OMP4) $(APP_NAME)_seqfun.F90 input.F90 $(APP_NAME)_op.F90 | mod/openmp4
	$(FC) $(FFLAGS) $(OMP_OFFLOAD_FFLAGS) $(F_MOD_OUT_OPT)$| $(OP2_MOD) $^ $(OP2_LIB_FOR_OPENMP) $(CXXLINK) -o $@

$(APP_NAME)_mpi_openmp4: constants.F90 $(GEN_KERNELS_OMP4) $(APP_NAME)_seqfun.F90 input.F90 $(APP_NAME)_op.F90 | mod/mpi_openmp4
	$(MPIFC) $(FFLAGS) $(OMP_OFFLOAD_FFLAGS) $(F_MOD_OUT_OPT)$| $(OP2_MOD) $^ $(OP2_LIB_FOR_MPI) $(CXXLINK) -o $@

# $(APP_NAME)_cuda: constants.F90 $(GEN_KERNELS_CUDA) $(APP_NAME)_seqfun.F90 input.F90 $(APP_NAME)_op.F90 | mod/cuda
# 	$(FC) $(FFLAGS) $(CUDA_FFLAGS) $(F_MOD_OUT_OPT)$| $(OP2_MOD) $^ $(OP2_LIB_FOR_CUDA) $(CXXLINK) -o $@
