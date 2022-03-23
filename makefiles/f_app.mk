TRANSLATOR ?= python3 $(ROOT_DIR)/translator/op2-translator -v

ifneq ($(F_HAS_PARALLEL_BUILDS),true)
  .NOTPARALLEL:
endif

PART_SIZE_ENV ?= 128
FFLAGS += -DOP_PART_SIZE_1=$(PART_SIZE_ENV)

APP_ENTRY ?= $(APP_NAME).F90
APP_ENTRY_BASENAME := $(basename $(APP_ENTRY))
APP_ENTRY_OP := $(APP_ENTRY_BASENAME)_op.F90

BASE_VARIANTS := seq genseq vec openmp cuda

ALL_VARIANTS := $(BASE_VARIANTS)
ALL_VARIANTS += $(foreach variant,$(ALL_VARIANTS),mpi_$(variant))
ALL_VARIANTS := $(foreach variant,$(ALL_VARIANTS),$(APP_NAME)_$(variant))

ifeq ($(HAVE_F),true)
  BUILDABLE_VARIANTS := seq genseq

  ifeq ($(F_HAS_OMP),true)
    BUILDABLE_VARIANTS += # vec openmp
  endif

  # TODO/openmp4 add omp declare target
  # ifeq ($(F_HAS_OMP_OFFLOAD),true)
  #   BUILDABLE_VARIANTS += openmp4
  # endif

  ifeq ($(F_HAS_CUDA),true)
    BUILDABLE_VARIANTS += cuda
  endif

  ifeq ($(HAVE_MPI_F),true)
    BUILDABLE_VARIANTS += $(foreach variant,$(BUILDABLE_VARIANTS),mpi_$(variant))
  endif
endif

BUILDABLE_VARIANTS := $(foreach variant,$(BUILDABLE_VARIANTS),$(APP_NAME)_$(variant))

VARIANT_FILTER ?= %
VARIANT_FILTER_OUT ?=

BUILDABLE_VARIANTS := $(filter-out $(VARIANT_FILTER_OUT),\
                      $(filter $(VARIANT_FILTER),$(BUILDABLE_VARIANTS)))

ifeq ($(OP2_LIBS_WITH_HDF5),true)
  ifneq ($(HAVE_HDF5_SEQ),true)
    BUILDABLE_VARIANTS := $(filter $(APP_NAME)_mpi_%,$(BUILDABLE_VARIANTS))
  endif

  ifneq ($(HAVE_HDF5_PAR),true)
    BUILDABLE_VARIANTS := $(filter-out $(APP_NAME)_mpi_%,$(BUILDABLE_VARIANTS))
  endif
endif

ifneq ($(MAKECMDGOALS),clean)
  $(info Buildable app variants: $(BUILDABLE_VARIANTS))
  $(info )
endif

KERNELS = $(patsubst %.inc,%,$(wildcard *.inc))
KERNEL_SOURCES = $(wildcard *.inc) $(wildcard *.inc2)

GEN_KERNELS_GENSEQ = $(foreach kernel,$(KERNELS),$(kernel)_seq_kernel.F95)
GEN_KERNELS_VEC = $(foreach kernel,$(KERNELS),$(kernel)_vec_kernel.F95)
GEN_KERNELS_OPENMP = $(foreach kernel,$(KERNELS),$(kernel)_openmp_kernel.F95)
GEN_KERNELS_OPENMP4 = $(foreach kernel,$(KERNELS),$(kernel)_openmp4_kernel.F95)
GEN_KERNELS_CUDA = $(foreach kernel,$(KERNELS),$(kernel)_cuda_kernel.CUF)

GENERATED = \
	$(APP_ENTRY_OP) \
	$(GEN_KERNELS_GENSEQ) \
	$(GEN_KERNELS_VEC) \
	$(GEN_KERNELS_OPENMP) \
	$(GEN_KERNELS_OPENMP4) \
	$(GEN_KERNELS_CUDA)

.PHONY: all clean

all: $(BUILDABLE_VARIANTS)

clean:
	-$(RM) $(ALL_VARIANTS)
	-$(RM) $(GENERATED)
	-$(RM) .generated
	-$(RM) *.o
	-$(RM) -r mod

.generated: $(APP_ENTRY)
	$(TRANSLATOR) $<
	@touch $@

mod/%:
	@mkdir -p $@

SEQ_SRC := constants.F90 $(APP_ENTRY_BASENAME)_seqfun.F90 input.F90 $(APP_ENTRY)

# $(1) = variant name
define SRC_template =
$(1)_SRC := constants.F90 $$(GEN_KERNELS_$(1)) $(APP_ENTRY_BASENAME)_seqfun.F90 input.F90 $(APP_ENTRY_OP)
endef

$(foreach variant,$(filter-out seq,$(BASE_VARIANTS)),\
	$(eval $(call SRC_template,$(call UPPERCASE,$(variant)))))

# $(1) = variant name
# $(2) = additional flags
# $(3) = OP2 library for sequential variant
# $(4) = OP2 library for parallel variant
# $(5) = extra module dependencies
define RULE_template_base =
$$(APP_NAME)_$(1): .generated | mod/$(1)
	$$(FC) $$(FFLAGS) $(2) $$(F_MOD_OUT_OPT)$$| $(5) $$(OP2_MOD) \
		$$($(call UPPERCASE,$(1))_SRC) $$(OP2_LIB_FOR_$(3)) $$(CXXLINK) -o $$@

$$(APP_NAME)_mpi_$(1): .generated | mod/mpi_$(1)
	$$(MPIFC) $$(FFLAGS) $(2) $$(F_MOD_OUT_OPT)$$| $(5) $$(OP2_MOD) \
		$$($(call UPPERCASE,$(1))_SRC) $$(OP2_LIB_FOR_$(4)) $$(CXXLINK) -o $$@

endef

# the same as RULE_template_base but it first strips its arguments of extra space
define RULE_template =
$(call RULE_template_base,$(strip $(1)),$(strip $(2)),$(strip $(3)),$(strip $(4)),$(strip $(5)))
endef

$(eval $(call RULE_template, seq,,                                           SEQ,     MPI,))
$(eval $(call RULE_template, genseq,,                                        SEQ,     MPI,))
$(eval $(call RULE_template, vec,     $(OMP_FFLAGS) -DVECTORIZE,             SEQ,     MPI,))
$(eval $(call RULE_template, openmp,  $(OMP_FFLAGS),                         OPENMP,  MPI,))
$(eval $(call RULE_template, openmp4, $(OMP_OFFLOAD_FFLAGS) -DOP2_WITH_OMP4, OPENMP4,    ,))
$(eval $(call RULE_template, cuda,    $(CUDA_FFLAGS),                        CUDA,    MPI_CUDA, $(OP2_MOD_CUDA)))
