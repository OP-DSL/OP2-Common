TRANSLATOR ?= $(ROOT_DIR)/translator-v2/op2-translator.sh -v

ifneq ($(F_HAS_PARALLEL_BUILDS),true)
  .NOTPARALLEL:
endif

PART_SIZE_ENV ?= 128
FFLAGS += -DOP_PART_SIZE_1=$(PART_SIZE_ENV)

APP_SRC_OP := $(APP_SRC:%.F90=generated/$(APP_NAME)/%.F90)

BASE_VARIANTS := seq genseq openmp cuda

ALL_VARIANTS := $(BASE_VARIANTS)
ALL_VARIANTS += $(foreach variant,$(ALL_VARIANTS),mpi_$(variant))

BUILDABLE_VARIANTS :=

ifeq ($(HAVE_F),true)
  BUILDABLE_VARIANTS += seq genseq

  ifeq ($(F_HAS_OMP),true)
    BUILDABLE_VARIANTS += openmp
  endif

  ifeq ($(F_HAS_CUDA),true)
    BUILDABLE_VARIANTS += cuda
  endif

  ifeq ($(HAVE_MPI_F),true)
    BUILDABLE_VARIANTS += $(foreach variant,$(BUILDABLE_VARIANTS),mpi_$(variant))
  endif
endif

VARIANT_FILTER ?= %
VARIANT_FILTER_OUT ?=

BUILDABLE_VARIANTS := $(filter-out $(VARIANT_FILTER_OUT),\
                      $(filter $(VARIANT_FILTER),$(BUILDABLE_VARIANTS)))

ifeq ($(OP2_LIBS_WITH_HDF5),true)
  ifneq ($(HAVE_HDF5_SEQ),true)
    BUILDABLE_VARIANTS := $(filter mpi_%,$(BUILDABLE_VARIANTS))
  endif

  ifneq ($(HAVE_HDF5_PAR),true)
    BUILDABLE_VARIANTS := $(filter-out mpi_%,$(BUILDABLE_VARIANTS))
  endif
endif

ifneq ($(MAKECMDGOALS),clean)
  $(info Buildable app variants for $(APP_NAME): $(BUILDABLE_VARIANTS))
  $(info )
endif

.PHONY: all clean generate

define ALL_template =
all: $(foreach variant,$(BUILDABLE_VARIANTS),$(APP_NAME)_$(variant))
endef

$(eval $(call ALL_template))

# Only define the clean rule on first include of this makefile
ifeq ($(words $(filter %/f_app.mk,$(MAKEFILE_LIST))),1)
clean:
	-$(RM) $(foreach variant,$(ALL_VARIANTS),*_$(variant))
	-$(RM) -rf generated
	-$(RM) *.o
	-$(RM) -r mod
endif

define GENERATED_template =
generated/$(APP_NAME): $(APP_SRC)
	@mkdir -p $$@
	$(TRANSLATOR) $(APP_EXTRA_FLAGS) $(APP_EXTRA_TRANSLATOR_FLAGS) $$^ -o $$@

generate: generated/$(APP_NAME)
endef

$(eval $(call GENERATED_template))

mod/%:
	@mkdir -p $@

# $(1) = variant name
define SRC_template =
$(call UPPERCASE,$(1))_SRC := generated/$(APP_NAME)/$(2)/op2_consts.F90 \
                              generated/$(APP_NAME)/$(2)/*_kernel.$(3) \
                              generated/$(APP_NAME)/$(2)/op2_kernels.F90 \
                              $(APP_SRC_OP)
endef

SEQ_SRC := $(APP_SRC)

$(eval $(call SRC_template,genseq,seq,F90))
$(eval $(call SRC_template,openmp,openmp,F90))
$(eval $(call SRC_template,cuda,cuda,F90))

include $(MAKEFILES_DIR)/lib_helpers.mk

# $(1) = variant name
# $(2) = additional flags
# $(3) = OP2 library for sequential variant
# $(4) = OP2 library for parallel variant
# $(5) = extra module dependencies
define RULE_template_base =
$(APP_NAME)_$(1): $(if $(filter-out seq,$(1)),generated/$(APP_NAME)) | mod/$(APP_NAME)/$(1)
	$$(FC) $$(FFLAGS) $(2) $(APP_EXTRA_FLAGS) $$(F_MOD_OUT_OPT)$$| $(5) $$(OP2_MOD) \
		$($(call UPPERCASE,$(1))_SRC) $(OP2_LIB_FOR_$(3)) $$(CXXLINK) -o $$@

$(APP_NAME)_mpi_$(1): $(if $(filter-out seq,$(1)),generated/$(APP_NAME)) | mod/$(APP_NAME)/mpi_$(1)
	$$(MPIFC) $$(FFLAGS) $(2) $(APP_EXTRA_FLAGS) $$(F_MOD_OUT_OPT)$$| $(5) $$(OP2_MOD) \
		$($(call UPPERCASE,$(1))_SRC) $(OP2_LIB_FOR_$(4)) $$(CXXLINK) -o $$@

endef

# the same as RULE_template_base but it first strips its arguments of extra space
define RULE_template =
$(call RULE_template_base,$(strip $(1)),$(strip $(2)),$(strip $(3)),$(strip $(4)),$(strip $(5)))
endef

$(eval $(call RULE_template, seq,,                    SEQ,     MPI,))
$(eval $(call RULE_template, genseq,,                 SEQ,     MPI,))
$(eval $(call RULE_template, openmp,  $(OMP_FFLAGS),  OPENMP,  MPI,))
$(eval $(call RULE_template, cuda,    $(CUDA_FFLAGS), CUDA,    MPI_CUDA, $(OP2_MOD_CUDA)))

# Reset optional input variables for following includes of this Makefile
VARIANT_FILTER := %
VARIANT_FILTER_OUT :=
