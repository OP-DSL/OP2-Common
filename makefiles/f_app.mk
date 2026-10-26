TRANSLATOR ?= $(ROOT_DIR)/translator-v2/op2-translator.sh -v

ifneq ($(F_HAS_PARALLEL_BUILDS),true)
  .NOTPARALLEL:
endif

PART_SIZE_ENV ?= 128
FFLAGS += -DOP_PART_SIZE_1=$(PART_SIZE_ENV)

APP_SRC_OP := $(APP_SRC:%.F90=generated/$(APP_NAME)/%.F90)

BASE_VARIANTS := seq genseq openmp cuda_legacy seq_f2c cuda_f2c hip_f2c

ALL_VARIANTS := $(BASE_VARIANTS)
ALL_VARIANTS += $(foreach variant,$(ALL_VARIANTS),mpi_$(variant))

BUILDABLE_VARIANTS :=

ifeq ($(HAVE_F),true)
  BUILDABLE_VARIANTS += seq genseq

  ifeq ($(F_HAS_OMP),true)
    BUILDABLE_VARIANTS += openmp
  endif

  ifeq ($(F_HAS_CUDA),true)
    BUILDABLE_VARIANTS += cuda_legacy
  endif

  ifeq ($(HAVE_C),true)
    BUILDABLE_VARIANTS += seq_f2c
  endif

  ifeq ($(HAVE_CUDA),true)
    BUILDABLE_VARIANTS += cuda_f2c
  endif

  ifeq ($(HAVE_HIP),true)
    BUILDABLE_VARIANTS += hip_f2c
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
                              generated/$(APP_NAME)/$(2)/op2_kernels.F90 \
                              $(APP_SRC_OP)
endef
# Note: per-loop `<loop>_kernel.F90` files are #include'd by op2_kernels.F90 at
# preprocessing time; they must NOT be listed here or the module definitions
# get compiled twice (once standalone, once inside op2_kernels) which produces
# duplicate-symbol link errors.  The $(3) argument is retained on the callers
# below for backwards signature compatibility but is unused.

SEQ_SRC := $(APP_SRC)

$(eval $(call SRC_template,genseq,seq,F90))
$(eval $(call SRC_template,openmp,openmp,F90))
$(eval $(call SRC_template,cuda_legacy,cuda_legacy,F90))
$(eval $(call SRC_template,seq_f2c,seq_f2c,F90))
$(eval $(call SRC_template,cuda_f2c,cuda_f2c,F90))
$(eval $(call SRC_template,hip_f2c,hip_f2c,F90))

SEQ_F2C_SRC += generated/$(APP_NAME)/seq_f2c/op2_kernels.o
CUDA_F2C_SRC += generated/$(APP_NAME)/cuda_f2c/op2_kernels.o
HIP_F2C_SRC += generated/$(APP_NAME)/hip_f2c/op2_kernels.o

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

define RULE_template_cuda_f2c_base =
$(APP_NAME)_$(1): $(if $(filter-out seq,$(1)),generated/$(APP_NAME)) | mod/$(APP_NAME)/$(1)
	$$(FC) $$(FFLAGS) $(2) $(APP_EXTRA_FLAGS) $$(F_MOD_OUT_OPT)$$| $(5) $$(OP2_MOD) \
		$($(call UPPERCASE,$(1))_SRC) $(OP2_LIB_FOR_$(3)) $$(CXXLINK) $(CUDA_LIB) -o $$@

$(APP_NAME)_mpi_$(1): $(if $(filter-out seq,$(1)),generated/$(APP_NAME)) | mod/$(APP_NAME)/mpi_$(1)
	$$(MPIFC) $$(FFLAGS) $(2) $(APP_EXTRA_FLAGS) $$(F_MOD_OUT_OPT)$$| $(5) $$(OP2_MOD) \
		$($(call UPPERCASE,$(1))_SRC) $(OP2_LIB_FOR_$(4)) $$(CXXLINK) $(CUDA_LIB) -o $$@
endef

define RULE_template_cuda_f2c =
$(call RULE_template_cuda_f2c_base,$(strip $(1)),$(strip $(2)),$(strip $(3)),$(strip $(4)),$(strip $(5)))
endef

define RULE_template_hip_f2c_base =
$(APP_NAME)_$(1): $(if $(filter-out seq,$(1)),generated/$(APP_NAME)) | mod/$(APP_NAME)/$(1)
	$$(FC) $$(FFLAGS) $(2) $(APP_EXTRA_FLAGS) $$(F_MOD_OUT_OPT)$$| $(5) $$(OP2_MOD) \
		$($(call UPPERCASE,$(1))_SRC) $(OP2_LIB_FOR_$(3)) $$(CXXLINK) $(HIP_LIB) -o $$@

$(APP_NAME)_mpi_$(1): $(if $(filter-out seq,$(1)),generated/$(APP_NAME)) | mod/$(APP_NAME)/mpi_$(1)
	$$(MPIFC) $$(FFLAGS) $(2) $(APP_EXTRA_FLAGS) $$(F_MOD_OUT_OPT)$$| $(5) $$(OP2_MOD) \
		$($(call UPPERCASE,$(1))_SRC) $(OP2_LIB_FOR_$(4)) $$(CXXLINK) $(HIP_LIB) -o $$@
endef

define RULE_template_hip_f2c =
$(call RULE_template_hip_f2c_base,$(strip $(1)),$(strip $(2)),$(strip $(3)),$(strip $(4)),$(strip $(5)))
endef

# Standard variants
$(eval $(call RULE_template,        seq,,                   SEQ,     MPI,))
$(eval $(call RULE_template,        genseq,,                SEQ,     MPI,))
$(eval $(call RULE_template,        openmp, $(OMP_FFLAGS),  OPENMP,  MPI,))
$(eval $(call RULE_template,        cuda_legacy, $(CUDA_FFLAGS), CUDA, MPI_CUDA, $(OP2_MOD_CUDA)))
$(eval $(call RULE_template,        seq_f2c,,                 SEQ,     MPI,))
$(eval $(call RULE_template_cuda_f2c, cuda_f2c, $(CUDA_FFLAGS), CUDA,    MPI_CUDA,))
$(eval $(call RULE_template_hip_f2c,  hip_f2c,  $(HIP_FFLAGS),  HIP,     MPI_HIP,))

# C++ build for seq_f2c
define SEQ_F2C_EXTRA_RULES_template =
$(APP_NAME)_seq_f2c: generated/$(APP_NAME)/seq_f2c/op2_kernels.o
$(APP_NAME)_mpi_seq_f2c: generated/$(APP_NAME)/seq_f2c/op2_kernels.o

generated/$(APP_NAME)/seq_f2c/op2_kernels.o: generated/$(APP_NAME)
	$$(CXX) $$(CXXFLAGS) $$(OP2_INC) $(APP_EXTRA_FLAGS) \
	    -c generated/$(APP_NAME)/seq_f2c/op2_kernels_aux1.cpp -o $$@
endef

$(eval $(call SEQ_F2C_EXTRA_RULES_template))

# NVCC build for cuda_f2c
define CUDA_F2C_EXTRA_RULES_template =
$(APP_NAME)_cuda_f2c: generated/$(APP_NAME)/cuda_f2c/op2_kernels.o
$(APP_NAME)_mpi_cuda_f2c: generated/$(APP_NAME)/cuda_f2c/op2_kernels.o

generated/$(APP_NAME)/cuda_f2c/op2_kernels.o: generated/$(APP_NAME)
	$$(NVCC) $$(NVCCFLAGS) $$(OP2_INC) $(APP_EXTRA_FLAGS) -DOP2_CUDA \
	    -c generated/$(APP_NAME)/cuda_f2c/op2_kernels_aux1.cu -o $$@
endef

$(eval $(call CUDA_F2C_EXTRA_RULES_template))

# HIPCC build for hip_f2c
define HIP_F2C_EXTRA_RULES_template =
$(APP_NAME)_hip_f2c: generated/$(APP_NAME)/hip_f2c/op2_kernels.o
$(APP_NAME)_mpi_hip_f2c: generated/$(APP_NAME)/hip_f2c/op2_kernels.o

generated/$(APP_NAME)/hip_f2c/op2_kernels.o: generated/$(APP_NAME)
	$$(HIPCC) $$(HIPCCFLAGS) $$(OP2_INC) $(APP_EXTRA_FLAGS) -DOP2_HIP \
	    -c generated/$(APP_NAME)/hip_f2c/op2_kernels_aux1.hip.cpp -o $$@
endef

$(eval $(call HIP_F2C_EXTRA_RULES_template))

# Reset optional input variables for following includes of this Makefile
VARIANT_FILTER := %
VARIANT_FILTER_OUT :=
