TRANSLATOR ?= $(ROOT_DIR)/translator-v2/op2-translator.sh -v

ifneq ($(MPI_INC),)
	TRANSLATOR += -I $(MPI_INC)
endif

APP_SRC_OP := $(foreach src,$(APP_SRC),$(notdir $(src)))
APP_SRC_OP := $(APP_SRC_OP:%.cpp=generated/$(APP_NAME)/%.cpp)

APP_INC ?= -I.

BASE_VARIANTS := seq genseq openmp cuda hip

ALL_VARIANTS := $(BASE_VARIANTS)
ALL_VARIANTS += $(foreach variant,$(ALL_VARIANTS),mpi_$(variant))

BUILDABLE_VARIANTS :=

ifeq ($(HAVE_C),true)
  BUILDABLE_VARIANTS += seq genseq

  ifeq ($(CPP_HAS_OMP),true)
    BUILDABLE_VARIANTS += openmp
  endif

  ifeq ($(HAVE_CUDA),true)
    BUILDABLE_VARIANTS += cuda
  endif

  ifeq ($(HAVE_HIP),true)
    BUILDABLE_VARIANTS += hip
  endif

  ifeq ($(HAVE_MPI_C),true)
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
ifeq ($(words $(filter %/c_app.mk,$(MAKEFILE_LIST))),1)
clean:
	-$(RM) $(foreach variant,$(ALL_VARIANTS),*_$(variant))
	-$(RM) -rf generated
	-$(RM) *.d
	-$(RM) *.o
	-$(RM) out_grid.*
	-$(RM) out_grid_mpi.*
endif

define GENERATED_template =
generated/$(APP_NAME): $(APP_SRC)
	@mkdir -p $$@
	$(TRANSLATOR) $(APP_INC) $$^ -o $$@

generate: generated/$(APP_NAME)
endef

$(eval $(call GENERATED_template))

SEQ_SRC := $(APP_SRC)

GENSEQ_SRC := $(APP_SRC_OP) generated/$(APP_NAME)/seq/op2_kernels.cpp
OPENMP_SRC := $(APP_SRC_OP) generated/$(APP_NAME)/openmp/op2_kernels.cpp
CUDA_SRC   := $(APP_SRC_OP) generated/$(APP_NAME)/cuda/op2_kernels.o
HIP_SRC    := $(APP_SRC_OP) generated/$(APP_NAME)/hip/op2_kernels.o

include $(MAKEFILES_DIR)/lib_helpers.mk

# $(1) = variant name
# $(2) = additional flags
# $(3) = OP2 library for sequential variant
# $(4) = OP2 library for parallel variant
define RULE_template_base =
$(APP_NAME)_$(1): $(if $(filter-out seq,$(1)),generated/$(APP_NAME))
	$$(CXX) $$(CXXFLAGS) $(2) $(APP_INC) $$(OP2_INC) $($(call UPPERCASE,$(1))_SRC) $(OP2_LIB_$(3)) -o $$@

$(APP_NAME)_mpi_$(1): $(if $(filter-out seq,$(1)),generated/$(APP_NAME))
	$$(MPICXX) $$(CXXFLAGS) $(2) $(APP_INC) $$(OP2_INC) $($(call UPPERCASE,$(1))_SRC) $(OP2_LIB_$(4)) -o $$@
endef

# the same as RULE_template_base but it first strips its arguments of extra space
define RULE_template =
$(call RULE_template_base,$(strip $(1)),$(strip $(2)),$(strip $(3)),$(strip $(4)))
endef

$(eval $(call RULE_template, seq,,                      SEQ,     MPI))
$(eval $(call RULE_template, genseq,,                   SEQ,     MPI))
$(eval $(call RULE_template, openmp,   $(OMP_CXXFLAGS), OPENMP,  MPI))
$(eval $(call RULE_template, cuda,,                     CUDA,    MPI_CUDA))
$(eval $(call RULE_template, hip,,                      HIP,     MPI_HIP))

define CUDA_EXTRA_RULES_template =
$(APP_NAME)_cuda: generated/$(APP_NAME)/cuda/op2_kernels.o
$(APP_NAME)_mpi_cuda: generated/$(APP_NAME)/cuda/op2_kernels.o

generated/$(APP_NAME)/cuda/op2_kernels.o: generated/$(APP_NAME)
	$$(NVCC) $$(NVCCFLAGS) $(APP_INC) $$(OP2_INC) -DOP2_CUDA -c generated/$(APP_NAME)/cuda/op2_kernels.cu -o $$@
endef

$(eval $(call CUDA_EXTRA_RULES_template))

define HIP_EXTRA_RULES_template =
$(APP_NAME)_hip: generated/$(APP_NAME)/hip/op2_kernels.o
$(APP_NAME)_mpi_hip: generated/$(APP_NAME)/hip/op2_kernels.o

generated/$(APP_NAME)/hip/op2_kernels.o: generated/$(APP_NAME)
	$$(HIPCC) $$(HIPCCFLAGS) $(APP_INC) $$(OP2_INC) -DOP2_HIP -c generated/$(APP_NAME)/hip/op2_kernels.hip.cpp -o $$@
endef

$(eval $(call HIP_EXTRA_RULES_template))

-include $(wildcard *.d)

# Reset optional input variables for following includes of this Makefile
APP_INC := -I.

VARIANT_FILTER := %
VARIANT_FILTER_OUT :=
