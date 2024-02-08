TRANSLATOR ?= python3 $(ROOT_DIR)/translator/op2-translator/__main__.py -v

ifneq ($(MPI_INC),)
	TRANSLATOR += -I $(MPI_INC)
endif

APP_SRC_OP := $(foreach src,$(APP_SRC),$(notdir $(src)))
APP_SRC_OP := $(APP_SRC_OP:%.cpp=generated/$(APP_NAME)/%.cpp)

APP_INC ?= -I.

APP_BIN_DIR ?= .

BASE_VARIANTS := seq genseq openmp cuda

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

.PHONY: all clean

define ALL_template =
all: $(foreach variant,$(BUILDABLE_VARIANTS),$(APP_NAME)_$(variant))
endef

$(eval $(call ALL_template))

# Only define the clean rule on first include of this makefile
ifeq ($(words $(filter %/c_app.mk,$(MAKEFILE_LIST))),1)
clean:
	-$(RM) $(addprefix $(APP_BIN_DIR)/,$(foreach variant,$(ALL_VARIANTS),*_$(variant)))
	-$(RM) -rf generated
	-$(RM) $(addprefix $(APP_BIN_DIR)/, *.d)  *.d
	-$(RM) $(addprefix $(APP_BIN_DIR)/, *.o)  *.o
	-$(RM) $(addprefix $(APP_BIN_DIR)/, out_grid.*) out_grid.*
	-$(RM) $(addprefix $(APP_BIN_DIR)/, out_grid_mpi.*) out_grid_mpi.*
endif

define GENERATED_template =
generated/$(APP_NAME): $(APP_SRC)
	@mkdir -p $$@
	$(TRANSLATOR) $(APP_INC) $$^ -o $$@
endef

$(eval $(call GENERATED_template))

# $(1) = variant name
# $(2) = folder name
# $(3) = extension
define SRC_template =
$(call UPPERCASE,$(1))_SRC := $(APP_SRC_OP) generated/$(APP_NAME)/$(2)/op2_kernels.$(3)
endef

SEQ_SRC := $(APP_SRC)

$(eval $(call SRC_template,genseq,seq,cpp))
$(eval $(call SRC_template,openmp,openmp,cpp))
$(eval $(call SRC_template,cuda,cuda,o))

include $(MAKEFILES_DIR)/lib_helpers.mk

# $(1) = variant name
# $(2) = additional flags
# $(3) = OP2 library for sequential variant
# $(4) = OP2 library for parallel variant
define RULE_template_base =
$(APP_NAME)_$(1): $(if $(filter-out seq,$(1)),generated/$(APP_NAME))
	$$(CXX) $$(CXXFLAGS) $(2) $(APP_INC) $$(OP2_INC) $($(call UPPERCASE,$(1))_SRC) $(OP2_LIB_$(3)) -o $(APP_BIN_DIR)/$$@

$(APP_NAME)_mpi_$(1): $(if $(filter-out seq,$(1)),generated/$(APP_NAME))
	$$(MPICXX) $$(CXXFLAGS) $(2) $(APP_INC) $$(OP2_INC) $($(call UPPERCASE,$(1))_SRC) $(OP2_LIB_$(4)) -o $(APP_BIN_DIR)/$$@
endef

# the same as RULE_template_base but it first strips its arguments of extra space
define RULE_template =
$(call RULE_template_base,$(strip $(1)),$(strip $(2)),$(strip $(3)),$(strip $(4)))
endef

# Check if APP_BIN_DIR is different from the current directory
ifneq ($(APP_BIN_DIR),.)
  # Create the bin directory if it does not exist
  $(shell mkdir -p $(APP_BIN_DIR))
endif

$(eval $(call RULE_template, seq,,                      SEQ,     MPI))
$(eval $(call RULE_template, genseq,,                   SEQ,     MPI))
$(eval $(call RULE_template, openmp,   $(OMP_CXXFLAGS), OPENMP,  MPI))
$(eval $(call RULE_template, cuda,,                     CUDA,    MPI_CUDA))

define CUDA_EXTRA_RULES_template =
$(APP_NAME)_cuda: generated/$(APP_NAME)/cuda/op2_kernels.o
$(APP_NAME)_mpi_cuda: generated/$(APP_NAME)/cuda/op2_kernels.o

generated/$(APP_NAME)/cuda/op2_kernels.o: generated/$(APP_NAME)
	$$(NVCC) $$(NVCCFLAGS) $(APP_INC) $$(OP2_INC) -c generated/$(APP_NAME)/cuda/op2_kernels.cu -o $$@
endef

$(eval $(call CUDA_EXTRA_RULES_template))

-include $(wildcard *.d)

# Reset optional input variables for following includes of this Makefile
APP_INC := -I.

VARIANT_FILTER := %
VARIANT_FILTER_OUT :=
