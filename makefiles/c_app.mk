TRANSLATOR ?= $(ROOT_DIR)/translator/c/op2.py

APP_ENTRY ?= $(APP_NAME).cpp
APP_ENTRY_MPI ?= $(APP_NAME)_mpi.cpp
APP_ENTRY_GPI ?= $(APP_ENTRY_MPI)

APP_ENTRY_BASENAME := $(basename $(APP_ENTRY))
APP_ENTRY_MPI_BASENAME := $(basename $(APP_ENTRY_MPI))

APP_ENTRY_OP := $(APP_ENTRY_BASENAME)_op.cpp
APP_ENTRY_MPI_OP := $(APP_ENTRY_MPI_BASENAME)_op.cpp

# This is used for clean
ALL_VARIANTS_BASE := seq genseq vec openmp openmp4 cuda cuda_hyb
ALL_VARIANTS := $(ALL_VARIANTS_BASE)
ALL_VARIANTS += $(foreach variant,$(ALL_VARIANTS_BASE),mpi_$(variant))
ALL_VARIANTS += $(foreach variant,$(ALL_VARIANTS_BASE),gpi_$(variant))
ALL_VARIANTS := $(foreach variant,$(ALL_VARIANTS),$(APP_NAME)_$(variant))

ifeq ($(HAVE_C),true)
  BASE_BUILDABLE_VARIANTS := seq genseq

  ifeq ($(CPP_HAS_OMP),true)
    BASE_BUILDABLE_VARIANTS += vec openmp
  endif

  ifeq ($(CPP_HAS_OMP_OFFLOAD),true)
    BASE_BUILDABLE_VARIANTS += openmp4
  endif

  ifeq ($(HAVE_CUDA),true)
    BASE_BUILDABLE_VARIANTS += cuda cuda_hyb
  endif

endif

BUILDABLE_VARIANTS :=
ifneq ($(wildcard ./$(APP_ENTRY)),)
  BUILDABLE_VARIANTS += $(foreach variant,$(BASE_BUILDABLE_VARIANTS),$(APP_NAME)_$(variant))
endif

ifneq ($(and $(wildcard ./$(APP_ENTRY_MPI)),$(HAVE_MPI_C)),)
  BUILDABLE_VARIANTS += $(foreach variant,$(BASE_BUILDABLE_VARIANTS),$(APP_NAME)_mpi_$(variant))

  # TODO/openmp4 MPI + OpenMP4 offload build not (yet) supported
  BUILDABLE_VARIANTS := $(filter-out %_mpi_openmp4,$(BUILDABLE_VARIANTS))
endif

# The GPI variant, but only built with the very basic SEQ and GENSEQ versions
# The code below excludes all the extra variants that are not supported (yet)
ifneq ($(and $(wildcard ./$(APP_ENTRY_GPI)),$(HAVE_GPI)),)
  BUILDABLE_VARIANTS += $(foreach variant,$(BASE_BUILDABLE_VARIANTS),$(APP_NAME)_gpi_$(variant))

  # Excluding the unsupported ones
  BUILDABLE_VARIANTS :=  $(filter-out %_gpi_vec,$(BUILDABLE_VARIANTS))
  BUILDABLE_VARIANTS :=  $(filter-out %_gpi_openmp,$(BUILDABLE_VARIANTS))
  BUILDABLE_VARIANTS :=  $(filter-out %_gpi_openmp4,$(BUILDABLE_VARIANTS))
  BUILDABLE_VARIANTS :=  $(filter-out %_gpi_cuda,$(BUILDABLE_VARIANTS))
  BUILDABLE_VARIANTS :=  $(filter-out %_gpi_cuda_hyb,$(BUILDABLE_VARIANTS))
endif

$(info Buildable app variants before filtering: $(BUILDABLE_VARIANTS))

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

.PHONY: all clean

all: $(BUILDABLE_VARIANTS)

clean:
	-$(RM) $(ALL_VARIANTS)
	-$(RM) -r seq vec openmp openmp4 cuda openacc
	-$(RM) *_op.cpp
	-$(RM) .generated .generated
	-$(RM) *.d
	-$(RM) *.o
	-$(RM) out_grid.*
	-$(RM) out_grid_mpi.*

.generated: $(APP_ENTRY) $(APP_ENTRY_MPI)
	[ ! -f $(APP_ENTRY) ] || $(TRANSLATOR) $(APP_ENTRY)
	[ ! -f $(APP_ENTRY_MPI) ] || [ $(APP_ENTRY_MPI) = $(APP_ENTRY) ] \
		|| $(TRANSLATOR) $(APP_ENTRY_MPI)
	@touch $@

SEQ_SRC := $(APP_ENTRY)
MPI_SEQ_SRC := $(APP_ENTRY_MPI)

define SRC_template =
$(1)_SRC := $$(APP_ENTRY_OP) $$(subst %,$$(APP_ENTRY_BASENAME),$(2))
MPI_$(1)_SRC := $$(APP_ENTRY_MPI_OP) $$(subst %,$$(APP_ENTRY_MPI_BASENAME),$(2))
endef

$(eval $(call SRC_template,GENSEQ,seq/%_seqkernels.cpp))
$(eval $(call SRC_template,VEC,vec/%_veckernels.cpp))
$(eval $(call SRC_template,OPENMP,openmp/%_kernels.cpp))
$(eval $(call SRC_template,OPENMP4,openmp4/%_omp4kernels.cpp))

#  TODO/openmp4 perhaps include this in _omp4kernels.cpp?
OPENMP4_SRC += openmp4/$(APP_ENTRY_BASENAME)_omp4kernel_funcs.cpp
MPI_OPENMP4_SRC += openmp4/$(APP_ENTRY_BASENAME)_omp4kernel_funcs.cpp

CUDA_SRC := $(APP_ENTRY_OP) cuda/$(APP_NAME)_kernels.o
MPI_CUDA_SRC := $(APP_ENTRY_MPI_OP) cuda/$(APP_NAME)_mpi_kernels.o

CUDA_HYB_SRC := $(APP_ENTRY_OP) \
	cuda/$(APP_NAME)_hybkernels_cpu.o cuda/$(APP_NAME)_hybkernels_gpu.o

MPI_CUDA_HYB_SRC := $(APP_ENTRY_MPI_OP) \
	cuda/$(APP_NAME)_mpi_hybkernels_cpu.o cuda/$(APP_NAME)_mpi_hybkernels_gpu.o


# $(1) = variant name
# $(2) = additional flags
# $(3) = OP2 library for sequential variant
# $(4) = OP2 library for parallel variant
define RULE_template_base =
$$(APP_NAME)_$(1): .generated
	$$(CXX) $$(CXXFLAGS) $(2) $$(OP2_INC) $$($(call UPPERCASE,$(1))_SRC) $$(OP2_LIB_$(3)) -o $$@

$$(APP_NAME)_mpi_$(1): .generated
	$$(MPICXX) $$(CXXFLAGS) $(2) $$(OP2_INC) $$(MPI_$(call UPPERCASE,$(1))_SRC) $$(OP2_LIB_$(4)) -o $$@

$$(APP_NAME)_gpi_$(1): .generated
	$$(MPICXX) $$(CXXFLAGS) $(2) $$(OP2_INC) $$(GPI_INC) $$(MPI_$(call UPPERCASE,$(1))_SRC) -L/dcs/19/u1906509/Documents/HPC/PGAS-OP2-Common/op2/lib -lop2_gpi  -L/usr/local//lib -lparmetis -lmetis  $$(GPI_LIB)  -o $$@
endef

# the same as RULE_template_base but it first strips its arguments of extra space
define RULE_template =
$(call RULE_template_base,$(strip $(1)),$(strip $(2)),$(strip $(3)),$(strip $(4)))
endef

$(eval $(call RULE_template, seq,,                                              SEQ,     MPI))
$(eval $(call RULE_template, genseq,,                                           SEQ,     MPI))
$(eval $(call RULE_template, vec,      $(OMP_CPPFLAGS) -DVECTORIZE,             SEQ,     MPI))
$(eval $(call RULE_template, openmp,   $(OMP_CPPFLAGS),                         OPENMP,  MPI))
$(eval $(call RULE_template, openmp4,  $(OMP_OFFLOAD_CPPFLAGS) -DOP2_WITH_OMP4, OPENMP4,    ))
$(eval $(call RULE_template, cuda,,                                             CUDA,    MPI_CUDA))
$(eval $(call RULE_template, cuda_hyb, $(OMP_CPPFLAGS),                         CUDA,    MPI_CUDA))
#$(eval $(call RULE_template, gpi,,                                              SEQ,     GPI))


$(APP_NAME)_cuda: cuda/$(APP_NAME)_kernels.o
$(APP_NAME)_mpi_cuda: cuda/$(APP_NAME)_mpi_kernels.o

$(APP_NAME)_cuda_hyb: cuda/$(APP_NAME)_hybkernels_gpu.o cuda/$(APP_NAME)_hybkernels_cpu.o
$(APP_NAME)_mpi_cuda_hyb: cuda/$(APP_NAME)_mpi_hybkernels_gpu.o cuda/$(APP_NAME)_mpi_hybkernels_cpu.o

cuda/$(APP_NAME)_kernels.o: .generated
	$(NVCC) $(NVCCFLAGS) $(OP2_INC) -c cuda/$(APP_ENTRY_BASENAME)_kernels.cu -o $@

cuda/$(APP_NAME)_mpi_kernels.o: .generated
	$(NVCC) $(NVCCFLAGS) $(OP2_INC) -c cuda/$(APP_ENTRY_MPI_BASENAME)_kernels.cu -o $@

cuda/$(APP_NAME)_hybkernels_gpu.o: .generated
	$(NVCC) $(NVCCFLAGS) -DOP_HYBRID_GPU -DGPUPASS $(OP2_INC) \
		-c cuda/$(APP_ENTRY_BASENAME)_hybkernels.cu -o $@

cuda/$(APP_NAME)_hybkernels_cpu.o: .generated
	$(CXX) $(CXXFLAGS) $(OMP_CPPFLAGS) -x c++ -DOP_HYBRID_GPU $(OP2_INC) \
		-c cuda/$(APP_ENTRY_BASENAME)_hybkernels.cu -o $@

cuda/$(APP_NAME)_mpi_hybkernels_gpu.o: .generated
	$(NVCC) $(NVCCFLAGS) -DOP_HYBRID_GPU -DGPUPASS $(OP2_INC) \
		-c cuda/$(APP_ENTRY_MPI_BASENAME)_hybkernels.cu -o $@

cuda/$(APP_NAME)_mpi_hybkernels_cpu.o: .generated
	$(MPICXX) $(CXXFLAGS) $(OMP_CPPFLAGS) -x c++ -DOP_HYBRID_GPU $(OP2_INC) \
		-c cuda/$(APP_ENTRY_MPI_BASENAME)_hybkernels.cu -o $@

-include $(wildcard *.d)
