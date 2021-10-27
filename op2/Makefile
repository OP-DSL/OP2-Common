include ../makefiles/common.mk

LIB = $(OP2_BUILD_DIR)/lib
OBJ = $(OP2_BUILD_DIR)/obj
MOD = $(OP2_BUILD_DIR)/mod

FFLAGS += $(F_MOD_OUT_OPT)$(MOD)

OP2_LIBS = hdf5 seq cuda openmp openmp4 mpi mpi_cuda
OP2_FOR_LIBS = $(foreach lib,$(OP2_LIBS),f_$(lib))

.PHONY: all clean $(OP2_LIBS) $(OP2_FOR_LIBS)

all: $(OP2_LIBS) $(OP2_FOR_LIBS)

clean:
	-rm -rf $(OBJ)
	-rm -rf $(LIB)
	-rm -rf $(MOD)


# Base set of objects shared across multiple libraries
OP2_BASE = $(addprefix $(OBJ)/core/,\
	op_lib_core.o \
	op_rt_support.o)

OP2_FOR_BASE = $(addprefix $(OBJ)/fortran/,\
	op2_for_declarations.o \
	op2_for_reference.o \
	op2_for_rt_support.o \
	op2_for_rt_wrappers.o)

OP2_FOR_BASE_MPI = $(OP2_FOR_BASE) $(addprefix $(OBJ)/fortran/,\
	op2_C_reference+mpi.o \
	op2_for_C_wrappers+mpi.o \
	op2_for_hdf5_declarations.o)

OP2_FOR_BASE += $(addprefix $(OBJ)/fortran/,\
	op2_C_reference.o \
	op2_for_C_wrappers.o)


# Complete object composition for each library
OP2_HDF5 = $(addprefix $(OBJ)/,\
	externlib/op_util.o \
	externlib/op_hdf5.o)

OP2_FOR_HDF5 = $(OP2_HDF5) $(addprefix $(OBJ)/fortran/,\
	op2_for_hdf5_declarations.o)

OP2_SEQ = $(OP2_BASE) $(addprefix $(OBJ)/,\
	core/op_dummy_singlenode.o \
	sequential/op_seq.o)

OP2_FOR_SEQ = $(OP2_SEQ) $(OP2_FOR_BASE) $(addprefix $(OBJ)/fortran/,\
	op_dummy_wrappers.o)

OP2_CUDA = $(OP2_BASE) $(addprefix $(OBJ)/,\
	cuda/op_cuda_decl.o \
	cuda/op_cuda_rt_support.o)

OP2_FOR_CUDA = $(OP2_CUDA) $(OP2_FOR_BASE) $(addprefix $(OBJ)/fortran/,\
	op2_for_rt_wrappers_cuda.o \
	cudaConfigurationParams.o)

OP2_OPENMP = $(OP2_BASE) $(addprefix $(OBJ)/,\
	core/op_dummy_singlenode.o \
	openmp/op_openmp_decl.o)

OP2_FOR_OPENMP = $(OP2_OPENMP) $(OP2_FOR_BASE) $(addprefix $(OBJ)/fortran/,\
	op_dummy_wrappers.o)

OP2_OPENMP4 = $(OP2_BASE) $(addprefix $(OBJ)/,\
	core/op_dummy_singlenode.o \
	openmp4/op_openmp4_decl.o \
	openmp4/op_openmp4_rt_support.o)

OP2_FOR_OPENMP4 = $(OP2_OPENMP4) $(OP2_FOR_BASE) $(addprefix $(OBJ)/fortran/,\
	op_dummy_wrappers.o)

OP2_MPI = $(OP2_BASE) $(addprefix $(OBJ)/,\
	mpi/op_mpi_core.o \
	mpi/op_mpi_part_core.o \
	mpi/op_mpi_decl.o \
	mpi/op_mpi_rt_support.o \
	mpi/op_mpi_hdf5.o \
	mpi/op_mpi_util.o \
	externlib/op_util.o \
	externlib/op_renumber.o)

OP2_FOR_MPI = $(OP2_MPI) $(OP2_FOR_BASE_MPI) $(addprefix $(OBJ)/fortran/,\
	op_dummy_wrappers+mpi.o)

OP2_MPI_CUDA = $(OP2_BASE) $(addprefix $(OBJ)/,\
	cuda/op_cuda_rt_support+mpi.o \
	mpi/op_mpi_core.o \
	mpi/op_mpi_part_core.o \
	mpi/op_mpi_cuda_decl.o \
	mpi/op_mpi_cuda_rt_support.o \
	mpi/op_mpi_cuda_kernels.o \
	mpi/op_mpi_hdf5.o \
	mpi/op_mpi_util.o \
	externlib/op_util.o \
	externlib/op_renumber.o)

OP2_FOR_MPI_CUDA = $(OP2_MPI_CUDA) $(OP2_FOR_BASE_MPI) $(addprefix $(OBJ)/fortran/,\
	op2_for_rt_wrappers_cuda.o)


# Directory rules
$(OBJ):
	@mkdir -p $@
	@mkdir -p $(foreach dir,$(shell find src -maxdepth 1 -mindepth 1 \
		-exec basename {} \;),$@/$(dir))

$(LIB):
	@mkdir -p $@

$(MOD):
	@mkdir -p $@


# Rules for files with a specific compilation setup
$(OBJ)/externlib/op_renumber.o: src/externlib/op_renumber.cpp | $(OBJ)
	$(MPICXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(OBJ)/cuda/%+mpi.o: src/cuda/%.cpp | $(OBJ)
	$(MPICXX) $(CXXFLAGS) $(INC) -DOPMPI -c $< -o $@

$(OBJ)/cuda/%.o: src/cuda/%.cpp | $(OBJ)
	$(CXX) $(CXXFLAGS) $(INC) -DSET_CUDA_CACHE_CONFIG -c $< -o $@

$(OBJ)/openmp4/%.o: src/openmp4/%.cpp | $(OBJ)
	$(CXX) $(CXXFLAGS) $(OMP_OFFLOAD_CXXFLAGS) $(INC) -c $< -o $@

$(OBJ)/mpi/%.o: src/mpi/%.cpp | $(OBJ)
	$(MPICXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(OBJ)/mpi/%.o: src/mpi/%.cu | $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(INC) -c $< -o $@

$(OBJ)/fortran/%+mpi.o: src/fortran/%.c | $(OBJ)
	$(CC) $(CFLAGS) $(INC) -DOPMPI -c $< -o $@

# Compile OP2_Fortran_Declarations first, as other F90 sources use it as a module
$(OBJ)/fortran/op2_for_declarations.o: src/fortran/op2_for_declarations.F90 | $(OBJ) $(MOD)
	$(FC) $(FFLAGS) $(INC) -DOP2_ARG_POINTERS -c $< -o $@

# Manually run C preprocessor on op2_for_reference.F90 to generate the op_par_loop subroutines
$(OBJ)/fortran/op2_for_reference+cpp.F90: src/fortran/op2_for_reference.F90 | $(OBJ)
	$(shell cpp $(INC) $< | sed s/##//g | sed s/\"@\"//g | tr "@" "\\n" > $@)

# Compile the expanded $(OBJ)/fortran/{}+cpp.F90 file(s)
$(OBJ)/fortran/%.o: $(OBJ)/fortran/%+cpp.F90 $(OBJ)/fortran/op2_for_declarations.o | $(OBJ) $(MOD)
	$(FC) $(FFLAGS) $(INC) -DOP2_ARG_POINTERS -c $< -o $@

$(OBJ)/fortran/%.o: src/fortran/%.F90 $(OBJ)/fortran/op2_for_declarations.o | $(OBJ) $(MOD)
	$(FC) $(FFLAGS) $(INC) -DOP2_ARG_POINTERS -c $< -o $@

# Catch-all rules
$(OBJ)/%.o: src/%.cpp | $(OBJ)
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

$(OBJ)/%.o: src/%.c | $(OBJ)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

# Include auto-generated dependencies for C/CXX sources (if they exist)
ifneq ($(wildcard $(OBJ)),)
-include $(shell find $(OBJ) -type f -name "*.d")
endif


# Helper function to upper-case a string
UPPERCASE = $(shell echo "$(1)" | tr "[:lower:]" "[:upper:]")

# Template for generating the library rules libop2_$1.a and libop2_for_$1.a
# and the compatability PHONY rules $1 and f_$1
define OP2_LIB_template =
$$(LIB)/libop2_$(1).a: $$(OP2_$(call UPPERCASE,$(1))) | $$(LIB)
	$(AR) $$@ $$?

$$(LIB)/libop2_for_$(1).a: $$(OP2_FOR_$(call UPPERCASE,$(1))) | $$(LIB)
	$(AR) $$@ $$?

$(1): $$(LIB)/libop2_$(1).a
f_$(1): $$(LIB)/libop2_for_$(1).a
endef

# Expand the template for all of the libraries (seq, ..., mpi_cuda)
$(foreach lib,$(OP2_LIBS),$(eval $(call OP2_LIB_template,$(lib))))