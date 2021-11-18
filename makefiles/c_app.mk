TRANSLATOR ?= $(ROOT_DIR)/translator/c/op2.py

APP_ENTRY ?= $(APP_NAME).cpp
APP_ENTRY_MPI ?= $(APP_NAME)_mpi.cpp

APP_ENTRY_BASENAME := $(basename $(APP_ENTRY))
APP_ENTRY_MPI_BASENAME := $(basename $(APP_ENTRY_MPI))

APP_ENTRY_OP := $(APP_ENTRY_BASENAME)_op.cpp
APP_ENTRY_MPI_OP := $(APP_ENTRY_MPI_BASENAME)_op.cpp

ALL_VARIANTS := seq genseq vec openmp openmp4 cuda cuda_hyb
ALL_VARIANTS += $(foreach variant,$(ALL_VARIANTS),mpi_$(variant))
ALL_VARIANTS := $(foreach variant,$(ALL_VARIANTS),$(APP_NAME)_$(variant))

BASE_BUILDABLE_VARIANTS := seq genseq

ifeq ($(CPP_HAS_OMP),true)
  BASE_BUILDABLE_VARIANTS += vec openmp
endif

ifeq ($(CPP_HAS_OMP_OFFLOAD),true)
  BASE_BUILDABLE_VARIANTS += openmp4
endif

ifneq ($(shell which $(NVCC) 2> /dev/null),)
  BASE_BUILDABLE_VARIANTS += cuda cuda_hyb
endif

BUILDABLE_VARIANTS :=
ifneq ($(wildcard ./$(APP_ENTRY)),)
  BUILDABLE_VARIANTS += $(foreach variant,$(BASE_BUILDABLE_VARIANTS),$(APP_NAME)_$(variant))
endif

ifneq ($(and $(wildcard ./$(APP_ENTRY_MPI)),$(shell which $(MPICC) 2> /dev/null)),)
  BUILDABLE_VARIANTS += $(foreach variant,$(BASE_BUILDABLE_VARIANTS),$(APP_NAME)_mpi_$(variant))
endif

VARIANT_FILTER ?= %
VARIANT_FILTER_OUT ?=

BUILDABLE_VARIANTS := $(filter-out $(VARIANT_FILTER_OUT),\
                      $(filter $(VARIANT_FILTER),$(BUILDABLE_VARIANTS)))

ifeq ($(OP2_LIBS_WITH_HDF5),true)
  ifndef HDF5_SEQ_INSTALL_PATH
    BUILDABLE_VARIANTS := $(filter $(APP_NAME)_mpi_%,$(BUILDABLE_VARIANTS))
  endif

  ifndef HDF5_PAR_INSTALL_PATH
    BUILDABLE_VARIANTS := $(filter-out $(APP_NAME)_mpi_%,$(BUILDABLE_VARIANTS))
  endif
endif

.PHONY: all generate clean

all: $(BUILDABLE_VARIANTS)

clean:
	-$(RM) $(ALL_VARIANTS)
	-$(RM) -r seq vec openmp openmp4 cuda openacc
	-$(RM) *_op.cpp
	-$(RM) .generated .generated
	-$(RM) *.d
	-$(RM) out_grid.*
	-$(RM) out_grid_mpi.*

.generated: $(APP_ENTRY) $(APP_ENTRY_MPI)
	[ ! -f $(APP_ENTRY) ] || $(TRANSLATOR) $(APP_ENTRY)
	[ ! -f $(APP_ENTRY_MPI) ] || [ $(APP_ENTRY_MPI) == $(APP_ENTRY) ] \
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

CUDA_SRC := $(APP_ENTRY_OP) cuda/$(APP_NAME)_kernels.o
MPI_CUDA_SRC := $(APP_ENTRY_MPI_OP) cuda/$(APP_NAME)_mpi_kernels.o

CUDA_HYB_SRC := $(APP_ENTRY_OP) \
	cuda/$(APP_NAME)_hybkernels_cpu.o cuda/$(APP_NAME)_hybkernels_gpu.o

MPI_CUDA_HYB_SRC := $(APP_ENTRY_MPI_OP) \
	cuda/$(APP_NAME)_mpi_hybkernels_cpu.o cuda/$(APP_NAME)_mpi_hybkernels_gpu.o

$(APP_NAME)_seq: $(SEQ_SRC)
	$(CXX) $(CXXFLAGS) $(OP2_INC) $^ $(OP2_LIB_SEQ) -o $@

$(APP_NAME)_mpi_seq: $(MPI_SEQ_SRC)
	$(MPICXX) $(CXXFLAGS) $(OP2_INC) $^ $(OP2_LIB_MPI) -o $@

$(APP_NAME)_genseq: .generated
	$(CXX) $(CXXFLAGS) $(OP2_INC) $(GENSEQ_SRC) $(OP2_LIB_SEQ) -o $@

$(APP_NAME)_mpi_genseq: .generated
	$(MPICXX) $(CXXFLAGS) $(OP2_INC) $(MPI_GENSEQ_SRC) $(OP2_LIB_MPI) -o $@

$(APP_NAME)_vec: .generated
	$(CXX) $(CXXFLAGS) -DVECTORIZE $(OMP_CPPFLAGS) $(OP2_INC) $(VEC_SRC) $(OP2_LIB_SEQ) -o $@

$(APP_NAME)_mpi_vec: .generated
	$(MPICXX) $(CXXFLAGS) -DVECTORIZE $(OMP_CPPFLAGS) $(OP2_INC) $(MPI_VEC_SRC) $(OP2_LIB_MPI) -o $@

$(APP_NAME)_openmp: .generated
	$(CXX) $(CXXFLAGS) $(OMP_CPPFLAGS) $(OP2_INC) $(OPENMP_SRC) $(OP2_LIB_OPENMP) -o $@

$(APP_NAME)_mpi_openmp: .generated
	$(MPICXX) $(CXXFLAGS) $(OMP_CPPFLAGS) $(OP2_INC) $(MPI_OPENMP_SRC) $(OP2_LIB_MPI) -o $@

$(APP_NAME)_openmp4: .generated
	$(CXX) $(CXXFLAGS) $(OMP_OFFLOAD_CPPFLAGS) $(OP2_INC) $(OPENMP4_SRC) $(OP2_LIB_OPENMP4) -o $@

$(APP_NAME)_mpi_openmp4: .generated
	$(MPICXX) $(CXXFLAGS) $(OMP_OFFLOAD_CPPFLAGS) $(OP2_INC) $(MPI_OPENMP4_SRC) $(OP2_LIB_MPI) -o $@

cuda/$(APP_NAME)_kernels.o: .generated
	$(NVCC) $(NVCCFLAGS) $(OP2_INC) -c cuda/$(APP_ENTRY_BASENAME)_kernels.cu -o $@

cuda/$(APP_NAME)_mpi_kernels.o: .generated
	$(NVCC) $(NVCCFLAGS) $(OP2_INC) -c cuda/$(APP_ENTRY_MPI_BASENAME)_kernels.cu -o $@

$(APP_NAME)_cuda: .generated cuda/$(APP_NAME)_kernels.o
	$(CXX) $(CXXFLAGS) $(OP2_INC) $(CUDA_SRC) $(OP2_LIB_CUDA) -o $@

$(APP_NAME)_mpi_cuda: .generated cuda/$(APP_NAME)_mpi_kernels.o
	$(MPICXX) $(CXXFLAGS) $(OP2_INC) $(MPI_CUDA_SRC) $(OP2_LIB_MPI_CUDA) -o $@

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

$(APP_NAME)_cuda_hyb: .generated cuda/$(APP_NAME)_hybkernels_gpu.o cuda/$(APP_NAME)_hybkernels_cpu.o
	$(CXX) $(CXXFLAGS) $(OMP_CPPFLAGS) $(OP2_INC) $(CUDA_HYB_SRC) $(OP2_LIB_CUDA) -o $@

$(APP_NAME)_mpi_cuda_hyb: .generated cuda/$(APP_NAME)_mpi_hybkernels_gpu.o cuda/$(APP_NAME)_mpi_hybkernels_cpu.o
	$(MPICXX) $(CXXFLAGS) $(OMP_CPPFLAGS) $(OP2_INC) $(MPI_CUDA_HYB_SRC) $(OP2_LIB_MPI_CUDA) -o $@

# TODO: sort out the headers with missing includes
# -include $(wildcard *.d)
