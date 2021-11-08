ALL_APP_VARIANTS := seq genseq vec openmp openmp4 cuda cuda_hyb
ALL_APP_VARIANTS := $(ALL_APP_VARIANTS) $(foreach variant,$(ALL_APP_VARIANTS),mpi_$(variant))

BUILDABLE_APP_VARIANTS := seq genseq

ifeq ($(CPP_HAS_OMP),true)
  BUILDABLE_APP_VARIANTS += vec openmp
endif

ifeq ($(CPP_HAS_OMP4),true)
  BUILDABLE_APP_VARIANTS += openmp4
endif

ifneq ($(shell which $(NVCC) 2> /dev/null),)
  BUILDABLE_APP_VARIANTS += cuda cuda_hyb
endif

ifneq ($(shell which $(MPICC) 2> /dev/null),)
  BUILDABLE_APP_VARIANTS += $(foreach variant,$(BUILDABLE_APP_VARIANTS),mpi_$(variant))
endif

VARIANT_FILTER ?= %
VARIANT_FILTER_OUT ?=

BUILDABLE_APP_VARIANTS := $(filter-out $(VARIANT_FILTER_OUT),\
						  $(filter $(VARIANT_FILTER),$(BUILDABLE_APP_VARIANTS)))

ALL_APP_VARIANTS := $(foreach variant,$(ALL_APP_VARIANTS),$(APP_NAME)_$(variant))
BUILDABLE_APP_VARIANTS := $(foreach variant,$(BUILDABLE_APP_VARIANTS),$(APP_NAME)_$(variant))

.PHONY: all generate clean

all: $(BUILDABLE_APP_VARIANTS)

clean:
	-rm -f $(ALL_APP_VARIANTS)
	-rm -rf seq vec openmp openmp4 cuda openacc
	-rm -f *_op.cpp
	-rm -f *.d
	-rm -f out_grid_mpi.bin
	-rm -f out_grid_mpi.dat

generate:
	[ -f $(APP_NAME).cpp ] && $(ROOT_DIR)/translator/c/op2.py $(APP_NAME).cpp
	[ -f $(APP_NAME)_mpi.cpp ] && $(ROOT_DIR)/translator/c/op2.py $(APP_NAME)_mpi.cpp

$(APP_NAME)_seq: $(APP_NAME).cpp
	$(CXX) $(CXXFLAGS) $(OP2_INC) $^ $(OP2_LIB_SEQ) -o $@

$(APP_NAME)_mpi_seq: $(APP_NAME)_mpi.cpp
	$(MPICXX) $(CXXFLAGS) $(OP2_INC) $^ $(OP2_LIB_MPI) -o $@

$(APP_NAME)_genseq: $(APP_NAME)_op.cpp seq/$(APP_NAME)_seqkernels.cpp
	$(CXX) $(CXXFLAGS) $(OP2_INC) $^ $(OP2_LIB_SEQ) -o $@

$(APP_NAME)_mpi_genseq: $(APP_NAME)_mpi_op.cpp seq/$(APP_NAME)_mpi_seqkernels.cpp
	$(MPICXX) $(CXXFLAGS) $(OP2_INC) $^ $(OP2_LIB_MPI) -o $@

$(APP_NAME)_vec: $(APP_NAME)_op.cpp vec/$(APP_NAME)_veckernels.cpp
	$(CXX) $(CXXFLAGS) -DVECTORIZE $(OMP_CPPFLAGS) $(OP2_INC) $^ $(OP2_LIB_SEQ) -o $@

$(APP_NAME)_mpi_vec: $(APP_NAME)_mpi_op.cpp vec/$(APP_NAME)_mpi_veckernels.cpp
	$(MPICXX) $(CXXFLAGS) -DVECTORIZE $(OMP_CPPFLAGS) $(OP2_INC) $^ $(OP2_LIB_MPI) -o $@

$(APP_NAME)_openmp: $(APP_NAME)_op.cpp openmp/$(APP_NAME)_kernels.cpp
	$(CXX) $(CXXFLAGS) $(OMP_CPPFLAGS) $(OP2_INC) $^ $(OP2_LIB_OPENMP) -o $@

$(APP_NAME)_mpi_openmp: $(APP_NAME)_mpi_op.cpp openmp/$(APP_NAME)_mpi_kernels.cpp
	$(MPICXX) $(CXXFLAGS) $(OMP_CPPFLAGS) $(OP2_INC) $^ $(OP2_LIB_MPI) -o $@

$(APP_NAME)_openmp4: $(APP_NAME)_op.cpp openmp4/$(APP_NAME)_omp4kernels.cpp
	$(CXX) $(CXXFLAGS) $(OMP_CPPFLAGS) $(OP2_INC) $^ $(OP2_LIB_OPENMP) -o $@

$(APP_NAME)_mpi_openmp4: $(APP_NAME)_mpi_op.cpp openmp4/$(APP_NAME)_mpi_omp4kernels.cpp
	$(MPICXX) $(CXXFLAGS) $(OMP4_CPPFLAGS) $(OP2_INC) $^ $(OP2_LIB_MPI) -o $@

cuda/%_kernels.o: cuda/%_kernels.cu
	$(NVCC) $(NVCCFLAGS) $(OP2_INC) -c $^ -o $@

$(APP_NAME)_cuda: $(APP_NAME)_op.cpp cuda/$(APP_NAME)_kernels.o
	$(CXX) $(CXXFLAGS) $(OP2_INC) $^ $(OP2_LIB_CUDA) -o $@

$(APP_NAME)_mpi_cuda: $(APP_NAME)_mpi_op.cpp cuda/$(APP_NAME)_mpi_kernels.o
	$(MPICXX) $(CXXFLAGS) $(OP2_INC) $^ $(OP2_LIB_MPI_CUDA) -o $@

cuda/$(APP_NAME)_hybkernels_gpu.o: cuda/$(APP_NAME)_hybkernels.cu
	$(NVCC) $(NVCCFLAGS) -DOP_HYBRID_GPU -DGPUPASS $(OP2_INC) -c $^ -o $@

cuda/$(APP_NAME)_hybkernels_cpu.o: cuda/$(APP_NAME)_hybkernels.cu
	$(CXX) $(CXXFLAGS) $(OMP_CPPFLAGS) -x c++ -DOP_HYBRID_GPU $(OP2_INC) -c $^ -o $@

cuda/$(APP_NAME)_mpi_hybkernels_gpu.o: cuda/$(APP_NAME)_mpi_hybkernels.cu
	$(NVCC) $(NVCCFLAGS) -DOP_HYBRID_GPU -DGPUPASS $(OP2_INC) -c $^ -o $@

cuda/$(APP_NAME)_mpi_hybkernels_cpu.o: cuda/$(APP_NAME)_mpi_hybkernels.cu
	$(MPICXX) $(CXXFLAGS) $(OMP_CPPFLAGS) -x c++ -DOP_HYBRID_GPU $(OP2_INC) -c $^ -o $@

$(APP_NAME)_cuda_hyb: $(APP_NAME)_op.cpp cuda/$(APP_NAME)_hybkernels_gpu.o cuda/$(APP_NAME)_hybkernels_cpu.o
	$(CXX) $(CXXFLAGS) $(OMP_CPPFLAGS) $(OP2_INC) $^ $(OP2_LIB_CUDA) -o $@

$(APP_NAME)_mpi_cuda_hyb: $(APP_NAME)_mpi_op.cpp cuda/$(APP_NAME)_mpi_hybkernels_gpu.o cuda/$(APP_NAME)_mpi_hybkernels_cpu.o
	$(MPICXX) $(CXXFLAGS) $(OMP_CPPFLAGS) $(OP2_INC) $^ $(OP2_LIB_MPI_CUDA) -o $@

# TODO: sort out the headers with missing includes
# -include $(wildcard *.d)
