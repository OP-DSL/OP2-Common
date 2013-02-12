.PHONY: test test_sequential test_mpi

NPROCS = 4
MPIEXEC = mpirun -np $(NPROCS)

JAC_SEQUENTIAL = $(filter-out $(wildcard jac*mpi),$(wildcard jac*))
JAC_MPI = $(wildcard jac*mpi)
JAC_SEQUENTIAL_TESTS = $(foreach test,$(JAC_SEQUENTIAL),testseq_$(test))
JAC_MPI_TESTS = $(foreach test,$(JAC_MPI),testmpi_$(test))

test: test_sequential test_mpi

test_sequential: $(JAC_SEQUENTIAL_TESTS)
test_mpi: $(JAC_MPI_TESTS)

testseq_%:
	./$*

testmpi_%:
	$(MPIEXEC) ./$*
