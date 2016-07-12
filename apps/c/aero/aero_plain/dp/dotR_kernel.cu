//
// auto-generated by op2.m on 19-Oct-2012 16:21:06
//

// user function

__device__
#include "dotR.h"

    // CUDA kernel function

    __global__ void
    op_cuda_dotR(double *arg0, double *arg1, int offset_s, int set_size) {

  double arg1_l[1];
  for (int d = 0; d < 1; d++)
    arg1_l[d] = ZERO_double;

  // process set elements

  for (int n = threadIdx.x + blockIdx.x * blockDim.x; n < set_size;
       n += blockDim.x * gridDim.x) {

    // user-supplied kernel call

    dotR(arg0 + n, arg1_l);
  }

  // global reductions

  for (int d = 0; d < 1; d++)
    op_reduction<OP_INC>(&arg1[d + blockIdx.x * 1], arg1_l[d]);
}

// host stub function

void op_par_loop_dotR(char const *name, op_set set, op_arg arg0, op_arg arg1) {

  double *arg1h = (double *)arg1.data;

  int nargs = 2;
  op_arg args[2];

  args[0] = arg0;
  args[1] = arg1;

  if (OP_diags > 2) {
    printf(" kernel routine w/o indirection:  dotR\n");
  }

  op_mpi_halo_exchanges_cuda(set, nargs, args);

  // initialise timers

  double cpu_t1, cpu_t2, wall_t1 = 0, wall_t2 = 0;
  op_timing_realloc(6);
  OP_kernels[6].name = name;
  OP_kernels[6].count += 1;

  if (set->size > 0) {

    op_timers_core(&cpu_t1, &wall_t1);

// set CUDA execution parameters

#ifdef OP_BLOCK_SIZE_6
    int nthread = OP_BLOCK_SIZE_6;
#else
    // int nthread = OP_block_size;
    int nthread = 128;
#endif

    int nblocks = 200;

    // transfer global reduction data to GPU

    int maxblocks = nblocks;

    int reduct_bytes = 0;
    int reduct_size = 0;
    reduct_bytes += ROUND_UP(maxblocks * 1 * sizeof(double));
    reduct_size = MAX(reduct_size, sizeof(double));

    reallocReductArrays(reduct_bytes);

    reduct_bytes = 0;
    arg1.data = OP_reduct_h + reduct_bytes;
    arg1.data_d = OP_reduct_d + reduct_bytes;
    for (int b = 0; b < maxblocks; b++)
      for (int d = 0; d < 1; d++)
        ((double *)arg1.data)[d + b * 1] = ZERO_double;
    reduct_bytes += ROUND_UP(maxblocks * 1 * sizeof(double));

    mvReductArraysToDevice(reduct_bytes);

    // work out shared memory requirements per element

    int nshared = 0;

    // execute plan

    int offset_s = nshared * OP_WARPSIZE;

    nshared = MAX(nshared * nthread, reduct_size * nthread);

    op_cuda_dotR<<<nblocks, nthread, nshared>>>(
        (double *)arg0.data_d, (double *)arg1.data_d, offset_s, set->size);

    cutilSafeCall(cudaDeviceSynchronize());
    cutilCheckMsg("op_cuda_dotR execution failed\n");

    // transfer global reduction data back to CPU

    mvReductArraysToHost(reduct_bytes);

    for (int b = 0; b < maxblocks; b++)
      for (int d = 0; d < 1; d++)
        arg1h[d] = arg1h[d] + ((double *)arg1.data)[d + b * 1];

    arg1.data = (char *)arg1h;

    op_mpi_reduce(&arg1, arg1h);
  }

  op_mpi_set_dirtybit_cuda(nargs, args);

  // update kernel record

  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[6].time += wall_t2 - wall_t1;
  OP_kernels[6].transfer += (float)set->size * arg0.size;
}
