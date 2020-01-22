//
// auto-generated by op2.py
//

#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
#include "op_cuda_reduction.h"
//global_constants - values #defined by JIT
#include "jit_const.h"

//user function
__device__ void adt_calc_gpu( const double *x1, const double *x2, const double *x3,
                     const double *x4, const double *q, double *adt)
{
  double dx, dy, ri, u, v, c;

  ri = 1.0f / q[0];
  u = ri * q[1];
  v = ri * q[2];
  c = sqrt(gam * gm1 * (ri * q[3] - 0.5f * (u * u + v * v)));

  dx = x2[0] - x1[0];
  dy = x2[1] - x1[1];
  *adt = fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  dx = x3[0] - x2[0];
  dy = x3[1] - x2[1];
  *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  dx = x4[0] - x3[0];
  dy = x4[1] - x3[1];
  *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  dx = x1[0] - x4[0];
  dy = x1[1] - x4[1];
  *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  *adt = (*adt) * (1.0f / cfl);

}

//C CUDA kernel function
__global__ void op_cuda_adt_calc_rec(
 const double* __restrict ind_arg0,
 const int* __restrict opDat0Map,
 const double* __restrict arg4,
 double* __restrict arg5,
 int start,
 int end,
 int set_size)
{

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid + start < end) {
    int n = tid + start;
    //Initialise locals
    int map0idx;
    map0idx = opDat0Map[n + set_size * 0];
    int map1idx;
    map1idx = opDat0Map[n + set_size * 1];
    int map2idx;
    map2idx = opDat0Map[n + set_size * 2];
    int map3idx;
    map3idx = opDat0Map[n + set_size * 3];

    //user function call
    adt_calc_gpu(ind_arg0+map0idx*2,
                 ind_arg0+map1idx*2,
                 ind_arg0+map2idx*2,
                 ind_arg0+map3idx*2,
                 arg4+n*4,
                 arg5+n*1
    );

  }
}

extern "C" {
void op_par_loop_adt_calc_rec_execute(op_kernel_descriptor* desc);

//Recompiled host stub function
void op_par_loop_adt_calc_rec_execute(op_kernel_descriptor* desc)
{
  op_set set = desc->set;
  int nargs = 6;

  op_arg arg0 = desc->args[0];
  op_arg arg1 = desc->args[1];
  op_arg arg2 = desc->args[2];
  op_arg arg3 = desc->args[3];
  op_arg arg4 = desc->args[4];
  op_arg arg5 = desc->args[5];

  op_arg args[6] = {arg0,
                    arg1,
                    arg2,
                    arg3,
                    arg4,
                    arg5,
  };


  //initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(1);
  op_timers_core(&cpu_t1, &wall_t1);

  if (OP_diags > 2) {
    printf(" kernel routine with indirection: adt_calc\n");
  }

  int set_size = op_mpi_halo_exchanges_cuda(set, nargs, args);

  if (set->size > 0) {

    //set CUDA execution parameters
    #ifdef OP_BLOCK_SIZE_1
      int nthread = OP_BLOCK_SIZE_1;
    #else
      int nthread = OP_block_size;
    #endif

    for (int round = 0; round < 2; ++round)
    {
      if (round==1) {
        op_mpi_wait_all_cuda(nargs, args);
      }
      int start = round==0 ? 0 : set->core_size;
      int end = round==0 ? set->core_size : set->size + set->exec_size;
      if (end - start>0) {
        int nblocks = (end-start-1)/nthread+1;
        op_cuda_adt_calc_rec<<<nblocks,nthread>>>(
          (double *)arg0.data_d,
          arg0.map_data_d,
          (double*)arg4.data_d,
          (double*)arg5.data_d,
          start,end,set->size+set->exec_size);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
          printf("CUDA error: %s\n", cudaGetErrorString(err));
          exit(1);
        }
      }
    }
  }
  op_mpi_set_dirtybit_cuda(nargs, args);

  cutilSafeCall(cudaDeviceSynchronize());
  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[1].time     += wall_t2 - wall_t1;
}

} //end extern c