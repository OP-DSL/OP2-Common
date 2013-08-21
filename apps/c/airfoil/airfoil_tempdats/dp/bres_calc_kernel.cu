//
// auto-generated by op2.m on 19-Oct-2012 16:21:10
//

// user function

__device__
#include "bres_calc.h"


// CUDA kernel function

__global__ void op_cuda_bres_calc(
  double *ind_arg0,
  double *ind_arg1,
  double *ind_arg2,
  double *ind_arg3,
  int   *ind_map,
  short *arg_map,
  int *arg5,
  int   *ind_arg_sizes,
  int   *ind_arg_offs,
  int    block_offset,
  int   *blkmap,
  int   *offset,
  int   *nelems,
  int   *ncolors,
  int   *colors,
  int   nblocks,
  int   set_size) {

  double arg4_l[4];

  __shared__ int   *ind_arg0_map, ind_arg0_size;
  __shared__ int   *ind_arg1_map, ind_arg1_size;
  __shared__ int   *ind_arg2_map, ind_arg2_size;
  __shared__ int   *ind_arg3_map, ind_arg3_size;
  __shared__ double *ind_arg0_s;
  __shared__ double *ind_arg1_s;
  __shared__ double *ind_arg2_s;
  __shared__ double *ind_arg3_s;
  __shared__ int    nelems2, ncolor;
  __shared__ int    nelem, offset_b;

  extern __shared__ char shared[];

  if (blockIdx.x+blockIdx.y*gridDim.x >= nblocks) return;
  if (threadIdx.x==0) {

    // get sizes and shift pointers and direct-mapped data

    int blockId = blkmap[blockIdx.x + blockIdx.y*gridDim.x  + block_offset];

    nelem    = nelems[blockId];
    offset_b = offset[blockId];

    nelems2  = blockDim.x*(1+(nelem-1)/blockDim.x);
    ncolor   = ncolors[blockId];

    ind_arg0_size = ind_arg_sizes[0+blockId*4];
    ind_arg1_size = ind_arg_sizes[1+blockId*4];
    ind_arg2_size = ind_arg_sizes[2+blockId*4];
    ind_arg3_size = ind_arg_sizes[3+blockId*4];

    ind_arg0_map = &ind_map[0*set_size] + ind_arg_offs[0+blockId*4];
    ind_arg1_map = &ind_map[2*set_size] + ind_arg_offs[1+blockId*4];
    ind_arg2_map = &ind_map[3*set_size] + ind_arg_offs[2+blockId*4];
    ind_arg3_map = &ind_map[4*set_size] + ind_arg_offs[3+blockId*4];

    // set shared memory pointers

    int nbytes = 0;
    ind_arg0_s = (double *) &shared[nbytes];
    nbytes    += ROUND_UP(ind_arg0_size*sizeof(double)*2);
    ind_arg1_s = (double *) &shared[nbytes];
    nbytes    += ROUND_UP(ind_arg1_size*sizeof(double)*4);
    ind_arg2_s = (double *) &shared[nbytes];
    nbytes    += ROUND_UP(ind_arg2_size*sizeof(double)*1);
    ind_arg3_s = (double *) &shared[nbytes];
  }

  __syncthreads(); // make sure all of above completed

  // copy indirect datasets into shared memory or zero increment

  for (int n=threadIdx.x; n<ind_arg0_size*2; n+=blockDim.x)
    ind_arg0_s[n] = ind_arg0[n%2+ind_arg0_map[n/2]*2];

  for (int n=threadIdx.x; n<ind_arg1_size*4; n+=blockDim.x)
    ind_arg1_s[n] = ind_arg1[n%4+ind_arg1_map[n/4]*4];

  for (int n=threadIdx.x; n<ind_arg2_size*1; n+=blockDim.x)
    ind_arg2_s[n] = ind_arg2[n%1+ind_arg2_map[n/1]*1];

  for (int n=threadIdx.x; n<ind_arg3_size*4; n+=blockDim.x)
    ind_arg3_s[n] = ZERO_double;

  __syncthreads();

  // process set elements

  for (int n=threadIdx.x; n<nelems2; n+=blockDim.x) {
    int col2 = -1;

    if (n<nelem) {

      // initialise local variables

      for (int d=0; d<4; d++)
        arg4_l[d] = ZERO_double;


      // user-supplied kernel call


      bres_calc(  ind_arg0_s+arg_map[0*set_size+n+offset_b]*2,
                  ind_arg0_s+arg_map[1*set_size+n+offset_b]*2,
                  ind_arg1_s+arg_map[2*set_size+n+offset_b]*4,
                  ind_arg2_s+arg_map[3*set_size+n+offset_b]*1,
                  arg4_l,
                  arg5+(n+offset_b)*1 );

      col2 = colors[n+offset_b];
    }

    // store local variables

      int arg4_map;

      if (col2>=0) {
        arg4_map = arg_map[4*set_size+n+offset_b];
      }

    for (int col=0; col<ncolor; col++) {
      if (col2==col) {
        for (int d=0; d<4; d++)
          ind_arg3_s[d+arg4_map*4] += arg4_l[d];
      }
      __syncthreads();
    }

  }

  // apply pointered write/increment

  for (int n=threadIdx.x; n<ind_arg3_size*4; n+=blockDim.x)
    ind_arg3[n%4+ind_arg3_map[n/4]*4] += ind_arg3_s[n];

}


// host stub function

void op_par_loop_bres_calc(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1,
  op_arg arg2,
  op_arg arg3,
  op_arg arg4,
  op_arg arg5 ){


  int    nargs   = 6;
  op_arg args[6];

  args[0] = arg0;
  args[1] = arg1;
  args[2] = arg2;
  args[3] = arg3;
  args[4] = arg4;
  args[5] = arg5;

  int    ninds   = 4;
  int    inds[6] = {0,0,1,2,3,-1};

  if (OP_diags>2) {
    printf(" kernel routine with indirection: bres_calc\n");
  }

  // get plan

  #ifdef OP_PART_SIZE_3
    int part_size = OP_PART_SIZE_3;
  #else
    int part_size = OP_part_size;
  #endif

  int set_size = op_mpi_halo_exchanges_cuda(set, nargs, args);

  // initialise timers

  double cpu_t1, cpu_t2, wall_t1=0, wall_t2=0;
  op_timing_realloc(3);
  OP_kernels[3].name      = name;
  OP_kernels[3].count    += 1;

  if (set->size >0) {

    op_plan *Plan = op_plan_get(name,set,part_size,nargs,args,ninds,inds);

    op_timers_core(&cpu_t1, &wall_t1);

    // execute plan

    int block_offset = 0;

    for (int col=0; col < Plan->ncolors; col++) {

      if (col==Plan->ncolors_core) op_mpi_wait_all_cuda(nargs,args);

    #ifdef OP_BLOCK_SIZE_3
      int nthread = OP_BLOCK_SIZE_3;
    #else
      int nthread = OP_block_size;
    #endif

      dim3 nblocks = dim3(Plan->ncolblk[col] >= (1<<16) ? 65535 : Plan->ncolblk[col],
                      Plan->ncolblk[col] >= (1<<16) ? (Plan->ncolblk[col]-1)/65535+1: 1, 1);
      if (Plan->ncolblk[col] > 0) {
        int nshared = Plan->nsharedCol[col];
        op_cuda_bres_calc<<<nblocks,nthread,nshared>>>(
           (double *)arg0.data_d,
           (double *)arg2.data_d,
           (double *)arg3.data_d,
           (double *)arg4.data_d,
           Plan->ind_map,
           Plan->loc_map,
           (int *)arg5.data_d,
           Plan->ind_sizes,
           Plan->ind_offs,
           block_offset,
           Plan->blkmap,
           Plan->offset,
           Plan->nelems,
           Plan->nthrcol,
           Plan->thrcol,
           Plan->ncolblk[col],
           set_size);

        cutilSafeCall(cudaDeviceSynchronize());
        cutilCheckMsg("op_cuda_bres_calc execution failed\n");
      }

      block_offset += Plan->ncolblk[col];
    }

    op_timing_realloc(3);
    OP_kernels[3].transfer  += Plan->transfer;
    OP_kernels[3].transfer2 += Plan->transfer2;

  }


  op_mpi_set_dirtybit_cuda(nargs, args);

  // update kernel record

  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[3].time     += wall_t2 - wall_t1;
}
