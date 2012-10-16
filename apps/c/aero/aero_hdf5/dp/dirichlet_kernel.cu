//
// auto-generated by op2.m on 16-Aug-2012 22:05:18
//

// user function

__device__
#include "dirichlet.h"


// CUDA kernel function

__global__ void op_cuda_dirichlet(
  double *ind_arg0,
  int   *ind_map,
  short *arg_map,
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


  __shared__ int   *ind_arg0_map, ind_arg0_size;
  __shared__ double *ind_arg0_s;
  __shared__ int    nelem, offset_b;

  extern __shared__ char shared[];

  if (blockIdx.x+blockIdx.y*gridDim.x >= nblocks) return;
  if (threadIdx.x==0) {

    // get sizes and shift pointers and direct-mapped data

    int blockId = blkmap[blockIdx.x + blockIdx.y*gridDim.x  + block_offset];

    nelem    = nelems[blockId];
    offset_b = offset[blockId];

    ind_arg0_size = ind_arg_sizes[0+blockId*1];

    ind_arg0_map = &ind_map[0*set_size] + ind_arg_offs[0+blockId*1];

    // set shared memory pointers

    int nbytes = 0;
    ind_arg0_s = (double *) &shared[nbytes];
  }

  __syncthreads(); // make sure all of above completed

  // copy indirect datasets into shared memory or zero increment

  __syncthreads();

  // process set elements

  for (int n=threadIdx.x; n<nelem; n+=blockDim.x) {

      // user-supplied kernel call


      dirichlet(  ind_arg0_s+arg_map[0*set_size+n+offset_b]*1 );
  }

  // apply pointered write/increment

  for (int n=threadIdx.x; n<ind_arg0_size*1; n+=blockDim.x)
    ind_arg0[n%1+ind_arg0_map[n/1]*1] = ind_arg0_s[n];

}


// host stub function

void op_par_loop_dirichlet(char const *name, op_set set,
  op_arg arg0 ){


  int    nargs   = 1;
  op_arg args[1];

  args[0] = arg0;

  int    ninds   = 1;
  int    inds[1] = {0};

  if (OP_diags>2) {
    printf(" kernel routine with indirection: dirichlet\n");
  }

  #ifdef OP_PART_SIZE_1
    int part_size = OP_PART_SIZE_1;
  #else
    int part_size = OP_part_size;
  #endif

  int set_size = op_mpi_halo_exchanges(set, nargs, args);

  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  if (set->size >0) {

    // get plan
    op_plan *Plan = op_plan_get(name,set,part_size,nargs,args,ninds,inds);

    // initialise timers
    op_timers_core(&cpu_t1, &wall_t1);

    // execute plan

    int block_offset = 0;

    for (int col=0; col < Plan->ncolors; col++) {

      if (col==Plan->ncolors_core) op_mpi_wait_all(nargs,args);

    #ifdef OP_BLOCK_SIZE_1
      int nthread = OP_BLOCK_SIZE_1;
    #else
      int nthread = OP_block_size;
    #endif

      dim3 nblocks = dim3(Plan->ncolblk[col] >= (1<<16) ? 65535 : Plan->ncolblk[col],
                      Plan->ncolblk[col] >= (1<<16) ? (Plan->ncolblk[col]-1)/65535+1: 1, 1);
      if (Plan->ncolblk[col] > 0) {
        int nshared = Plan->nsharedCol[col];
        op_cuda_dirichlet<<<nblocks,nthread,nshared>>>(
           (double *)arg0.data_d,
           Plan->ind_map,
           Plan->loc_map,
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

        cutilSafeCall(cudaThreadSynchronize());
        cutilCheckMsg("op_cuda_dirichlet execution failed\n");
      }

      block_offset += Plan->ncolblk[col];
    }

    op_timing_realloc(1);
    OP_kernels[1].transfer  += Plan->transfer;
    OP_kernels[1].transfer2 += Plan->transfer2;

  }


  op_mpi_set_dirtybit(nargs, args);

  // update kernel record

  op_timers_core(&cpu_t2, &wall_t2);
  op_timing_realloc(1);
  OP_kernels[1].name      = name;
  OP_kernels[1].count    += 1;
  OP_kernels[1].time     += wall_t2 - wall_t1;
}

