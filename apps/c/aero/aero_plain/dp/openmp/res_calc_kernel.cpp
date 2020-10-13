//
// auto-generated by op2.py
//

//user function
#include "../res_calc.h"

// host stub function
void op_par_loop_res_calc(char const *name, op_set set,
  op_arg arg0,
  op_arg arg4,
  op_arg arg8,
  op_arg arg9){

  int nargs = 13;
  op_arg args[13];

  arg0.idx = 0;
  args[0] = arg0;
  for ( int v=1; v<4; v++ ){
    args[0 + v] = op_arg_dat(arg0.dat, v, arg0.map, 2, "double", OP_READ);
  }

  arg4.idx = 0;
  args[4] = arg4;
  for ( int v=1; v<4; v++ ){
    args[4 + v] = op_arg_dat(arg4.dat, v, arg4.map, 1, "double", OP_READ);
  }

  args[8] = arg8;
  arg9.idx = 0;
  args[9] = arg9;
  for ( int v=1; v<4; v++ ){
    args[9 + v] = op_arg_dat(arg9.dat, v, arg9.map, 1, "double", OP_INC);
  }


  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(0);
  op_timers_core(&cpu_t1, &wall_t1);

  int  ninds   = 3;
  int  inds[13] = {0,0,0,0,1,1,1,1,-1,2,2,2,2};

  if (OP_diags>2) {
    printf(" kernel routine with indirection: res_calc\n");
  }

  // get plan
  #ifdef OP_PART_SIZE_0
    int part_size = OP_PART_SIZE_0;
  #else
    int part_size = OP_part_size;
  #endif

  int set_size = op_mpi_halo_exchanges(set, nargs, args);

  if (set_size > 0) {

    op_plan *Plan = op_plan_get_stage_upload(name,set,part_size,nargs,args,ninds,inds,OP_STAGE_ALL,0);

    // execute plan
    int block_offset = 0;
    for ( int col=0; col<Plan->ncolors; col++ ){
      if (col==Plan->ncolors_core) {
        op_mpi_wait_all(nargs, args);
      }
      int nblocks = Plan->ncolblk[col];

      #pragma omp parallel for
      for ( int blockIdx=0; blockIdx<nblocks; blockIdx++ ){
        int blockId  = Plan->blkmap[blockIdx + block_offset];
        int nelem    = Plan->nelems[blockId];
        int offset_b = Plan->offset[blockId];
        for ( int n=offset_b; n<offset_b+nelem; n++ ){
          int map0idx;
          int map1idx;
          int map2idx;
          int map3idx;
          map0idx = arg0.map_data[n * arg0.map->dim + 0];
          map1idx = arg0.map_data[n * arg0.map->dim + 1];
          map2idx = arg0.map_data[n * arg0.map->dim + 2];
          map3idx = arg0.map_data[n * arg0.map->dim + 3];

          const double* arg0_vec[] = {
             &((double*)arg0.data)[2 * map0idx],
             &((double*)arg0.data)[2 * map1idx],
             &((double*)arg0.data)[2 * map2idx],
             &((double*)arg0.data)[2 * map3idx]};
          const double* arg4_vec[] = {
             &((double*)arg4.data)[1 * map0idx],
             &((double*)arg4.data)[1 * map1idx],
             &((double*)arg4.data)[1 * map2idx],
             &((double*)arg4.data)[1 * map3idx]};
          double* arg9_vec[] = {
             &((double*)arg9.data)[1 * map0idx],
             &((double*)arg9.data)[1 * map1idx],
             &((double*)arg9.data)[1 * map2idx],
             &((double*)arg9.data)[1 * map3idx]};

          res_calc(
            arg0_vec,
            arg4_vec,
            &((double*)arg8.data)[16 * n],
            arg9_vec);
        }
      }

      block_offset += nblocks;
    }
    OP_kernels[0].transfer  += Plan->transfer;
    OP_kernels[0].transfer2 += Plan->transfer2;
  }

  if (set_size == 0 || set_size == set->core_size) {
    op_mpi_wait_all(nargs, args);
  }
  // combine reduction data
  op_mpi_set_dirtybit(nargs, args);

  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[0].name      = name;
  OP_kernels[0].count    += 1;
  OP_kernels[0].time     += wall_t2 - wall_t1;
}
