//
// auto-generated by op2.py
//

//user function
int opDat0_res_calc_stride_OP2CONSTANT;
int opDat0_res_calc_stride_OP2HOST=-1;
int direct_res_calc_stride_OP2CONSTANT;
int direct_res_calc_stride_OP2HOST=-1;
//user function

void res_calc_omp4_kernel(
  int *map0,
  int map0size,
  double *data8,
  int dat8size,
  double *data0,
  int dat0size,
  double *data4,
  int dat4size,
  double *data9,
  int dat9size,
  int *col_reord,
  int set_size1,
  int start,
  int end,
  int num_teams,
  int nthread,
  int opDat0_res_calc_stride_OP2CONSTANT,
  int direct_res_calc_stride_OP2CONSTANT);

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
  OP_kernels[0].name      = name;
  OP_kernels[0].count    += 1;

  int  ninds   = 3;
  int  inds[13] = {0,0,0,0,1,1,1,1,-1,2,2,2,2};

  if (OP_diags>2) {
    printf(" kernel routine with indirection: res_calc\n");
  }

  // get plan
  int set_size = op_mpi_halo_exchanges_cuda(set, nargs, args);

  #ifdef OP_PART_SIZE_0
    int part_size = OP_PART_SIZE_0;
  #else
    int part_size = OP_part_size;
  #endif
  #ifdef OP_BLOCK_SIZE_0
    int nthread = OP_BLOCK_SIZE_0;
  #else
    int nthread = OP_block_size;
  #endif


  int ncolors = 0;
  int set_size1 = set->size + set->exec_size;

  if (set_size >0) {

    if ((OP_kernels[0].count==1) || (opDat0_res_calc_stride_OP2HOST != getSetSizeFromOpArg(&arg0))) {
      opDat0_res_calc_stride_OP2HOST = getSetSizeFromOpArg(&arg0);
      opDat0_res_calc_stride_OP2CONSTANT = opDat0_res_calc_stride_OP2HOST;
    }
    if ((OP_kernels[0].count==1) || (direct_res_calc_stride_OP2HOST != getSetSizeFromOpArg(&arg8))) {
      direct_res_calc_stride_OP2HOST = getSetSizeFromOpArg(&arg8);
      direct_res_calc_stride_OP2CONSTANT = direct_res_calc_stride_OP2HOST;
    }

    //Set up typed device pointers for OpenMP
    int *map0 = arg0.map_data_d;
     int map0size = arg0.map->dim * set_size1;

    double* data8 = (double*)arg8.data_d;
    int dat8size = getSetSizeFromOpArg(&arg8) * arg8.dat->dim;
    double *data0 = (double *)arg0.data_d;
    int dat0size = getSetSizeFromOpArg(&arg0) * arg0.dat->dim;
    double *data4 = (double *)arg4.data_d;
    int dat4size = getSetSizeFromOpArg(&arg4) * arg4.dat->dim;
    double *data9 = (double *)arg9.data_d;
    int dat9size = getSetSizeFromOpArg(&arg9) * arg9.dat->dim;

    op_plan *Plan = op_plan_get_stage(name,set,part_size,nargs,args,ninds,inds,OP_COLOR2);
    ncolors = Plan->ncolors;
    int *col_reord = Plan->col_reord;

    // execute plan
    for ( int col=0; col<Plan->ncolors; col++ ){
      if (col==1) {
        op_mpi_wait_all_cuda(nargs, args);
      }
      int start = Plan->col_offsets[0][col];
      int end = Plan->col_offsets[0][col+1];

      res_calc_omp4_kernel(
        map0,
        map0size,
        data8,
        dat8size,
        data0,
        dat0size,
        data4,
        dat4size,
        data9,
        dat9size,
        col_reord,
        set_size1,
        start,
        end,
        part_size!=0?(end-start-1)/part_size+1:(end-start-1)/nthread,
        nthread,
        opDat0_res_calc_stride_OP2CONSTANT,
        direct_res_calc_stride_OP2CONSTANT);

    }
    OP_kernels[0].transfer  += Plan->transfer;
    OP_kernels[0].transfer2 += Plan->transfer2;
  }

  if (set_size == 0 || set_size == set->core_size || ncolors == 1) {
    op_mpi_wait_all_cuda(nargs, args);
  }
  // combine reduction data
  op_mpi_set_dirtybit_cuda(nargs, args);

  if (OP_diags>1) deviceSync();
  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[0].time     += wall_t2 - wall_t1;
}
