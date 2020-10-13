//
// auto-generated by op2.py
//

__constant__ int opDat0_res_calc_stride_OP2CONSTANT;
int opDat0_res_calc_stride_OP2HOST=-1;
__constant__ int opDat2_res_calc_stride_OP2CONSTANT;
int opDat2_res_calc_stride_OP2HOST=-1;
//user function
__device__ void res_calc_gpu( const double *x1, const double *x2, const double *q1,
                     const double *q2, const double *adt1, const double *adt2,
                     double *res1, double *res2) {
  double dx, dy, mu, ri, p1, vol1, p2, vol2, f;

  dx = x1[(0)*opDat0_res_calc_stride_OP2CONSTANT] - x2[(0)*opDat0_res_calc_stride_OP2CONSTANT];
  dy = x1[(1)*opDat0_res_calc_stride_OP2CONSTANT] - x2[(1)*opDat0_res_calc_stride_OP2CONSTANT];

  ri = 1.0f / q1[(0)*opDat2_res_calc_stride_OP2CONSTANT];
  p1 = gm1_cuda * (q1[(3)*opDat2_res_calc_stride_OP2CONSTANT] - 0.5f * ri * (q1[(1)*opDat2_res_calc_stride_OP2CONSTANT] * q1[(1)*opDat2_res_calc_stride_OP2CONSTANT] + q1[(2)*opDat2_res_calc_stride_OP2CONSTANT] * q1[(2)*opDat2_res_calc_stride_OP2CONSTANT]));
  vol1 = ri * (q1[(1)*opDat2_res_calc_stride_OP2CONSTANT] * dy - q1[(2)*opDat2_res_calc_stride_OP2CONSTANT] * dx);

  ri = 1.0f / q2[(0)*opDat2_res_calc_stride_OP2CONSTANT];
  p2 = gm1_cuda * (q2[(3)*opDat2_res_calc_stride_OP2CONSTANT] - 0.5f * ri * (q2[(1)*opDat2_res_calc_stride_OP2CONSTANT] * q2[(1)*opDat2_res_calc_stride_OP2CONSTANT] + q2[(2)*opDat2_res_calc_stride_OP2CONSTANT] * q2[(2)*opDat2_res_calc_stride_OP2CONSTANT]));
  vol2 = ri * (q2[(1)*opDat2_res_calc_stride_OP2CONSTANT] * dy - q2[(2)*opDat2_res_calc_stride_OP2CONSTANT] * dx);

  mu = 0.5f * ((*adt1) + (*adt2)) * eps_cuda;

  f = 0.5f * (vol1 * q1[(0)*opDat2_res_calc_stride_OP2CONSTANT] + vol2 * q2[(0)*opDat2_res_calc_stride_OP2CONSTANT]) + mu * (q1[(0)*opDat2_res_calc_stride_OP2CONSTANT] - q2[(0)*opDat2_res_calc_stride_OP2CONSTANT]);
  res1[0] += f;
  res2[0] -= f;
  f = 0.5f * (vol1 * q1[(1)*opDat2_res_calc_stride_OP2CONSTANT] + p1 * dy + vol2 * q2[(1)*opDat2_res_calc_stride_OP2CONSTANT] + p2 * dy) +
      mu * (q1[(1)*opDat2_res_calc_stride_OP2CONSTANT] - q2[(1)*opDat2_res_calc_stride_OP2CONSTANT]);
  res1[1] += f;
  res2[1] -= f;
  f = 0.5f * (vol1 * q1[(2)*opDat2_res_calc_stride_OP2CONSTANT] - p1 * dx + vol2 * q2[(2)*opDat2_res_calc_stride_OP2CONSTANT] - p2 * dx) +
      mu * (q1[(2)*opDat2_res_calc_stride_OP2CONSTANT] - q2[(2)*opDat2_res_calc_stride_OP2CONSTANT]);
  res1[2] += f;
  res2[2] -= f;
  f = 0.5f * (vol1 * (q1[(3)*opDat2_res_calc_stride_OP2CONSTANT] + p1) + vol2 * (q2[(3)*opDat2_res_calc_stride_OP2CONSTANT] + p2)) + mu * (q1[(3)*opDat2_res_calc_stride_OP2CONSTANT] - q2[(3)*opDat2_res_calc_stride_OP2CONSTANT]);
  res1[3] += f;
  res2[3] -= f;

}

// CUDA kernel function
__global__ void op_cuda_res_calc(
  const double *__restrict ind_arg0,
  const double *__restrict ind_arg1,
  const double *__restrict ind_arg2,
  double *__restrict ind_arg3,
  const int *__restrict opDat0Map,
  const int *__restrict opDat2Map,
  int start,
  int end,
  int   set_size) {
  double arg6_l[4];
  double arg7_l[4];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid + start < end) {
    int n = tid + start;
    //initialise local variables
    double arg6_l[4];
    for ( int d=0; d<4; d++ ){
      arg6_l[d] = ZERO_double;
    }
    double arg7_l[4];
    for ( int d=0; d<4; d++ ){
      arg7_l[d] = ZERO_double;
    }
    int map0idx;
    int map1idx;
    int map2idx;
    int map3idx;
    map0idx = opDat0Map[n + set_size * 0];
    map1idx = opDat0Map[n + set_size * 1];
    map2idx = opDat2Map[n + set_size * 0];
    map3idx = opDat2Map[n + set_size * 1];

    //user-supplied kernel call
    res_calc_gpu(ind_arg0+map0idx,
             ind_arg0+map1idx,
             ind_arg1+map2idx,
             ind_arg1+map3idx,
             ind_arg2+map2idx*1,
             ind_arg2+map3idx*1,
             arg6_l,
             arg7_l);
    atomicAdd(&ind_arg3[0*opDat2_res_calc_stride_OP2CONSTANT+map2idx],arg6_l[0]);
    atomicAdd(&ind_arg3[1*opDat2_res_calc_stride_OP2CONSTANT+map2idx],arg6_l[1]);
    atomicAdd(&ind_arg3[2*opDat2_res_calc_stride_OP2CONSTANT+map2idx],arg6_l[2]);
    atomicAdd(&ind_arg3[3*opDat2_res_calc_stride_OP2CONSTANT+map2idx],arg6_l[3]);
    atomicAdd(&ind_arg3[0*opDat2_res_calc_stride_OP2CONSTANT+map3idx],arg7_l[0]);
    atomicAdd(&ind_arg3[1*opDat2_res_calc_stride_OP2CONSTANT+map3idx],arg7_l[1]);
    atomicAdd(&ind_arg3[2*opDat2_res_calc_stride_OP2CONSTANT+map3idx],arg7_l[2]);
    atomicAdd(&ind_arg3[3*opDat2_res_calc_stride_OP2CONSTANT+map3idx],arg7_l[3]);
  }
}


//host stub function
void op_par_loop_res_calc(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1,
  op_arg arg2,
  op_arg arg3,
  op_arg arg4,
  op_arg arg5,
  op_arg arg6,
  op_arg arg7){

  int nargs = 8;
  op_arg args[8];

  args[0] = arg0;
  args[1] = arg1;
  args[2] = arg2;
  args[3] = arg3;
  args[4] = arg4;
  args[5] = arg5;
  args[6] = arg6;
  args[7] = arg7;

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(2);
  op_timers_core(&cpu_t1, &wall_t1);
  OP_kernels[2].name      = name;
  OP_kernels[2].count    += 1;


  int    ninds   = 4;
  int    inds[8] = {0,0,1,1,2,2,3,3};

  if (OP_diags>2) {
    printf(" kernel routine with indirection: res_calc\n");
  }
  int set_size = op_mpi_halo_exchanges_cuda(set, nargs, args);
  if (set_size > 0) {

    if ((OP_kernels[2].count==1) || (opDat0_res_calc_stride_OP2HOST != getSetSizeFromOpArg(&arg0))) {
      opDat0_res_calc_stride_OP2HOST = getSetSizeFromOpArg(&arg0);
      cudaMemcpyToSymbol(opDat0_res_calc_stride_OP2CONSTANT, &opDat0_res_calc_stride_OP2HOST,sizeof(int));
    }
    if ((OP_kernels[2].count==1) || (opDat2_res_calc_stride_OP2HOST != getSetSizeFromOpArg(&arg2))) {
      opDat2_res_calc_stride_OP2HOST = getSetSizeFromOpArg(&arg2);
      cudaMemcpyToSymbol(opDat2_res_calc_stride_OP2CONSTANT, &opDat2_res_calc_stride_OP2HOST,sizeof(int));
    }
    //set CUDA execution parameters
    #ifdef OP_BLOCK_SIZE_2
      int nthread = OP_BLOCK_SIZE_2;
    #else
      int nthread = OP_block_size;
    #endif

    for ( int round=0; round<2; round++ ){
      if (round==1) {
        op_mpi_wait_all_cuda(nargs, args);
      }
      int start = round==0 ? 0 : set->core_size;
      int end = round==0 ? set->core_size : set->size + set->exec_size;
      if (end-start>0) {
        int nblocks = (end-start-1)/nthread+1;
        op_cuda_res_calc<<<nblocks,nthread>>>(
        (double *)arg0.data_d,
        (double *)arg2.data_d,
        (double *)arg4.data_d,
        (double *)arg6.data_d,
        arg0.map_data_d,
        arg2.map_data_d,
        start,end,set->size+set->exec_size);
      }
    }
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
  //update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[2].time     += wall_t2 - wall_t1;
}
