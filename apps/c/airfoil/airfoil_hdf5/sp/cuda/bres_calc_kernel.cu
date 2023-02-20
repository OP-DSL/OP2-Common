//
// auto-generated by op2.py
//

//user function
__device__ void bres_calc_gpu( const float *x1, const float *x2, const float *q1,
                      const float *adt1, float *res1, const int *bound) {
  float dx, dy, mu, ri, p1, vol1, p2, vol2, f;

  dx = x1[0] - x2[0];
  dy = x1[1] - x2[1];

  ri = 1.0f / q1[0];
  p1 = gm1_cuda * (q1[3] - 0.5f * ri * (q1[1] * q1[1] + q1[2] * q1[2]));

  vol1 = ri * (q1[1] * dy - q1[2] * dx);

  ri = 1.0f / qinf_cuda[0];
  p2 = gm1_cuda * (qinf_cuda[3] - 0.5f * ri * (qinf_cuda[1] * qinf_cuda[1] + qinf_cuda[2] * qinf_cuda[2]));
  vol2 = ri * (qinf_cuda[1] * dy - qinf_cuda[2] * dx);

  mu = (*adt1) * eps_cuda;

  f = 0.5f * (vol1 * q1[0] + vol2 * qinf_cuda[0]) + mu * (q1[0] - qinf_cuda[0]);
  res1[0] += *bound == 1 ? 0.0f : f;
  f = 0.5f * (vol1 * q1[1] + p1 * dy + vol2 * qinf_cuda[1] + p2 * dy) +
      mu * (q1[1] - qinf_cuda[1]);
  res1[1] += *bound == 1 ? p1 * dy : f;
  f = 0.5f * (vol1 * q1[2] - p1 * dx + vol2 * qinf_cuda[2] - p2 * dx) +
      mu * (q1[2] - qinf_cuda[2]);
  res1[2] += *bound == 1 ? -p1 * dx : f;
  f = 0.5f * (vol1 * (q1[3] + p1) + vol2 * (qinf_cuda[3] + p2)) +
      mu * (q1[3] - qinf_cuda[3]);
  res1[3] += *bound == 1 ? 0.0f : f;

}

// CUDA kernel function
__global__ void op_cuda_bres_calc(
  const float *__restrict ind_arg0,
  const float *__restrict ind_arg1,
  const float *__restrict ind_arg2,
  float *__restrict ind_arg3,
  const int *__restrict opDat0Map,
  const int *__restrict opDat2Map,
  const int *__restrict arg5,
  int start,
  int end,
  int   set_size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid + start < end) {
    int n = tid + start;
    //initialise local variables
    float arg4_l[4];
    for ( int d=0; d<4; d++ ){
      arg4_l[d] = ZERO_float;
    }
    int map0idx;
    int map1idx;
    int map2idx;
    map0idx = opDat0Map[n + set_size * 0];
    map1idx = opDat0Map[n + set_size * 1];
    map2idx = opDat2Map[n + set_size * 0];

    //user-supplied kernel call
    bres_calc_gpu(ind_arg0+map0idx*2,
              ind_arg0+map1idx*2,
              ind_arg1+map2idx*4,
              ind_arg2+map2idx*1,
              arg4_l,
              arg5+n*1);
    atomicAdd(&ind_arg3[0+map2idx*4],arg4_l[0]);
    atomicAdd(&ind_arg3[1+map2idx*4],arg4_l[1]);
    atomicAdd(&ind_arg3[2+map2idx*4],arg4_l[2]);
    atomicAdd(&ind_arg3[3+map2idx*4],arg4_l[3]);
  }
}


//host stub function
void op_par_loop_bres_calc(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1,
  op_arg arg2,
  op_arg arg3,
  op_arg arg4,
  op_arg arg5){

  int nargs = 6;
  op_arg args[6];

  args[0] = arg0;
  args[1] = arg1;
  args[2] = arg2;
  args[3] = arg3;
  args[4] = arg4;
  args[5] = arg5;

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(3);
  op_timers_core(&cpu_t1, &wall_t1);
  OP_kernels[3].name      = name;
  OP_kernels[3].count    += 1;


  int    ninds   = 4;
  int    inds[6] = {0,0,1,2,3,-1};

  if (OP_diags>2) {
    printf(" kernel routine with indirection: bres_calc\n");
  }
  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2);
  if (set_size > 0) {

    //set CUDA execution parameters
    #ifdef OP_BLOCK_SIZE_3
      int nthread = OP_BLOCK_SIZE_3;
    #else
      int nthread = OP_block_size;
    #endif

    for ( int round=0; round<2; round++ ){
      if (round==1) {
        op_mpi_wait_all_grouped(nargs, args, 2);
      }
      int start = round==0 ? 0 : set->core_size;
      int end = round==0 ? set->core_size : set->size + set->exec_size;
      if (end-start>0) {
        int nblocks = (end-start-1)/nthread+1;
        op_cuda_bres_calc<<<nblocks,nthread>>>(
        (float *)arg0.data_d,
        (float *)arg2.data_d,
        (float *)arg3.data_d,
        (float *)arg4.data_d,
        arg0.map_data_d,
        arg2.map_data_d,
        (int*)arg5.data_d,
        start,end,set->size+set->exec_size);
      }
    }
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
  //update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[3].time     += wall_t2 - wall_t1;
}
