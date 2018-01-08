//
// auto-generated by op2.py
//

//user function
//user function
//#pragma acc routine
inline void update( double *data, int *count) {
  data[0] = 0.0;
  (*count)++;
}

// host stub function
void op_par_loop_update(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1){

  int*arg1h = (int *)arg1.data;
  int nargs = 2;
  op_arg args[2];

  args[0] = arg0;
  args[1] = arg1;

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(1);
  op_timers_core(&cpu_t1, &wall_t1);
  OP_kernels[1].name      = name;
  OP_kernels[1].count    += 1;


  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  update");
  }

  op_mpi_halo_exchanges_cuda(set, nargs, args);

  int arg1_l = arg1h[0];

  if (set->size >0) {


    //Set up typed device pointers for OpenACC

    double* data0 = (double*)arg0.data_d;
    #pragma acc parallel loop independent deviceptr(data0) reduction(+:arg1_l)
    for ( int n=0; n<set->size; n++ ){
      update(&data0[4 * n], &arg1_l);
    }
  }

  // combine reduction data
  arg1h[0] = arg1_l;
  op_mpi_reduce_int(&arg1,arg1h);
  op_mpi_set_dirtybit_cuda(nargs, args);

  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[1].time     += wall_t2 - wall_t1;
  OP_kernels[1].transfer += (float)set->size * arg0.size * 2.0f;
}
