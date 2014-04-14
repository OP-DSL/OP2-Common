// host stub function

//#include <iacaMarks.h>
void op_par_loop_res_calc(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1,
  op_arg arg2,
  op_arg arg3,
  op_arg arg4,
  op_arg arg5,
  op_arg arg6,
  op_arg arg7 ){

  buildOpenCLKernels();

  int    nargs   = 8;
  op_arg args[8];

  args[0] = arg0;
  args[1] = arg1;
  args[2] = arg2;
  args[3] = arg3;
  args[4] = arg4;
  args[5] = arg5;
  args[6] = arg6;
  args[7] = arg7;

  int    ninds   = 4;
  int    inds[8] = {0,0,1,1,2,2,3,3};

  if (OP_diags>2) {
    printf(" kernel routine with indirection: res_calc\n");
  }

  // get plan

  #ifdef OP_PART_SIZE_2
    int part_size = OP_PART_SIZE_2;
  #else
    int part_size = OP_part_size;
    //int part_size = OP_block_size;
  #endif

  int set_size = op_mpi_halo_exchanges(set, nargs, args);

  // initialise timers

  double cpu_t1, cpu_t2, wall_t1=0, wall_t2=0;
  op_timing_realloc(2);
  OP_kernels[2].name      = name;
  OP_kernels[2].count    += 1;

  if (set->size >0) {

    op_plan *Plan = op_plan_get(name,set,part_size,nargs,args,ninds,inds);

    op_timers_core(&cpu_t1, &wall_t1);

    // execute plan
    int block_offset = 0;
    for (int col=0; col < Plan->ncolors; col++) {
      //if (col==Plan->ncolors_core) op_mpi_wait_all(nargs,args);

#ifdef OP_BLOCK_SIZE_2
      int nthread = OP_BLOCK_SIZE_2;
#else
      //int nthread = OP_block_size;
      int nthread = OP_part_size;
#endif
      // For CPU OpenCL divide it by 4 (SSE) or by 8 (AVX)
//      nthreads /= 4;

      size_t nblocks[3] = {
          Plan->ncolblk[col] >= (1<<16) ? 65535 : Plan->ncolblk[col],
          Plan->ncolblk[col] >= (1<<16) ? (Plan->ncolblk[col]-1)/65535+1: 1,
          1 };
      size_t globalWorkSize[3] = {nblocks[0]*nthread, nblocks[1], nblocks[2]};
      size_t localWorkSize[3] = {nthread, 1, 1};
      if (Plan->ncolblk[col] > 0) {
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[2], 0, sizeof(cl_mem), (void*) &arg0.data_d) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[2], 1, sizeof(cl_mem), (void*) &arg2.data_d) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[2], 2, sizeof(cl_mem), (void*) &arg4.data_d) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[2], 3, sizeof(cl_mem), (void*) &arg6.data_d) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[2], 4, sizeof(cl_mem), (void*) &arg0.map_data_d) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[2], 5, sizeof(cl_mem), (void*) &arg2.map_data_d) ); 
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[2], 6, sizeof(cl_mem), (void*) &arg4.map_data_d) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[2], 7, sizeof(cl_mem), (void*) &arg6.map_data_d) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[2], 8, sizeof(cl_int), (void*) &block_offset) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[2], 9, sizeof(cl_mem), (void*) &Plan->blkmap) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[2],10, sizeof(cl_mem), (void*) &Plan->offset) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[2],11, sizeof(cl_mem), (void*) &Plan->nelems) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[2],12, sizeof(cl_mem), (void*) &Plan->nthrcol) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[2],13, sizeof(cl_mem), (void*) &Plan->thrcol) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[2],14, sizeof(cl_int), (void*) &Plan->ncolblk[col]) ); 
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[2],15, sizeof(cl_int), (void*) &set_size) );

        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[2],16, sizeof(cl_mem), (void*) &OP_opencl_core.constant[1]) ); // gm1
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[2],17, sizeof(cl_mem), (void*) &OP_opencl_core.constant[3]) ); // eps

        // Start kernel and flush queue
        clSafeCall( clEnqueueNDRangeKernel(OP_opencl_core.command_queue, OP_opencl_core.kernel[2], 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL) );
        clSafeCall( clFlush(OP_opencl_core.command_queue) );
        clSafeCall( clFinish(OP_opencl_core.command_queue) );
      }

      block_offset += Plan->ncolblk[col];
    }

    op_timing_realloc(2);
    OP_kernels[2].transfer  += Plan->transfer;
    OP_kernels[2].transfer2 += Plan->transfer2;

  }

  //op_mpi_set_dirtybit(nargs, args);

  // update kernel record

  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[2].time     += wall_t2 - wall_t1;
//  op_printf("op_par_loop_res_calc() ran \n");
}
