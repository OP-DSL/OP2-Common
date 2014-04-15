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

  int set_size = op_mpi_halo_exchanges(set, nargs, args);

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

      if (col==Plan->ncolors_core) op_mpi_wait_all(nargs,args);

    #ifdef OP_BLOCK_SIZE_3
      int nthread = OP_BLOCK_SIZE_3;
    #else
      int nthread = OP_block_size;
      //int nthread = 128;
    #endif

      size_t nblocks[3] = {
          Plan->ncolblk[col] >= (1<<16) ? 65535 : Plan->ncolblk[col],
          Plan->ncolblk[col] >= (1<<16) ? (Plan->ncolblk[col]-1)/65535+1: 1,
          1 };

      size_t globalWorkSize[3] = {nblocks[0]*nthread, nblocks[1], nblocks[2]};
      size_t localWorkSize[3] = {nthread, 1, 1};

      if (Plan->ncolblk[col] > 0) {
        int nshared = Plan->nsharedCol[col];
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[3], 0, sizeof(cl_mem), (void*) &arg0.data_d) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[3], 1, sizeof(cl_mem), (void*) &arg2.data_d) ); 
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[3], 2, sizeof(cl_mem), (void*) &arg3.data_d) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[3], 3, sizeof(cl_mem), (void*) &arg4.data_d) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[3], 4, sizeof(cl_mem), (void*) &arg0.map_data_d) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[3], 5, sizeof(cl_mem), (void*) &arg2.map_data_d) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[3], 6, sizeof(cl_mem), (void*) &arg5.data_d) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[3], 7, sizeof(cl_int), (void*) &block_offset) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[3], 8, sizeof(cl_mem), (void*) &Plan->blkmap) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[3], 9, sizeof(cl_mem), (void*) &Plan->offset) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[3],10, sizeof(cl_mem), (void*) &Plan->nelems) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[3],11, sizeof(cl_mem), (void*) &Plan->nthrcol) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[3],12, sizeof(cl_mem), (void*) &Plan->thrcol) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[3],13, sizeof(cl_int), (void*) &Plan->ncolblk[col]) ); // int array is on host
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[3],14, sizeof(cl_int), (void*) &set_size) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[3],15, sizeof(cl_mem), (void*) &OP_opencl_core.constant[1]) ); // gm1
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[3],16, sizeof(cl_mem), (void*) &OP_opencl_core.constant[3]) ); // eps
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[3],17, sizeof(cl_mem), (void*) &OP_opencl_core.constant[6]) ); // qinf[4]

        clSafeCall( clEnqueueNDRangeKernel(OP_opencl_core.command_queue, OP_opencl_core.kernel[3], 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL) );
        clSafeCall( clFlush(OP_opencl_core.command_queue) );
        clSafeCall( clFinish(OP_opencl_core.command_queue) );


//        op_cuda_bres_calc<<<nblocks,nthread,nshared>>>(
//           (double *)arg0.data_d,
//           (double *)arg2.data_d,
//           (double *)arg3.data_d,
//           (double *)arg4.data_d,
//           Plan->ind_map,
//           Plan->loc_map,
//           (int *)arg5.data_d,
//           Plan->ind_sizes,
//           Plan->ind_offs,
//           block_offset,
//           Plan->blkmap,
//           Plan->offset,
//           Plan->nelems,
//           Plan->nthrcol,
//           Plan->thrcol,
//           Plan->ncolblk[col],
//           set_size);
//
//        cutilSafeCall(cudaThreadSynchronize());
//        cutilCheckMsg("op_cuda_bres_calc execution failed\n");
      }

      block_offset += Plan->ncolblk[col];
    }

    op_timing_realloc(3);
    OP_kernels[3].transfer  += Plan->transfer;
    OP_kernels[3].transfer2 += Plan->transfer2;

  }


  op_mpi_set_dirtybit(nargs, args);

  // update kernel record

  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[3].time     += wall_t2 - wall_t1;
//  op_printf("op_par_loop_bres_calc() ran \n");
}

