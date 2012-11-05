  void op_par_loop_save_soln(char const *name,
      op_set set,
      op_arg arg0,
      op_arg arg1) {

    buildOpenCLKernels();

    int    nargs   = 2;
    op_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

    if (OP_diags>2) {
      printf(" kernel routine w/o indirection:  save_soln\n");
    }

    op_mpi_halo_exchanges(set, nargs, args);

    // initialise timers

    double cpu_t1, cpu_t2, wall_t1=0, wall_t2=0;
    op_timing_realloc(0);
    OP_kernels[0].name      = name;
    OP_kernels[0].count    += 1;

    if (set->size >0) {

      op_timers_core(&cpu_t1, &wall_t1);

      // set OpenCL execution parameters

      //     #ifdef OP_BLOCK_SIZE_0
      //       int nthread = OP_BLOCK_SIZE_0;
      //     #else
      //       // int nthread = OP_block_size;
      //       int nthread = 128;
      //     #endif

      //     int nblocks = 200;

      // work out local memory requirements per element

      int nlocal = 0;
      nlocal = MAX(nlocal,sizeof(cl_float)*4);
      nlocal = MAX(nlocal,sizeof(cl_float)*4);

      // execute plan

      int offset_s = nlocal*OP_WARPSIZE;

      size_t localWorkSize = 128;
//      size_t globalWorkSize = set->size;
      size_t globalWorkSize = (set->size/(int)localWorkSize) * (int)localWorkSize;
      globalWorkSize += ( (set->size % localWorkSize) > 0 ? localWorkSize : 0);
      globalWorkSize = globalWorkSize < localWorkSize ? localWorkSize : globalWorkSize;

      nlocal = nlocal*localWorkSize;

      clSafeCall( clSetKernelArg(OP_opencl_core.kernel[0], 0, sizeof(cl_mem), (void *) &arg0.data_d) );
      clSafeCall( clSetKernelArg(OP_opencl_core.kernel[0], 1, sizeof(cl_mem), (void *) &arg1.data_d) );
      clSafeCall( clSetKernelArg(OP_opencl_core.kernel[0], 2, sizeof(cl_int), (void *) &offset_s) );
      clSafeCall( clSetKernelArg(OP_opencl_core.kernel[0], 3, sizeof(cl_int), (void *) &(set->size)) );
      clSafeCall( clSetKernelArg(OP_opencl_core.kernel[0], 4, nlocal, NULL) );

      clSafeCall( clEnqueueNDRangeKernel(OP_opencl_core.command_queue, OP_opencl_core.kernel[0], 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL) );
      clSafeCall( clFlush(OP_opencl_core.command_queue) );
      clSafeCall( clFinish(OP_opencl_core.command_queue) );
    }

    op_mpi_set_dirtybit(nargs, args);

    // update kernel record

    op_timers_core(&cpu_t2, &wall_t2);
    OP_kernels[0].time     += wall_t2 - wall_t1;
    OP_kernels[0].transfer += (float)set->size * arg0.size;
    OP_kernels[0].transfer += (float)set->size * arg1.size;
    op_printf("op_par_loop_save_soln() ran \n");
  }
