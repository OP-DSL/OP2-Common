// host stub function
void op_par_loop_adt_calc(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1,
  op_arg arg2,
  op_arg arg3,
  op_arg arg4,
  op_arg arg5 ){

  buildOpenCLKernels();

  int    nargs   = 6;
  op_arg args[6];

  args[0] = arg0;
  args[1] = arg1;
  args[2] = arg2;
  args[3] = arg3;
  args[4] = arg4;
  args[5] = arg5;

  int    ninds   = 1;
  int    inds[6] = {0,0,0,0,-1,-1};

  if (OP_diags>2) {
    printf(" kernel routine with indirection: adt_calc\n");
  }

  // get plan

  #ifdef OP_PART_SIZE_1
    int part_size = OP_PART_SIZE_1;
  #else
    int part_size = OP_part_size;
  #endif

  int set_size = op_mpi_halo_exchanges(set, nargs, args);

  // initialise timers

  double cpu_t1, cpu_t2, wall_t1=0, wall_t2=0;
  op_timing_realloc(1);
  OP_kernels[1].name      = name;
  OP_kernels[1].count    += 1;

  if (set->size >0) {

    op_plan *Plan = op_plan_get(name,set,part_size,nargs,args,ninds,inds);

    op_timers_core(&cpu_t1, &wall_t1);

    // execute plan

    int block_offset = 0;

    for (int col=0; col < Plan->ncolors; col++) {

      if (col==Plan->ncolors_core) op_mpi_wait_all(nargs,args);

    #ifdef OP_BLOCK_SIZE_1
      size_t nthread = OP_BLOCK_SIZE_1;
    #else
      size_t nthread = OP_part_size;
      //size_t nthread = OP_block_size;
      //size_t nthread = 256;
    #endif


//      op_printf("part_size = %d\n",part_size);
//      op_printf("block_size = %d\n",nthread);


      size_t nblocks[3] = {
          Plan->ncolblk[col] >= (1<<16) ? 65535 : Plan->ncolblk[col],
          Plan->ncolblk[col] >= (1<<16) ? (Plan->ncolblk[col]-1)/65535+1: 1,
          1 };

      size_t globalWorkSize[3] = {nblocks[0]*nthread, nblocks[1], nblocks[2]};
      size_t localWorkSize[3] = {nthread, 1, 1};

//      printf("global: %d %d %d\n",globalWorkSize[0],globalWorkSize[1],globalWorkSize[2]);
//      printf("local: %d %d %d\n",localWorkSize[0],localWorkSize[1],localWorkSize[2]);
//      dim3 nblocks = dim3(Plan->ncolblk[col] >= (1<<16) ? 65535 : Plan->ncolblk[col],
//                      Plan->ncolblk[col] >= (1<<16) ? (Plan->ncolblk[col]-1)/65535+1: 1, 1);
      if (Plan->ncolblk[col] > 0) {
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[1], 0, sizeof(cl_mem), (void*) &arg0.data_d) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[1], 1, sizeof(cl_mem), (void*) &arg0.map_data_d) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[1], 2, sizeof(cl_mem), (void*) &arg4.data_d) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[1], 3, sizeof(cl_mem), (void*) &arg5.data_d) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[1], 4, sizeof(cl_int), (void*) &block_offset) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[1], 5, sizeof(cl_mem), (void*) &Plan->blkmap) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[1], 6, sizeof(cl_mem), (void*) &Plan->offset) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[1], 7, sizeof(cl_mem), (void*) &Plan->nelems) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[1], 8, sizeof(cl_mem), (void*) &Plan->nthrcol) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[1], 9, sizeof(cl_mem), (void*) &Plan->thrcol) );
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[1],10, sizeof(cl_int), (void*) &Plan->ncolblk[col]) ); // int array is on host
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[1],11, sizeof(cl_int), (void*) &set_size) );

//        cl_int ret;
//
//        OP_opencl_core.constant = (cl_mem*) malloc((OP_opencl_core.n_constants)*sizeof(cl_mem));
//        OP_opencl_core.constant[0] = clCreateBuffer(OP_opencl_core.context, CL_MEM_READ_ONLY, 4, NULL, &ret);
//        OP_opencl_core.constant[1] = clCreateBuffer(OP_opencl_core.context, CL_MEM_READ_ONLY, 4, NULL, &ret);
//        OP_opencl_core.constant[2] = clCreateBuffer(OP_opencl_core.context, CL_MEM_READ_ONLY, 4, NULL, &ret);
//        double a=1.4;
//        clSafeCall( clEnqueueWriteBuffer(OP_opencl_core.command_queue, OP_opencl_core.constant[0], CL_TRUE, 0, 4, (void*) &a, 0, NULL, NULL) );
//        a=1.0;
//        clSafeCall( clEnqueueWriteBuffer(OP_opencl_core.command_queue, OP_opencl_core.constant[1], CL_TRUE, 0, 4, (void*) &a, 0, NULL, NULL) );
//        a=2.0;
//        clSafeCall( clEnqueueWriteBuffer(OP_opencl_core.command_queue, OP_opencl_core.constant[2], CL_TRUE, 0, 4, (void*) &a, 0, NULL, NULL) );
//
//        clSafeCall( clFlush(OP_opencl_core.command_queue) );
//        clSafeCall( clFinish(OP_opencl_core.command_queue) );
//
////        for(int i=0; i < OP_opencl_core.n_constants; i++) {
////          printf("cl_mem %d = %d \n", i, &OP_opencl_core.constant[i]);
////        }

        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[1],12, sizeof(cl_mem), (void*) &OP_opencl_core.constant[0]) ); // gam
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[1],13, sizeof(cl_mem), (void*) &OP_opencl_core.constant[1]) ); // gm1
        clSafeCall( clSetKernelArg(OP_opencl_core.kernel[1],14, sizeof(cl_mem), (void*) &OP_opencl_core.constant[2]) ); // cfl

        clSafeCall( clEnqueueNDRangeKernel(OP_opencl_core.command_queue, OP_opencl_core.kernel[1], 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL) );
        clSafeCall( clFlush(OP_opencl_core.command_queue) );
        clSafeCall( clFinish(OP_opencl_core.command_queue) );
      }

      block_offset += Plan->ncolblk[col];
    }

    op_timing_realloc(1);
    OP_kernels[1].transfer  += Plan->transfer;
    OP_kernels[1].transfer2 += Plan->transfer2;

  }


  //op_mpi_set_dirtybit(nargs, args);

  // update kernel record

  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[1].time     += wall_t2 - wall_t1;

//  for(int i=0;i < set->size; i++)
//    ((double*)(arg5.dat->data))[i] = 1.1f;
//  op_printf("op_par_loop_adt_calc() ran \n");
}

