#define ROUND_UP(bytes) (((bytes) + 15) & ~15)
// host stub function

void op_par_loop_update(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1,
  op_arg arg2,
  op_arg arg3,
  op_arg arg4 ){

  float *arg4h = (float *)arg4.data;

  int    nargs   = 5;
  op_arg args[5];

  args[0] = arg0;
  args[1] = arg1;
  args[2] = arg2;
  args[3] = arg3;
  args[4] = arg4;

  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  update\n");
  }

  op_mpi_halo_exchanges(set, nargs, args);

  // initialise timers

  double cpu_t1, cpu_t2, wall_t1=0, wall_t2=0;
  op_timing_realloc(4);
  OP_kernels[4].name      = name;
  OP_kernels[4].count    += 1;

  if (set->size >0) {

    op_timers_core(&cpu_t1, &wall_t1);

    // set CUDA execution parameters

    #ifdef OP_BLOCK_SIZE_4
      int nthread = OP_BLOCK_SIZE_4;
    #else
      //int nthread = OP_block_size;
      int nthread = OP_part_size;
      //int nthread = 256;
    #endif



      // For CPU OpenCL divide it by 4 (SSE) or by 8 (AVX)
      //nthread /= 4;





    //int nblocks = 200;
    int nblocks = 1+(set->size-1)/nthread;

    // transfer global reduction data to GPU
//    int maxblocks = nblocks;
//    int reduct_bytes = 0;
//    int reduct_size  = 0;
//    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(float)*64);
//    reduct_size   = MAX(reduct_size,sizeof(float)*64);
//int simd_width = (64/sizeof(float));// scalar code
//    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(float)*64*simd_width);
//    reduct_size   = MAX(reduct_size,sizeof(float)*64*simd_width);
//
//
//    reallocReductArrays(reduct_bytes);
//    reduct_bytes = 0;
//    arg4.data   = OP_reduct_h + reduct_bytes;
//    arg4.data_d = OP_reduct_d + reduct_bytes;
//    for (int b=0; b<maxblocks; b++) {
//    	for (int d=0; d<1; d++) {
//    		((float *)arg4.data)[d+b*1*64] = ZERO_float;
//    		reduct_bytes += ROUND_UP(maxblocks*1*sizeof(float)*64);
//    		for (int lane=0; lane<simd_width; lane++) {
//    			((float *)arg4.data)[lane+d+b*1*64*simd_width] = ZERO_float;
//    		}
//    	}
//    }
//    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(float)*64*simd_width);
//
//    mvReductArraysToDevice(reduct_bytes);
    int maxblocks = nblocks;
    int reduct_bytes = 0;
    int reduct_size  = 0;
    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(float)*64);
    reduct_size   = MAX(reduct_size,sizeof(float)*64);
    reallocReductArrays(reduct_bytes);
    reduct_bytes = 0;
    arg4.data   = OP_reduct_h + reduct_bytes;
    arg4.data_d = OP_reduct_d + reduct_bytes;
    for (int b=0; b<maxblocks; b++) {
    	for (int d=0; d<1; d++) {
    		((float *)arg4.data)[d+b*1*64] = ZERO_float;
    	}
    }
    reduct_bytes += ROUND_UP(maxblocks*1*sizeof(float)*64);
    mvReductArraysToDevice(reduct_bytes);


//    size_t nblocks[3] = {
//        Plan->ncolblk[col] >= (1<<16) ? 65535 : Plan->ncolblk[col],
//        Plan->ncolblk[col] >= (1<<16) ? (Plan->ncolblk[col]-1)/65535+1: 1,
//        1 };
//
    size_t globalWorkSize = nblocks*nthread;
    size_t localWorkSize = nthread;

    clSafeCall( clSetKernelArg(OP_opencl_core.kernel[4], 0, sizeof(cl_mem), (void*) &arg0.data_d) );
    clSafeCall( clSetKernelArg(OP_opencl_core.kernel[4], 1, sizeof(cl_mem), (void*) &arg1.data_d) ); // int array is on device; Plan->ind_map was casted to cl_mem previously in op_plan_get
    clSafeCall( clSetKernelArg(OP_opencl_core.kernel[4], 2, sizeof(cl_mem), (void*) &arg2.data_d) );
    clSafeCall( clSetKernelArg(OP_opencl_core.kernel[4], 3, sizeof(cl_mem), (void*) &arg3.data_d) );
    clSafeCall( clSetKernelArg(OP_opencl_core.kernel[4], 4, sizeof(cl_mem), (void*) &arg4.data_d) );
    clSafeCall( clSetKernelArg(OP_opencl_core.kernel[4], 5, sizeof(cl_int), (void*) &set->size) );

    clSafeCall( clEnqueueNDRangeKernel(OP_opencl_core.command_queue, OP_opencl_core.kernel[4], 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL) );
    clSafeCall( clFlush(OP_opencl_core.command_queue) );
    clSafeCall( clFinish(OP_opencl_core.command_queue) );



//    op_cuda_update<<<nblocks,nthread,nshared>>>( (float *) arg0.data_d,
//                                                 (float *) arg1.data_d,
//                                                 (float *) arg2.data_d,
//                                                 (float *) arg3.data_d,
//                                                 (float *) arg4.data_d,
//                                                 offset_s,
//                                                 set->size );

    // transfer global reduction data back to CPU

    mvReductArraysToHost(reduct_bytes);

    //clSafeCall( clEnqueueReadBuffer(OP_opencl_core.command_queue, (cl_mem) arg4.data_d, CL_FALSE, 0, 4, arg4.data, 0, NULL, NULL) );
    //clSafeCall( clEnqueueReadBuffer(OP_opencl_core.command_queue, (cl_mem) arg4.data_d, CL_TRUE, 0, arg4.size * arg4.set->size, arg4.data, 0, NULL, NULL) );


    for (int b=0; b<maxblocks; b++) {
      for (int d=0; d<1; d++) {
       	arg4h[d] = arg4h[d] + ((float *)arg4.data)[d+b*1*64];
//      	for (int lane=0; lane<simd_width; lane++)
//          arg4h[d] = arg4h[d] + ((float *)arg4.data)[lane+d+b*1*64*simd_width];
    //    printf("((float *)arg4.data)[d+b*1*64] = %e\n",((float *)arg4.data)[d+b*1*64]);
      }
    }

  arg4.data = (char *)arg4h;

  //printf("arg4.data = %e \n",*((float*)arg4.data));
  //printf("arg4ih = %e \n",(float)arg4h[0]);

  //op_mpi_reduce(&arg4,arg4h);

  }


  //op_mpi_set_dirtybit(nargs, args);

  // update kernel record

  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[4].time     += wall_t2 - wall_t1;
  OP_kernels[4].transfer += (float)set->size * arg0.size;
  OP_kernels[4].transfer += (float)set->size * arg1.size;
  OP_kernels[4].transfer += (float)set->size * arg2.size * 2.0f;
  OP_kernels[4].transfer += (float)set->size * arg3.size;
//  op_printf("op_par_loop_update() ran \n");
}

