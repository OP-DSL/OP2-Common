/*
 * Open source copyright declaration based on BSD open source template:
 * http://www.opensource.org/licenses/bsd-license.php
 *
 * This file is part of the OP2 distribution.
 *
 * Copyright (c) 2011, Mike Giles and others. Please see the AUTHORS file in
 * the main source directory for a full list of copyright holders.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * The name of Mike Giles may not be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//
// This file implements the CUDA-specific run-time support functions
//

//
// header files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif
//#include <cuda_runtime_api.h>
//#include <math_constants.h>

#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_rt_support.h>
#include <op_opencl_rt_support.h>
#include <op_opencl_core.h>

// Small re-declaration to avoid using struct in the C version.
// This is due to the different way in which C and C++ see structs

//typedef struct cudaDeviceProp cudaDeviceProp_t;

// arrays for global constants and reductions

int OP_consts_bytes = 0,
    OP_reduct_bytes = 0;

char * OP_consts_h,
     * OP_consts_d,
     * OP_reduct_h,
     * OP_reduct_d;

op_opencl_core OP_opencl_core;

//
// Get return (error) messages from OpenCL run-time
//
char *clGetErrorString(cl_int err) {
  switch (err) {
  case CL_SUCCESS: return (char *) "Success!";
  case CL_DEVICE_NOT_FOUND: return (char *) "Device not found.";
  case CL_DEVICE_NOT_AVAILABLE: return (char *) "Device not available";
  case CL_COMPILER_NOT_AVAILABLE: return (char *) "Compiler not available";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE: return (char *) "Memory object allocation failure";
  case CL_OUT_OF_RESOURCES: return (char *) "Out of resources";
  case CL_OUT_OF_HOST_MEMORY: return (char *) "Out of host memory";
  case CL_PROFILING_INFO_NOT_AVAILABLE: return (char *) "Profiling information not available";
  case CL_MEM_COPY_OVERLAP: return (char *) "Memory copy overlap";
  case CL_IMAGE_FORMAT_MISMATCH: return (char *) "Image format mismatch";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED: return (char *) "Image format not supported";
  case CL_BUILD_PROGRAM_FAILURE: return (char *) "Program build failure";
  case CL_MAP_FAILURE: return (char *) "Map failure";
  case CL_MISALIGNED_SUB_BUFFER_OFFSET: return (char*) "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return (char*) "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case CL_INVALID_VALUE: return (char *) "Invalid value";
  case CL_INVALID_DEVICE_TYPE: return (char *) "Invalid device type";
  case CL_INVALID_PLATFORM: return (char *) "Invalid platform";
  case CL_INVALID_DEVICE: return (char *) "Invalid device";
  case CL_INVALID_CONTEXT: return (char *) "Invalid context";
  case CL_INVALID_QUEUE_PROPERTIES: return (char *) "Invalid queue properties";
  case CL_INVALID_COMMAND_QUEUE: return (char *) "Invalid command queue";
  case CL_INVALID_HOST_PTR: return (char *) "Invalid host pointer";
  case CL_INVALID_MEM_OBJECT: return (char *) "Invalid memory object";
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return (char *) "Invalid image format descriptor";
  case CL_INVALID_IMAGE_SIZE: return (char *) "Invalid image size";
  case CL_INVALID_SAMPLER: return (char *) "Invalid sampler";
  case CL_INVALID_BINARY: return (char *) "Invalid binary";
  case CL_INVALID_BUILD_OPTIONS: return (char *) "Invalid build options";
  case CL_INVALID_PROGRAM: return (char *) "Invalid program";
  case CL_INVALID_PROGRAM_EXECUTABLE: return (char *) "Invalid program executable";
  case CL_INVALID_KERNEL_NAME: return (char *) "Invalid kernel name";
  case CL_INVALID_KERNEL_DEFINITION: return (char *) "Invalid kernel definition";
  case CL_INVALID_KERNEL: return (char *) "Invalid kernel";
  case CL_INVALID_ARG_INDEX: return (char *) "Invalid argument index";
  case CL_INVALID_ARG_VALUE: return (char *) "Invalid argument value";
  case CL_INVALID_ARG_SIZE: return (char *) "Invalid argument size";
  case CL_INVALID_KERNEL_ARGS: return (char *) "Invalid kernel arguments";
  case CL_INVALID_WORK_DIMENSION: return (char *) "Invalid work dimension";
  case CL_INVALID_WORK_GROUP_SIZE: return (char *) "Invalid work group size";
  case CL_INVALID_WORK_ITEM_SIZE: return (char *) "Invalid work item size";
  case CL_INVALID_GLOBAL_OFFSET: return (char *) "Invalid global offset";
  case CL_INVALID_EVENT_WAIT_LIST: return (char *) "Invalid event wait list";
  case CL_INVALID_EVENT: return (char *) "Invalid event";
  case CL_INVALID_OPERATION: return (char *) "Invalid operation";
  case CL_INVALID_GL_OBJECT: return (char *) "Invalid OpenGL object";
  case CL_INVALID_BUFFER_SIZE: return (char *) "Invalid buffer size";
  case CL_INVALID_MIP_LEVEL: return (char *) "Invalid mip-map level";
  case CL_INVALID_GLOBAL_WORK_SIZE: return (char *) "Invalid global work size";
  case CL_INVALID_PROPERTY: return (char *) "Invalid property";
  default: return (char *) "Unknown";
  }
//  if(err != CL_SUCCESS) {
//    char* build_log;
//    size_t log_size;
//    clGetProgramBuildInfo(OP_opencl_core.program, OP_opencl_core.device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
//    build_log = (char*) malloc(log_size+1);
//    clGetProgramBuildInfo(OP_opencl_core.program, OP_opencl_core.device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
//    build_log[log_size] = '\0';
//    fprintf(stderr, "=============== OpenCL Program Build Info ================= \n%s", build_log);
//    fprintf(stderr, "\n=========================================================== \n");
//    free(build_log);
//    exit(EXIT_FAILURE);
//  }
}

//
// CUDA utility functions
//

//void __cudaSafeCall ( cudaError_t err, const char * file, const int line )
//{
//  if ( cudaSuccess != err ) {
//    fprintf ( stderr, "%s(%i) : cutilSafeCall() Runtime API error : %s.\n",
//              file, line, cudaGetErrorString ( err ) );
//    exit ( -1 );
//  }
//}
//
//void __cutilCheckMsg ( const char * errorMessage, const char * file,
//                       const int line )
//{
//  cudaError_t err = cudaGetLastError (  );
//  if ( cudaSuccess != err ) {
//    fprintf ( stderr, "%s(%i) : cutilCheckMsg() error : %s : %s.\n",
//              file, line, errorMessage, cudaGetErrorString ( err ) );
//    exit ( -1 );
//  }
//}

void __clSafeCall(cl_int ret, const char * file, const int line) {
  if ( CL_SUCCESS != ret ) {
    fprintf ( stderr, "%s(%i) : clSafeCall() Runtime API error : %s.\n",
        file, line, clGetErrorString ( ret ) );
    exit ( -1 );
  }
}

void openclDeviceInit( int argc, char ** argv )
{
  (void)argc;
  (void)argv;

  cl_int ret;
  char buffer[10240];
  cl_bool available = false;
  float *test_h1 = (float*) malloc(sizeof(float));
  float *test_h2 = (float*) malloc(sizeof(float));
  cl_mem test_d = NULL;

  // Get platform and device information
  OP_opencl_core.platform_id = NULL;
  OP_opencl_core.device_id = NULL;

  clSafeCall( clGetPlatformIDs(0, NULL, &OP_opencl_core.n_platforms) );
  printf("num_platforms = %i \n",(int) OP_opencl_core.n_platforms);
  OP_opencl_core.platform_id = (cl_platform_id*) malloc( OP_opencl_core.n_platforms*sizeof(cl_uint) );
  clSafeCall( clGetPlatformIDs( OP_opencl_core.n_platforms, OP_opencl_core.platform_id, NULL) );

// switch(device_type) {
//  case CPU:
//    ret = clGetDeviceIDs(device.platform_id[0], CL_DEVICE_TYPE_CPU, 1, &device.device_id, &device.ret_num_devices);
//  break;
//  case GPU:
//    ret = clGetDeviceIDs(device.platform_id[1], CL_DEVICE_TYPE_GPU, 1, &device.device_id, &device.ret_num_devices);
//  break;
// }
  clSafeCall( clGetDeviceIDs(OP_opencl_core.platform_id[0], CL_DEVICE_TYPE_GPU, 4, &OP_opencl_core.device_id, &OP_opencl_core.n_devices) );
//  printf("ret clGetDeviceIDs(.,%d,...) = %d\n", device_type,ret);

  if(OP_opencl_core.n_platforms == 0 && OP_opencl_core.n_devices == 0) {
    printf("No OpenCL platform or device is available! Exiting.\n");
    exit(-1);
  }
  printf("\nNo. of device on platform = %d\n", OP_opencl_core.n_devices);

  printf("\nChoosen device: \n");
  clSafeCall( clGetDeviceInfo(OP_opencl_core.device_id, CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL) );
  printf("\nCL_DEVICE_VENDOR    = %s \n", buffer);
  clSafeCall( clGetDeviceInfo(OP_opencl_core.device_id, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL) );
  printf("\nCL_DEVICE_NAME     = %s \n", buffer);
  clSafeCall( clGetDeviceInfo(OP_opencl_core.device_id, CL_DEVICE_AVAILABLE, sizeof(cl_bool), &available, NULL) );
  printf("\nCL_DEVICE_AVAILABLE = %s \n", available ? "true" : "false");

  // Create an OpenCL context
  OP_opencl_core.context = clCreateContext( NULL, 1, &OP_opencl_core.device_id, NULL, NULL, &ret);
  clSafeCall( ret );

  // Create a command queue
  OP_opencl_core.command_queue = clCreateCommandQueue(OP_opencl_core.context, OP_opencl_core.device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
  clSafeCall( ret );

  // Make a read/write test
  test_h1[0] = 1986;
  test_d = clCreateBuffer(OP_opencl_core.context, CL_MEM_READ_WRITE, sizeof(float), NULL, &ret);
  clSafeCall( ret );
  clSafeCall( clEnqueueWriteBuffer(OP_opencl_core.command_queue, test_d, CL_TRUE, 0, sizeof(float), (void*)test_h1, 0, NULL, NULL) );
  clSafeCall( clEnqueueReadBuffer(OP_opencl_core.command_queue, test_d, CL_TRUE, 0, sizeof(float), (void*)test_h2, 0, NULL, NULL) );
  if(test_h1[0] != test_h2[0]) {
    printf("Error during buffer read/write test! Exiting \n");
    exit(-1);
  }
  clSafeCall( clReleaseMemObject(test_d) );
  clSafeCall( clFlush(OP_opencl_core.command_queue) );
  clSafeCall( clFinish(OP_opencl_core.command_queue) );

  // Number of constants in constant array
  OP_opencl_core.n_constants = 0;
}

//
// routines to move arrays to/from GPU device
//

void op_mvHostToDevice ( void ** map, int size )
{
  cl_int ret = 0;
  cl_mem tmp;
  tmp = (cl_mem) clCreateBuffer(OP_opencl_core.context, CL_MEM_READ_WRITE, size, NULL, &ret);
  clSafeCall( ret );
  clSafeCall( clEnqueueWriteBuffer(OP_opencl_core.command_queue, (cl_mem) tmp, CL_TRUE, 0, size, *map, 0, NULL, NULL) );
  clSafeCall( clFlush(OP_opencl_core.command_queue) );
  clSafeCall( clFinish(OP_opencl_core.command_queue) );
  free ( *map );
  *map = (void*)tmp;
//  void *tmp;
//  cutilSafeCall ( cudaMalloc ( &tmp, size ) );
//  cutilSafeCall ( cudaMemcpy ( tmp, *map, size,
//                               cudaMemcpyHostToDevice ) );
//  cutilSafeCall ( cudaThreadSynchronize (  ) );
//  free ( *map );
//  *map = tmp;
}

void op_cpHostToDevice ( void ** data_d, void ** data_h, int size )
{
  cl_int ret = 0;
  *data_d = (cl_mem) clCreateBuffer(OP_opencl_core.context, CL_MEM_READ_WRITE, size, NULL, &ret);
  clSafeCall( ret );
  clSafeCall( clEnqueueWriteBuffer(OP_opencl_core.command_queue, (cl_mem) *data_d, CL_TRUE, 0, size, *data_h, 0, NULL, NULL) );
  clSafeCall( clFlush(OP_opencl_core.command_queue) );
  clSafeCall( clFinish(OP_opencl_core.command_queue) );

//  cutilSafeCall ( cudaMalloc ( data_d, size ) );
//  cutilSafeCall ( cudaMemcpy ( *data_d, *data_h, size,
//                               cudaMemcpyHostToDevice ) );
//  cutilSafeCall ( cudaThreadSynchronize (  ) );
}

void op_fetch_data ( op_dat dat )
{
  //transpose data
  if (strstr( dat->type, ":soa")!= NULL) {
    char *temp_data = (char *)malloc(dat->size*dat->set->size*sizeof(char));
    clSafeCall( clEnqueueReadBuffer(OP_opencl_core.command_queue, (cl_mem) dat->data_d, CL_TRUE, 0, dat->size * dat->set->size, temp_data, 0, NULL, NULL) );
    clSafeCall( clFlush(OP_opencl_core.command_queue) );
    clSafeCall( clFinish(OP_opencl_core.command_queue) );
    int element_size = dat->size/dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (int j = 0; j < dat->set->size; j++) {
        for (int c = 0; c < element_size; c++) {
          dat->data[dat->size*j+element_size*i+c] = temp_data[element_size*i*dat->set->size + element_size*j + c];
        }
      }
    }
    free(temp_data);
  } else {
    clSafeCall( clEnqueueReadBuffer(OP_opencl_core.command_queue, (cl_mem) dat->data_d, CL_TRUE, 0, dat->size * dat->set->size, dat->data, 0, NULL, NULL) );
    clSafeCall( clFlush(OP_opencl_core.command_queue) );
    clSafeCall( clFinish(OP_opencl_core.command_queue) );
  }

//  //transpose data
//  if (strstr( dat->type, ":soa")!= NULL) {
//    char *temp_data = (char *)malloc(dat->size*dat->set->size*sizeof(char));
//    cutilSafeCall ( cudaMemcpy ( temp_data, dat->data_d,
//                                 dat->size * dat->set->size,
//                                 cudaMemcpyDeviceToHost ) );
//    cutilSafeCall ( cudaThreadSynchronize (  ) );
//    int element_size = dat->size/dat->dim;
//    for (int i = 0; i < dat->dim; i++) {
//      for (int j = 0; j < dat->set->size; j++) {
//        for (int c = 0; c < element_size; c++) {
//        	dat->data[dat->size*j+element_size*i+c] = temp_data[element_size*i*dat->set->size + element_size*j + c];
//        }
//      }
//    }
//    free(temp_data);
//  } else {
//  cutilSafeCall ( cudaMemcpy ( dat->data, dat->data_d,
//                               dat->size * dat->set->size,
//                               cudaMemcpyDeviceToHost ) );
//  cutilSafeCall ( cudaThreadSynchronize (  ) );
//  }
}



op_plan * op_plan_get ( char const * name, op_set set, int part_size,
                        int nargs, op_arg * args, int ninds, int *inds )
{
  op_plan *plan = op_plan_core ( name, set, part_size,
                                 nargs, args, ninds, inds );

  int set_size = set->size;
  for(int i = 0; i< nargs; i++) {
    if(args[i].idx != -1 && args[i].acc != OP_READ ) {
      set_size += set->exec_size;
      break;
    }
  }

  if ( plan->count == 1 ) {
    int *offsets = (int *)malloc((ninds+1)*sizeof(int));
    offsets[0] = 0;
    for ( int m = 0; m < ninds; m++ ) {
      int count = 0;
      for ( int m2 = 0; m2 < nargs; m2++ )
        if ( inds[m2] == m )
          count++;
      offsets[m+1] = offsets[m] + count;
    }
    op_mvHostToDevice ( ( void ** ) &( plan->ind_map ), offsets[ninds] * set_size * sizeof ( int ));
    for ( int m = 0; m < ninds; m++ ) {
      plan->ind_maps[m] = &plan->ind_map[set_size*offsets[m]];
    }
    free(offsets);

    int counter = 0;
    for ( int m = 0; m < nargs; m++ ) if ( plan->loc_maps[m] != NULL ) counter++;
    op_mvHostToDevice ( ( void ** ) &( plan->loc_map ), sizeof ( short ) * counter * set_size );
    counter = 0;
    for ( int m = 0; m < nargs; m++ ) if ( plan->loc_maps[m] != NULL ) {
      plan->loc_maps[m] = &plan->loc_map[set_size * counter]; counter++;
    }

    op_mvHostToDevice ( ( void ** ) &( plan->ind_sizes ),
                          sizeof ( int ) * plan->nblocks * plan->ninds );
    op_mvHostToDevice ( ( void ** ) &( plan->ind_offs ),
                          sizeof ( int ) * plan->nblocks * plan->ninds );
    op_mvHostToDevice ( ( void ** ) &( plan->nthrcol ),
                          sizeof ( int ) * plan->nblocks );
    op_mvHostToDevice ( ( void ** ) &( plan->thrcol ),
                          sizeof ( int ) * set_size );
    op_mvHostToDevice ( ( void ** ) &( plan->offset ),
                          sizeof ( int ) * plan->nblocks );
    op_mvHostToDevice ( ( void ** ) &( plan->nelems ),
                          sizeof ( int ) * plan->nblocks );
    op_mvHostToDevice ( ( void ** ) &( plan->blkmap ),
                          sizeof ( int ) * plan->nblocks );
  }

  return plan;
}

void op_opencl_exit ( )
{
    op_dat_entry *item;
    TAILQ_FOREACH(item, &OP_dat_list, entries)
    {
      clReleaseMemObject((cl_mem)(item->dat)->data_d);
//    cutilSafeCall (cudaFree((item->dat)->data_d));
   }

    for ( int ip = 0; ip < OP_plan_index; ip++ )
    {
      OP_plans[ip].ind_map = NULL;
      OP_plans[ip].loc_map = NULL;
      OP_plans[ip].ind_sizes = NULL;
      OP_plans[ip].ind_offs = NULL;
      OP_plans[ip].nthrcol = NULL;
      OP_plans[ip].thrcol = NULL;
      OP_plans[ip].offset = NULL;
      OP_plans[ip].nelems = NULL;
      OP_plans[ip].blkmap = NULL;
    }
   clSafeCall( clFlush(OP_opencl_core.command_queue) );
   clSafeCall( clFinish(OP_opencl_core.command_queue) );
//   clSafeCall( clReleaseKernel(OP_opencl_core.kernel) );
//   clSafeCall( clReleaseProgram(OP_opencl_core.program) );
   clSafeCall( clReleaseCommandQueue(OP_opencl_core.command_queue) );
   clSafeCall( clReleaseContext(OP_opencl_core.context) );
  free(OP_opencl_core.platform_id);
  printf("op_opencl_exit ()\n");
//  op_dat_entry *item;
//  TAILQ_FOREACH(item, &OP_dat_list, entries)
//  {
//		cutilSafeCall (cudaFree((item->dat)->data_d));
//	}
//
//  for ( int ip = 0; ip < OP_plan_index; ip++ )
//  {
//    OP_plans[ip].ind_map = NULL;
//    OP_plans[ip].loc_map = NULL;
//    OP_plans[ip].ind_sizes = NULL;
//    OP_plans[ip].ind_offs = NULL;
//    OP_plans[ip].nthrcol = NULL;
//    OP_plans[ip].thrcol = NULL;
//    OP_plans[ip].offset = NULL;
//    OP_plans[ip].nelems = NULL;
//    OP_plans[ip].blkmap = NULL;
//  }
//  cudaThreadExit ( );


}

//
// routines to resize constant/reduct arrays, if necessary
//

void reallocConstArrays ( int consts_bytes )
{
//  if ( consts_bytes > OP_consts_bytes ) {
//    if ( OP_consts_bytes > 0 ) {
//      free ( OP_consts_h );
//      cutilSafeCall ( cudaFree ( OP_consts_d ) );
//    }
//    OP_consts_bytes = 4 * consts_bytes;  // 4 is arbitrary, more than needed
//    OP_consts_h = ( char * ) malloc ( OP_consts_bytes );
//    cutilSafeCall ( cudaMalloc ( ( void ** ) &OP_consts_d,
//                                 OP_consts_bytes ) );
//  }
}

void reallocReductArrays ( int reduct_bytes )
{
//  if ( reduct_bytes > OP_reduct_bytes ) {
//    if ( OP_reduct_bytes > 0 ) {
//      free ( OP_reduct_h );
//      cutilSafeCall ( cudaFree ( OP_reduct_d ) );
//    }
//    OP_reduct_bytes = 4 * reduct_bytes;  // 4 is arbitrary, more than needed
//    OP_reduct_h = ( char * ) malloc ( OP_reduct_bytes );
//    cutilSafeCall ( cudaMalloc ( ( void ** ) &OP_reduct_d,
//                                 OP_reduct_bytes ) );
//  }
}

//
// routines to move constant/reduct arrays
//

void mvConstArraysToDevice ( int consts_bytes )
{
//  cutilSafeCall ( cudaMemcpy ( OP_consts_d, OP_consts_h, consts_bytes,
//                               cudaMemcpyHostToDevice ) );
//  cutilSafeCall ( cudaThreadSynchronize (  ) );
}

void mvReductArraysToDevice ( int reduct_bytes )
{
//  cutilSafeCall ( cudaMemcpy ( OP_reduct_d, OP_reduct_h, reduct_bytes,
//                               cudaMemcpyHostToDevice ) );
//  cutilSafeCall ( cudaThreadSynchronize (  ) );
}

void mvReductArraysToHost ( int reduct_bytes )
{
//  cutilSafeCall ( cudaMemcpy ( OP_reduct_h, OP_reduct_d, reduct_bytes,
//                               cudaMemcpyDeviceToHost ) );
//  cutilSafeCall ( cudaThreadSynchronize (  ) );
}

