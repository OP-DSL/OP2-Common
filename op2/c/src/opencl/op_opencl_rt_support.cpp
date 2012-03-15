/*
  Open source copyright declaration based on BSD open source template:
  http://www.opensource.org/licenses/bsd-license.php

* Copyright ( c ) 2009-2011, Mike Giles
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
* ( INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION ) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* ( INCLUDING NEGLIGENCE OR OTHERWISE ) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

//
// header files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

//#include <math_constants.h>

#include "op_opencl_rt_support.h"

#include <utility>
#define __NO_STD_VECTOR //use cl::vector
#include <CL/cl.h>

//#define OP_WARPSIZE 4
#define OP_WARPSIZE 1

//#define ASYNC 1
#define PROFILE

#ifndef VEC
#define VEC 1
#endif
//#define HOST_MEMORY 1

// arrays for global constants and reductions

int   OP_consts_bytes=0,    OP_reduct_bytes=0;
char *OP_consts_h, *OP_reduct_h;
cl_mem OP_consts_d, OP_reduct_d;


cl_context      cxGPUContext;
cl_command_queue cqCommandQueue;
cl_device_id    *cpDevice;
cl_uint         ciNumDevices;
cl_uint         ciNumPlatforms;
cl_platform_id  *cpPlatform;

inline void assert_m( int val, const char* errmsg ) {
  if (!val) {
    LOG( LOG_FATAL, "%s\n", errmsg );
    exit( -1 );
  }
}

cl_program cpProgram;

void compileProgram ( const char *filename ) {
  printf( "building OpenCL program... " );

  cl_int ciErrNum;
  char *program_buf;
  FILE *file;

  file = fopen( filename, "rb" );
  assert_m( file != NULL, "error opening OpenCL program file" );

  fseek( file, 0, SEEK_END );
  int len = ftell( file ) + 1;
  program_buf = ( char * ) malloc( sizeof( char ) * (len+2) );
  bzero( program_buf, len );
  rewind( file );
  fread( program_buf, sizeof( char ), len, file );
  program_buf[ len ] = '\0';

  cpProgram = clCreateProgramWithSource( cxGPUContext, 1, (const char **) &program_buf, NULL, &ciErrNum );
  assert_m( ciErrNum == CL_SUCCESS, "error creating program from source" );

  char oclFlags[1000];

#if VEC > 1
//  sprintf( oclFlags,  "-cl-mad-enable -cl-fast-relaxed-math -D OP_WARPSIZE=%d -D VEC=%d -D VECTYPE=float%d", OP_WARPSIZE, VEC, VEC );
//  sprintf( oclFlags,  "-cl-opt-disable -D OP_WARPSIZE=%d -D VEC=%d -D VECTYPE=float%d", OP_WARPSIZE, VEC, VEC );
  sprintf( oclFlags,  "-cl-opt-disable -D OP_WARPSIZE=%d -D VEC=%d -D VECTYPE=float%d", OP_WARPSIZE, VEC, VEC );
#else
//  sprintf( oclFlags,  "-cl-mad-enable -cl-fast-relaxed-math -D OP_WARPSIZE=%d -D VEC=%d -D VECTYPE=float", OP_WARPSIZE, VEC );
//  sprintf( oclFlags,  "-cl-opt-disable -D OP_WARPSIZE=%d -D VEC=%d -D VECTYPE=float", OP_WARPSIZE, VEC );
  sprintf( oclFlags,  "-cl-opt-disable -D OP_WARPSIZE=%d -D VEC=%d -D VECTYPE=float", OP_WARPSIZE, VEC );
#endif


  ciErrNum = clBuildProgram( cpProgram, 1, cpDevice, oclFlags, NULL, NULL );
  if ( ciErrNum != CL_SUCCESS ) {
    char *log;
    size_t log_size = 0;
    clGetProgramBuildInfo( cpProgram, cpDevice[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size );
    log = ( char * ) malloc( sizeof( char ) * log_size );
    clGetProgramBuildInfo( cpProgram, cpDevice[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL );
    printf( "\n*******\n%s\n*******\n", log );
    free( log );
  }
  assert_m( ciErrNum == CL_SUCCESS, "error building program" );

  free( program_buf );

  printf( "OK\n" );
}

cl_kernel getKernel ( const char *kernel_name ) {
  LOG( LOG_INFO,  "getting kernel %s... ", kernel_name );

  cl_int ciErrNum;
  cl_kernel hKernel = clCreateKernel( cpProgram, kernel_name, &ciErrNum );
  assert( ciErrNum == CL_SUCCESS );
  assert_m( ciErrNum == CL_SUCCESS, "error creating kernel" );

  LOG( LOG_INFO, "OK\n" );

  return hKernel;
}

cl_mem op_allocate_constant( void *buf, size_t size ) {
  cl_int ciErrNum;
  cl_mem tmp;
  LOG( LOG_INFO, "allocating constant... " );

  tmp = clCreateBuffer( cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, buf, &ciErrNum);
  assert_m( ciErrNum == CL_SUCCESS, "error allocating constant" );

  LOG( LOG_INFO, "OK\n" );

  return tmp;
}

inline void OpenCLDeviceInit( int argc, char **argv ) {

  cl_int ciErrNum;


  LOG( LOG_INFO, "initialising device... " );
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers(&cpu_t1, &wall_t1);

  ciErrNum = 0;
  ciErrNum = clGetPlatformIDs( 0, NULL, &ciNumPlatforms );
  LOG( LOG_INFO, "obtained %d platforms", ciNumPlatforms );
  printf( "error %d\n", ciErrNum );
  assert_m( ciNumPlatforms > 0, "no platforms found!" );
  cpPlatform = ( cl_platform_id * ) malloc( sizeof( cl_platform_id ) * ciNumPlatforms );
  ciErrNum = clGetPlatformIDs( ciNumPlatforms, cpPlatform, NULL );
  assert_m( ciErrNum == CL_SUCCESS, "error getting platform IDs" );
  printf("%d num platforms\n", ciNumPlatforms);
  for(int i = 0; i < ciNumPlatforms; ++i) {
    printf("ID %d  ", cpPlatform[i]);
  }

//  ciErrNum = clGetDeviceIDs( cpPlatform[0], CL_DEVICE_TYPE_ALL, 0, NULL, &ciNumDevices );
  ciErrNum = clGetDeviceIDs( cpPlatform[1], CL_DEVICE_TYPE_ALL, 0, NULL, &ciNumDevices );
//  ciErrNum = clGetDeviceIDs( cpPlatform[0], CL_DEVICE_TYPE_GPU, 0, NULL, &ciNumDevices );
  LOG( LOG_INFO, "obtained %d devices", ciNumDevices );
  assert_m( ciNumDevices > 0, "no devices found!" );
  cpDevice = ( cl_device_id * ) malloc( sizeof( cl_device_id ) * ciNumDevices );
//  ciErrNum = clGetDeviceIDs( cpPlatform[0], CL_DEVICE_TYPE_ALL, ciNumDevices, cpDevice, NULL );
  ciErrNum = clGetDeviceIDs( cpPlatform[1], CL_DEVICE_TYPE_ALL, ciNumDevices, cpDevice, NULL  );
  //ciErrNum = clGetDeviceIDs( cpPlatform[0], CL_DEVICE_TYPE_GPU, ciNumDevices, cpDevice, NULL );
  assert_m( ciErrNum == CL_SUCCESS, "error getting device IDs" );
  LOG( LOG_INFO, "obtained device IDs");

  cxGPUContext = clCreateContext( 0, 1, cpDevice, NULL, NULL, &ciErrNum );
  assert_m( ciErrNum == CL_SUCCESS, "error creating context" );
  LOG( LOG_INFO, "created context");

#ifdef PROFILE
  cqCommandQueue = clCreateCommandQueue( cxGPUContext, cpDevice[0], CL_QUEUE_PROFILING_ENABLE, &ciErrNum );
  cl_ulong resolution;
  clGetDeviceInfo(cpDevice[0], CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(cl_ulong), &resolution, NULL);
  printf("device timer resolution: %lu ns\n", resolution);
#else
  cqCommandQueue = clCreateCommandQueue( cxGPUContext, cpDevice[0], 0, &ciErrNum );
#endif
  assert_m( ciErrNum == CL_SUCCESS, "error creating command queue" );
  LOG( LOG_INFO, "created command queue");

  LOG( LOG_INFO, "OK\n" );

  compileProgram( "kernels.cl" );

  op_timers(&cpu_t2, &wall_t2);
  op_timing_realloc(2);
  printf("initialisation time: %lf\n", wall_t2 - wall_t1 );

}

//
// routines to move arrays to/from device
//
//

void op_mvHostToDevice( void **map, int size ) {
  cl_int ciErrNum;
  cl_mem tmp;
  void * tmp_w;

  LOG( LOG_INFO, "moving data to device... " );

#ifdef HOST_MEMORY

  tmp = clCreateBuffer( cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, NULL, &ciErrNum);
  assert_m( ciErrNum == CL_SUCCESS, "error creating buffer" );

  tmp_w = clEnqueueMapBuffer( cqCommandQueue, tmp, CL_TRUE, CL_MAP_WRITE, 0, size, 0, NULL, NULL, &ciErrNum);
  assert_m( ciErrNum == CL_SUCCESS, "error mapping buffer" );

  memcpy( tmp_w, *map, size );

  ciErrNum = clEnqueueUnmapMemObject( cqCommandQueue, tmp, tmp_w, 0, NULL, NULL);
  assert_m( ciErrNum == CL_SUCCESS, "error unmapping buffer" );

#else

  tmp = clCreateBuffer( cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, *map, &ciErrNum);
  assert_m( ciErrNum == CL_SUCCESS, "error creating buffer" );
#endif

  ciErrNum = clFinish( cqCommandQueue );
  assert_m( ciErrNum == CL_SUCCESS, "error completing device commands" );

  *map = ( void * ) tmp; //implementation-specific? based on the fact that cl_mem is in fact a pointer type

  LOG( LOG_INFO, "OK" );
}

void op_cpHostToDevice( cl_mem *data_d, void **data_h, int size ) {

  cl_int ciErrNum;
  cl_mem tmp;
  void * tmp_w;

  LOG( LOG_INFO, "copying data to device... " );

#ifdef HOST_MEMORY
  tmp = clCreateBuffer( cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, NULL, &ciErrNum );
  assert_m( ciErrNum == CL_SUCCESS, "error creating buffer" );

  tmp_w = clEnqueueMapBuffer( cqCommandQueue, tmp, CL_TRUE, CL_MAP_WRITE, 0, size, 0, NULL, NULL, &ciErrNum);
  assert_m( ciErrNum == CL_SUCCESS, "error mapping buffer" );

  memcpy( tmp_w, *data_h, size );

  ciErrNum = clEnqueueUnmapMemObject( cqCommandQueue, tmp, tmp_w, 0, NULL, NULL);
  assert_m( ciErrNum == CL_SUCCESS, "error unmapping buffer" );
#else
  tmp = clCreateBuffer( cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, *data_h, &ciErrNum );
  assert_m( ciErrNum == CL_SUCCESS, "error creating buffer" );

#endif

  ciErrNum = clFinish( cqCommandQueue );
  assert_m( ciErrNum == CL_SUCCESS, "error completing device commands" );

  *data_d = tmp;

  LOG( LOG_INFO, "OK\n" );
}

void op_fetch_data( op_dat dat ) {
  cl_int ciErrNum;

  LOG( LOG_INFO, "fetching data from device... " );

  ciErrNum = clEnqueueReadBuffer(
      cqCommandQueue,
      (cl_mem) dat->data_d,
      CL_TRUE, //blocking read
      0,
      dat->size * dat->set->size,
      dat->data,
      0,
      NULL,
      NULL );

  assert_m( ciErrNum == CL_SUCCESS, "error copying data from device" );

  LOG( LOG_INFO, "OK\n" );
}


//
// OpenCL-specific OP2 functions
//
/*
void op_init( int argc, char **argv, int diags ){
  op_init_core( argc, argv, diags );

  OpenCLDeviceInit( argc, argv );

}

op_dat op_decl_dat_char( op_set set, int dim, char const *type,
                        int size, char *data, char const *name ){
  op_dat dat = op_decl_dat_core( set, dim, type, size, data, name );

  op_cpHostToDevice( (cl_mem*) &(dat->data_d ), (void **)&(dat->data), dat->size*set->size);
  return dat;
}

op_plan *op_plan_get( char const *name, op_set set, int part_size,
                     int nargs, op_arg *args, int ninds, int *inds ){

  op_plan *plan = op_plan_core( name, set, part_size, nargs, args, ninds, inds );

  // move plan arrays to device if first time

  if ( plan->count == 1 ) {
    for ( int m=0; m<ninds; m++ )
      op_mvHostToDevice( (void ** )&(plan->ind_maps[m]), sizeof(int)*plan->nindirect[m]);

    for ( int m=0; m<nargs; m++ )
      if ( plan->loc_maps[m] != NULL )
        op_mvHostToDevice( (void ** )&(plan->loc_maps[m]), sizeof(short)*plan->set->size);



    op_mvHostToDevice( (void ** )&(plan->ind_sizes),sizeof(int)*plan->nblocks *plan->ninds);
    op_mvHostToDevice( (void ** )&(plan->ind_offs), sizeof(int)*plan->nblocks *plan->ninds);
    op_mvHostToDevice( (void ** )&(plan->nthrcol),sizeof(int)*plan->nblocks);
    op_mvHostToDevice( (void ** )&(plan->thrcol ),sizeof(int)*plan->set->size);
    op_mvHostToDevice( (void ** )&(plan->offset ),sizeof(int)*plan->nblocks);
    op_mvHostToDevice( (void ** )&(plan->nelems ),sizeof(int)*plan->nblocks);
    op_mvHostToDevice( (void ** )&(plan->blkmap ),sizeof(int)*plan->nblocks);

  }
  return plan;
}

void op_exit( ){
  cl_int ciErrNum;

  ciErrNum = 0;

  for( int ip=0; ip<OP_plan_index; ip++ ) {
    for ( int m=0; m<OP_plans[ip].ninds; m++ )
      ciErrNum |= clReleaseMemObject( (cl_mem) OP_plans[ip].ind_maps[m] );
    for ( int m=0; m<OP_plans[ip].nargs; m++ )
      if ( OP_plans[ip].loc_maps[m] != NULL )
        ciErrNum |= clReleaseMemObject( (cl_mem) OP_plans[ip].loc_maps[m] );
    ciErrNum |= clReleaseMemObject( (cl_mem) OP_plans[ip].ind_offs );
    ciErrNum |= clReleaseMemObject( (cl_mem) OP_plans[ip].ind_sizes );
    ciErrNum |= clReleaseMemObject( (cl_mem) OP_plans[ip].nthrcol );
    ciErrNum |= clReleaseMemObject( (cl_mem) OP_plans[ip].thrcol );
    ciErrNum |= clReleaseMemObject( (cl_mem) OP_plans[ip].offset );
    ciErrNum |= clReleaseMemObject( (cl_mem) OP_plans[ip].nelems );
    ciErrNum |= clReleaseMemObject( (cl_mem) OP_plans[ip].blkmap );
  }

  for( int i=0; i<OP_dat_index; i++ ) {
    ciErrNum |= clReleaseMemObject( (cl_mem) OP_dat_list[i]->data_d );
  }

  assert_m( ciErrNum == CL_SUCCESS, "error releasing device memory" );

  op_exit_core( );

}
*/

//
// routines to resize constant/reduct arrays, if necessary
//

void releaseMemory ( cl_mem memobj ) {
  cl_int ciErrNum;
  LOG( LOG_INFO, "releasing memory... " );

  ciErrNum = clReleaseMemObject( memobj );
  assert_m( ciErrNum == CL_SUCCESS, "error releasing memory" );

  LOG( LOG_INFO, "OK\n" );
}

void reallocConstArrays( int consts_bytes ) {
  cl_int ciErrNum;

  LOG( LOG_INFO, "reallocating const arrays from %d to %d... ", OP_consts_bytes, consts_bytes );

  if ( consts_bytes>OP_consts_bytes ) {
    if ( OP_consts_bytes>0 ) {
      free( OP_consts_h );
      releaseMemory( OP_consts_d );
    }
    OP_consts_bytes = 4*consts_bytes;  // 4 is arbitrary, more than needed
    OP_consts_h = ( char * ) malloc(OP_consts_bytes);
    OP_consts_d = clCreateBuffer( cxGPUContext, CL_MEM_READ_WRITE, OP_consts_bytes, NULL, &ciErrNum );
    assert_m( ciErrNum == CL_SUCCESS, "error creating buffer" );
  } else {
   LOG( LOG_INFO, "(nothing to do)... " );
  }

   LOG( LOG_INFO, "OK\n" );
}

void reallocReductArrays( int reduct_bytes ) {
  cl_int ciErrNum;

  LOG( LOG_INFO, "reallocating reduct arrays from %d to %d... ", OP_reduct_bytes, reduct_bytes );

  if ( reduct_bytes>OP_reduct_bytes ) {
    if ( OP_reduct_bytes>0 ) {
      free( OP_reduct_h );
      releaseMemory( OP_reduct_d );
    }
    OP_reduct_bytes = 4*reduct_bytes;  // 4 is arbitrary, more than needed
    OP_reduct_h = ( char * ) malloc(OP_reduct_bytes);
    OP_reduct_d = clCreateBuffer( cxGPUContext, CL_MEM_READ_WRITE, OP_reduct_bytes, NULL, &ciErrNum );
    assert_m( ciErrNum == CL_SUCCESS, "error creating buffer" );
    // printf( "\n allocated %d bytes for reduction arrays \n",OP_reduct_bytes );
  } else {
    LOG( LOG_INFO, "(nothing to do)... " );
  }

  LOG( LOG_INFO, "OK\n" );
}


//
// routines to move constant/reduct arrays
//

void mvConstArraysToDevice( int consts_bytes ) {
  cl_int ciErrNum;

  LOG( LOG_INFO, "moving const arrays to device... " );

  ciErrNum = clEnqueueWriteBuffer( cqCommandQueue, OP_consts_d, CL_TRUE, 0, consts_bytes, OP_consts_h, 0, NULL, NULL );
  assert_m( ciErrNum == CL_SUCCESS, "error writing to device memory" );

  LOG( LOG_INFO, "OK\n" );
}

void mvReductArraysToDevice( int reduct_bytes ) {
  cl_int ciErrNum;
  LOG( LOG_INFO, "moving reduct arrays to device... " );

  //printf( "OP_reduct_d: %p\n OP_reduct_h: %p\n", OP_reduct_d, OP_reduct_h );

  ciErrNum = clEnqueueWriteBuffer( cqCommandQueue, OP_reduct_d, CL_TRUE, 0, reduct_bytes, OP_reduct_h, 0, NULL, NULL );
  //printf( "err: %d\n", ciErrNum );

  assert_m( ciErrNum == CL_SUCCESS, "error writing to device memory" );

  LOG( LOG_INFO, "OK\n" );
}

void mvReductArraysToHost( int reduct_bytes ) {
  cl_int ciErrNum;
  LOG( LOG_INFO, "moving reduct arrays to host... " );

  ciErrNum = clEnqueueReadBuffer( cqCommandQueue, OP_reduct_d, CL_TRUE, 0, reduct_bytes, OP_reduct_h, 0, NULL, NULL );
  assert_m( ciErrNum == CL_SUCCESS, "error reading from device memory" );

  LOG( LOG_INFO, "OK\n" );
}


cl_mem allocateSharedMemory ( size_t size ) {
  cl_int ciErrNum;
  LOG( LOG_INFO, "allocating shared memory... " );

  cl_mem temp = clCreateBuffer( cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum );
  assert_m( ciErrNum == CL_SUCCESS, "error allocating shared memory" );

  LOG( LOG_INFO, "OK\n" );

  return temp;
}

