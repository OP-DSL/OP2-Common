// 
// auto-generated by op2.m on 25-Oct-2011 14:59:20 
//

// user function                                                     
                                                                     
__device__                                                           
#include "update.h"                                                  
                                                                     
                                                                     
// CUDA kernel function                                              
                                                                     
__global__ void op_cuda_update(                                      
  float *arg0,                                                       
  float *arg1,                                                       
  float *arg2,                                                       
  float *arg3,                                                       
  float *arg4,                                                       
  int   offset_s,                                                    
  int   set_size ) {                                                 
                                                                     
  float arg0_l[4];                                                   
  float arg1_l[4];                                                   
  float arg2_l[4];                                                   
  float arg4_l[1];                                                   
  for (int d=0; d<1; d++) arg4_l[d]=ZERO_float;                      
  int   tid = threadIdx.x%OP_WARPSIZE;                               
                                                                     
  extern __shared__ char shared[];                                   
                                                                     
  char *arg_s = shared + offset_s*(threadIdx.x/OP_WARPSIZE);         
                                                                     
  // process set elements                                            
                                                                     
  for (int n=threadIdx.x+blockIdx.x*blockDim.x;                      
       n<set_size; n+=blockDim.x*gridDim.x) {                        
                                                                     
    int offset = n - tid;                                            
    int nelems = MIN(OP_WARPSIZE,set_size-offset);                   
                                                                     
    // copy data into shared memory, then into local                 
                                                                     
    for (int m=0; m<4; m++)                                          
      ((float *)arg_s)[tid+m*nelems] = arg0[tid+m*nelems+offset*4];  
                                                                     
    for (int m=0; m<4; m++)                                          
      arg0_l[m] = ((float *)arg_s)[m+tid*4];                         
                                                                     
    for (int m=0; m<4; m++)                                          
      ((float *)arg_s)[tid+m*nelems] = arg2[tid+m*nelems+offset*4];  
                                                                     
    for (int m=0; m<4; m++)                                          
      arg2_l[m] = ((float *)arg_s)[m+tid*4];                         
                                                                     
                                                                     
    // user-supplied kernel call                                     
                                                                     
    update( arg0_l,                                                  
            arg1_l,                                                  
            arg2_l,                                                  
            arg3+n,                                                  
            arg4_l );                                                
                                                                     
    // copy back into shared memory, then to device                  
                                                                     
    for (int m=0; m<4; m++)                                          
      ((float *)arg_s)[m+tid*4] = arg1_l[m];                         
                                                                     
    for (int m=0; m<4; m++)                                          
      arg1[tid+m*nelems+offset*4] = ((float *)arg_s)[tid+m*nelems];  
                                                                     
    for (int m=0; m<4; m++)                                          
      ((float *)arg_s)[m+tid*4] = arg2_l[m];                         
                                                                     
    for (int m=0; m<4; m++)                                          
      arg2[tid+m*nelems+offset*4] = ((float *)arg_s)[tid+m*nelems];  
                                                                     
  }                                                                  
                                                                     
  // global reductions                                               
                                                                     
  for(int d=0; d<1; d++)                                             
    op_reduction<OP_INC>(&arg4[d+blockIdx.x*1],arg4_l[d]);           
}                                                                    
                                                                     
                                                                     
// host stub function                                                
                                                                     
void op_par_loop_update(char const *name, op_set set,                
  op_arg arg0,                                                       
  op_arg arg1,                                                       
  op_arg arg2,                                                       
  op_arg arg3,                                                       
  op_arg arg4 ){                                                     
                                                                     
  float *arg4h = (float *)arg4.data;                                 
                                                                     
  if (OP_diags>2) {                                                  
    printf(" kernel routine w/o indirection:  update \n");           
  }                                                                  
                                                                     
  // initialise timers                                               
                                                                     
  double cpu_t1, cpu_t2, wall_t1, wall_t2;                           
  op_timers(&cpu_t1, &wall_t1);                                      
                                                                     
  // set CUDA execution parameters                                   
                                                                     
  #ifdef OP_BLOCK_SIZE_4                                             
    int nthread = OP_BLOCK_SIZE_4;                                   
  #else                                                              
    // int nthread = OP_block_size;                                  
    int nthread = 128;                                               
  #endif                                                             
                                                                     
  int nblocks = 200;                                                 
                                                                     
  // transfer global reduction data to GPU                           
                                                                     
  int maxblocks = nblocks;                                           
                                                                     
  int reduct_bytes = 0;                                              
  int reduct_size  = 0;                                              
  reduct_bytes += ROUND_UP(maxblocks*1*sizeof(float));               
  reduct_size   = MAX(reduct_size,sizeof(float));                    
                                                                     
  reallocReductArrays(reduct_bytes);                                 
                                                                     
  reduct_bytes = 0;                                                  
  arg4.data   = OP_reduct_h + reduct_bytes;                          
  arg4.data_d = OP_reduct_d + reduct_bytes;                          
  for (int b=0; b<maxblocks; b++)                                    
    for (int d=0; d<1; d++)                                          
      ((float *)arg4.data)[d+b*1] = ZERO_float;                      
  reduct_bytes += ROUND_UP(maxblocks*1*sizeof(float));               
                                                                     
  mvReductArraysToDevice(reduct_bytes);                              
                                                                     
  // work out shared memory requirements per element                 
                                                                     
  int nshared = 0;                                                   
  nshared = MAX(nshared,sizeof(float)*4);                            
  nshared = MAX(nshared,sizeof(float)*4);                            
  nshared = MAX(nshared,sizeof(float)*4);                            
                                                                     
  // execute plan                                                    
                                                                     
  int offset_s = nshared*OP_WARPSIZE;                                
                                                                     
  nshared = MAX(nshared*nthread,reduct_size*nthread);                
                                                                     
  op_cuda_update<<<nblocks,nthread,nshared>>>( (float *) arg0.data_d,
                                               (float *) arg1.data_d,
                                               (float *) arg2.data_d,
                                               (float *) arg3.data_d,
                                               (float *) arg4.data_d,
                                               offset_s,             
                                               set->size );          
                                                                     
  cutilSafeCall(cudaThreadSynchronize());                            
  cutilCheckMsg("op_cuda_update execution failed\n");                
                                                                     
  // transfer global reduction data back to CPU                      
                                                                     
  mvReductArraysToHost(reduct_bytes);                                
                                                                     
  for (int b=0; b<maxblocks; b++)                                    
    for (int d=0; d<1; d++)                                          
      arg4h[d] = arg4h[d] + ((float *)arg4.data)[d+b*1];             
                                                                     
  // update kernel record                                            
                                                                     
  op_timers(&cpu_t2, &wall_t2);                                      
  op_timing_realloc(4);                                              
  OP_kernels[4].name      = name;                                    
  OP_kernels[4].count    += 1;                                       
  OP_kernels[4].time     += wall_t2 - wall_t1;                       
  OP_kernels[4].transfer += (float)set->size * arg0.size;            
  OP_kernels[4].transfer += (float)set->size * arg1.size;            
  OP_kernels[4].transfer += (float)set->size * arg2.size * 2.0f;     
  OP_kernels[4].transfer += (float)set->size * arg3.size;            
}                                                                    
                                                                     
