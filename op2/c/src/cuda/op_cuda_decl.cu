
// includes op_lib_core.h (wraps core declarations) and op_cuda_rt_support.h (uses cpHostToDevice)

#include "op_lib_core.h"
#include "op_cuda_rt_support.h"


//
// CUDA-specific OP2 functions
//

void op_init(int argc, char **argv, int diags){
  op_init_core(argc, argv, diags);

  #if CUDART_VERSION < 3020
    #error : "must be compiled using CUDA 3.2 or later"
  #endif

  #ifdef CUDA_NO_SM_13_DOUBLE_INTRINSICS
    #warning : " *** no support for double precision arithmetic *** "
  #endif

  cutilDeviceInit(argc, argv);

  cutilSafeCall(cudaThreadSetCacheConfig(cudaFuncCachePreferShared));
  printf("\n 16/48 L1/shared \n");
}

op_dat op_decl_dat_char(op_set set, int dim, char const *type,
                        int size, char *data, char const *name){
  op_dat dat = op_decl_dat_core(set, dim, type, size, data, name);

  op_cpHostToDevice((void **)&(dat->data_d),
                    (void **)&(dat->data),
                               dat->size*set->size);
  return dat;
}

