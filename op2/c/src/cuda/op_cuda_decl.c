
// includes op_lib_core.h (wraps core declarations) and op_cuda_rt_support.h (uses cpHostToDevice)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

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

  
  //  cutilSafeCall(cudaThreadSetCacheConfig(cudaFuncCachePreferShared));
  printf("\n 16/48 L1/shared \n");
  
}

op_dat op_decl_dat ( op_set set, int dim, char const *type,
                     int size, char *data, char const *name )
{
  op_dat dat = op_decl_dat_core ( set, dim, type, size, data, name );

  printf ( "Copy from host to device of %d\n", dat->size * set->size );

        op_cpHostToDevice((void **)&(dat->data_d),
                    (void **)&(dat->data),
                             dat->size*set->size);
  
  return dat;
}

op_set op_decl_set ( int size, char const * name )
{
  return op_decl_set_core ( size, name );
}

op_map op_decl_map ( op_set from, op_set to, int dim, int * imap, char const * name )
{
  return op_decl_map_core ( from, to, dim, imap, name );
}

op_arg op_arg_dat ( op_dat dat, int idx, op_map map, int dim, char const * type, op_access acc )
{
	return op_arg_dat_core ( dat, idx, map, dim, type, acc );
}

op_arg op_arg_gbl ( char * data, int dim, const char * type, op_access acc )
{
	return op_arg_gbl ( data, dim, type, acc );
}

void op_decl_const_char ( int dim, char const *type,
			  int size, char *dat, char const *name)
{
  cutilSafeCall ( cudaMemcpyToSymbol ( name, dat, dim*size, 0, cudaMemcpyHostToDevice ) );
}
