#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
#include "op_cuda_utils.h"

#include "types.h"
__device__
#include "rhs.h"

__global__ void op_cuda_rhs(ValueType *vec_data,
                             int *map,
                             int map_dim,
                             int *nelems,
                             int *blockoffset,
                             int nblocks,
                             ValueType *data0,
                             int dim0,
                             ValueType *data1,
                             int dim1)
{
  extern __shared__ char shared[];

  __shared__ ValueType *data_s0;
  __shared__ ValueType *data_s1;
  __shared__ int *map_s;
  ValueType *data_vec[3];
  ValueType *data_vec0[3];
  ValueType *data_vec1[3];
  if ( blockIdx.x >= nblocks ) return;

  const int nelem = nelems[blockIdx.x];
  const int boffset = blockoffset[blockIdx.x];

  if ( threadIdx.x == 0 ) {
    data_s0 = (ValueType *)&shared[0];
    data_s1 = (ValueType *)&shared[ROUND_UP(nelem * map_dim * dim0 * sizeof(ValueType))];
    map_s = (int *)&((char *)data_s1)[ROUND_UP(nelem * map_dim * dim1 * sizeof(ValueType))];
  }

  __syncthreads();
  // Needs nelem * map_dim * dim ValueTypes for data.
  // Plus a further nelem * map_dim ints for the map data
  // So for P1 triangles this requires
  // 6 * nelem * sizeof(ValueType) + 3 * nelem * sizeof(int) bytes (plus
  // slop for 16-byte alignment)
  for ( int n = threadIdx.x; n < nelem; n+= blockDim.x ) {
    for ( int k = 0; k < map_dim; k++ ) {
        map_s[n * map_dim + k] = map[(n + boffset)*map_dim + k];
    }
  }

  for ( int n = threadIdx.x; n < nelem * map_dim * dim0; n+=blockDim.x )
    data_s0[n] = data0[n%dim0 + map_s[n/dim0]*dim0];
  for ( int n = threadIdx.x; n < nelem * map_dim * dim1; n+=blockDim.x )
    data_s1[n] = data1[n%dim1 + map_s[n/dim1]*dim1];

  __syncthreads();
  for ( int n = threadIdx.x; n < nelem; n+=blockDim.x ) {
    data_vec[0] = vec_data + map_s[n*map_dim];
    data_vec[1] = vec_data + map_s[n*map_dim + 1];
    data_vec[2] = vec_data + map_s[n*map_dim + 2];
    data_vec0[0] = data_s0 + n * map_dim * dim0;
    data_vec0[1] = data_s0 + n * map_dim * dim0 + 1 * dim0;
    data_vec0[2] = data_s0 + n * map_dim * dim0 + 2 * dim0;
    data_vec1[0] = data_s1 + n * map_dim * dim1;
    data_vec1[1] = data_s1 + n * map_dim * dim1 + 1 * dim1;
    data_vec1[2] = data_s1 + n * map_dim * dim1 + 2 * dim1;
    // Compute an element vector
    rhs(data_vec, data_vec0, data_vec1);
  }
}

void op_par_loop_rhs(const char *name, op_set elements, op_arg arg_vec,
                     op_arg arg_dat0, op_arg arg_dat1)
{
  int *map_d;
  cutilSafeCall(cudaMalloc((void **)&map_d,
                           sizeof(int) * arg_vec.map->dim * elements->size));
  cutilSafeCall(cudaMemcpy(map_d, arg_vec.map->map,
                           sizeof(int) * arg_vec.map->dim * elements->size,
                           cudaMemcpyHostToDevice));

  int nthread = 128;
  int nblocks = 1;
  int nblock = 1;
  int nelems_h[nblock];
  int *nelems_d;
  cutilSafeCall(cudaMalloc((void **)&nelems_d, nblock * sizeof(int)));

  for ( int i = 0; i < nblock; i++ ) {
      nelems_h[i] = elements->size / nblock;
  }
  // Fix up leftovers
  for ( int i = 0; i < elements->size - nblock * (elements->size/nblock); i++ ) {
      nelems_h[i]++;
  }

  int boffset_h[nblock];
  int *boffset_d;

  boffset_h[0] = 0;
  for ( int i = 1; i < nblock; i++ ) {
      boffset_h[i] = boffset_h[i-1] + nelems_h[i-1];
  }
  cutilSafeCall(cudaMemcpy(nelems_d, nelems_h, nblock * sizeof(int),
                cudaMemcpyHostToDevice));

  cutilSafeCall(cudaMalloc((void **)&boffset_d, nblock * sizeof(int)));
  cutilSafeCall(cudaMemcpy(boffset_d, boffset_h, nblock * sizeof(int),
                cudaMemcpyHostToDevice));

  int nshared = ROUND_UP(nelems_h[0] * arg_vec.map->dim * arg_dat0.dat->dim * sizeof(ValueType));
  nshared += ROUND_UP(nelems_h[0] * arg_vec.map->dim * arg_dat1.dat->dim * sizeof(ValueType));
  nshared += ROUND_UP(nelems_h[0] * arg_vec.map->dim * sizeof(int));

  op_cuda_rhs<<<nblocks, nthread, nshared>>>((ValueType *)arg_vec.data_d,
                                              map_d,
                                              arg_vec.map->dim,
                                              nelems_d,
                                              boffset_d,
                                              nblock,
                                              (ValueType *)arg_dat0.data_d,
                                              arg_dat0.dat->dim,
                                              (ValueType *)arg_dat1.data_d,
                                              arg_dat1.dat->dim);

  // Print out resulting vector if it comes from 2-element problem
  const int ndofs = arg_vec.dat->set->size;
  if ( elements->size == 2 ) {
    ValueType *vec_h = (ValueType *)malloc(ndofs * sizeof(ValueType));
    cutilSafeCall(cudaMemcpy(vec_h, arg_vec.data_d, ndofs * sizeof(ValueType),
          cudaMemcpyDeviceToHost));
    for ( int i = 0; i < ndofs; i++ ) {
      printf("%g ", vec_h[i]);
    }
    printf("\n");
    free(vec_h);
  }
  cutilSafeCall(cudaFree(boffset_d));
  cutilSafeCall(cudaFree(nelems_d));
}
