#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
__device__
#include "mass_cuda.h"

__global__ void op_cuda_mass(float *mat,
           int *map,
           int map_dim,
           int *nelems,
           int nblocks,
           float *data,
           int dim)
{
  extern __shared__ char shared[];

  __shared__ float *data_s;
  int nelem;
  if ( blockIdx.x >= nblocks ) return;
  if ( threadIdx.x == 0 ) {
    data_s = (float *)&shared[0];
  }

  nelem = nelems[blockIdx.x];
  __syncthreads();
  for ( int n = threadIdx.x; n < nelem; n+= blockDim.x ) {
    for ( int k = 0; k < map_dim; k++ ) {
      data_s[dim*(n * map_dim + k)] = data[dim*map[n*map_dim + k]];
      data_s[dim*(n * map_dim + k) + 1] = data[dim*map[n*map_dim + k] + 1];
    }
  }
  __syncthreads();

  // Need to calculate global offset correctly for multiple blocks.
  // nelems would become an array index by blockDim, etc...
  for ( int n = threadIdx.x; n < nelem; n+=blockDim.x ) {
    mass((float (*)[3])(mat + n*map_dim*map_dim),
         (float (*)[2])(data_s + n*map_dim*dim));
  }
  __syncthreads();
}

void op_par_loop_mass(const char *name, op_set elements, int mat_size,
                     op_arg arg_dat)
{
  int *map_d;
  cutilSafeCall(cudaMalloc((void **)&map_d,
                           sizeof(int) * arg_dat.map->dim * elements->size));
  cutilSafeCall(cudaMemcpy(map_d, arg_dat.map->map,
                           sizeof(int) * arg_dat.map->dim * elements->size,
                           cudaMemcpyHostToDevice));

  float *mat_d;
  // Enough space for NUM_ELE 3x3 blocks
  // This is the number of nonzeros we have in the LMA approach.
  cutilSafeCall(cudaMalloc((void **)&mat_d,
                           sizeof(float) * arg_dat.map->dim
                           * arg_dat.map->dim * elements->size));
  cutilSafeCall(cudaMemset(mat_d, 0, sizeof(float) * arg_dat.map->dim
                           * arg_dat.map->dim * elements->size));
  int *nelems_h;
  nelems_h = (int *)malloc(1 * sizeof(int));
  int *nelems_d;
  cutilSafeCall(cudaMalloc((void **)&nelems_d,
         1 * sizeof(int)));

  nelems_h[0] = elements->size;
  cutilSafeCall(cudaMemcpy(nelems_d, nelems_h, 1 * sizeof(int),
         cudaMemcpyHostToDevice));

  int nthread = 128;
  int nblocks = 128;
  int nshared;

  nshared = nelems_h[0] * arg_dat.map->dim * arg_dat.dat->dim * sizeof(float);
  op_cuda_mass<<<nthread, nblocks, nshared>>>(mat_d,
                map_d,
                arg_dat.map->dim,
                nelems_d,
                1,
                (float *)arg_dat.data_d,
                arg_dat.dat->dim);

  // Copy matrix back and print, to check we got it right.

  float *mat_h = (float *)malloc(sizeof(float) * arg_dat.map->dim * arg_dat.map->dim * elements->size);
  cutilSafeCall(cudaMemcpy(mat_h, mat_d, sizeof(float) * arg_dat.map->dim * arg_dat.map->dim * elements->size,
         cudaMemcpyDeviceToHost));

  for ( int i = 0; i < elements->size; i++ ) {
    float *tmp = (float (*))(mat_h + i * arg_dat.map->dim * arg_dat.map->dim);
    printf("Block %d:\n", i);
    for ( int j = 0; j < arg_dat.map->dim; j++ ) {
      for ( int k = 0; k < arg_dat.map->dim; k++ ) {
  printf("%g ", ((float (*)[3])tmp)[j][k]);
      }
      printf("\n");
    }
    printf("\n");
  }

  free(mat_h);
  cutilSafeCall(cudaFree(mat_d));
  cutilSafeCall(cudaFree(map_d));
  cutilSafeCall(cudaFree(nelems_d));

  free(nelems_h);
}
