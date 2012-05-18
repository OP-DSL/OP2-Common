#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
#include "op_cuda_utils.h"

#include "types.h"
__device__
#include "mass.h"

__global__ void op_cuda_mass(ValueType *mat_data,
                             int *rowptr,
                             int *colptr,
                             int nrow,
                             int *map,
                             int map_dim,
                             int *nelems,
                             int *blockoffset,
                             int nblocks,
                             ValueType *data,
                             int dim)
{
  extern __shared__ char shared[];

  __shared__ ValueType *data_s;
  __shared__ int *map_s;
  ValueType *data_vec[3];
  if ( blockIdx.x >= nblocks ) return;

  int nelem = nelems[blockIdx.x];
  int boffset = blockoffset[blockIdx.x];

  if ( threadIdx.x == 0 ) {
    data_s = (ValueType *)&shared[0];
    map_s = (int *)&shared[ROUND_UP(nelem * map_dim * dim * sizeof(ValueType))];
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

  for ( int n = threadIdx.x; n < nelem * map_dim * 2; n+=blockDim.x ) {
      data_s[n] = data[n%2 + map_s[n/2]*2];
  }
  __syncthreads();
  ValueType entry;
  for ( int k = threadIdx.x; k < nelem * 3 * 3; k+=blockDim.x ) {
    // k == j + 3*i + 9*n
    int n = k / 9;
    int i = (k - 9*n) / 3;
    int j = (k - 9*n - 3*i);
    int mapi = map_s[n*map_dim + i];
    int mapj = map_s[n*map_dim + j];
    data_vec[0] = data_s + n * map_dim * dim;
    data_vec[1] = data_s + n * map_dim * dim + 1 * dim;
    data_vec[2] = data_s + n * map_dim * dim + 2 * dim;
    // Compute a single matrix entry
    entry = 0.0f;
    mass(&entry, data_vec, i, j);
    // Insert matrix entry into global matrix.
    // find column offset
    int offset;
    for (int p = rowptr[mapi]; p < rowptr[mapi+1]; p++ ) {
      if ( colptr[p] == mapj )
        offset = p;
    }
    // To avoid these atomics we'd have to do colour-order traversal
    // of the elements.  And, if q is iterated over here, warp-level
    // reductions.
    op_atomic_add(mat_data + offset, entry);
  }
}

void op_par_loop_mass(const char *name, op_set elements, op_arg arg_mat,
                     op_arg arg_dat)
{
  int *map_d;
  cutilSafeCall(cudaMalloc((void **)&map_d,
                           sizeof(int) * arg_dat.map->dim * elements->size));
  cutilSafeCall(cudaMemcpy(map_d, arg_dat.map->map,
                           sizeof(int) * arg_dat.map->dim * elements->size,
                           cudaMemcpyHostToDevice));

  int nthread = 128;
  int nblocks = 1;
  int nblock = 1;
  int nelems_h[nblock];
  int *nelems_d;
  cutilSafeCall(cudaMalloc((void **)&nelems_d,
         nblock * sizeof(int)));

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

  op_sparsity sparsity = arg_mat.mat->sparsity;
  int nrow = sparsity->nrows;
  int nnz = sparsity->total_nz;
  int nshared = ROUND_UP(nelems_h[0] * arg_dat.map->dim * arg_dat.dat->dim * sizeof(ValueType));
  nshared += ROUND_UP(nelems_h[0] * arg_dat.map->dim * sizeof(int));


  op_cuda_mass<<<nblocks, nthread, nshared>>>((ValueType *)arg_mat.mat->data,
                                              sparsity->rowptr,
                                              sparsity->colidx,
                                              nrow,
                                              map_d,
                                              arg_dat.map->dim,
                                              nelems_d,
                                              boffset_d,
                                              nblock,
                                              (ValueType *)arg_dat.data_d,
                                              arg_dat.dat->dim);

  // Print out resulting matrix if it comes from 2-element problem
  if ( elements->size == 2 ) {
    ValueType *mat_h = (ValueType *)malloc(nnz * sizeof(ValueType));
    cutilSafeCall(cudaMemcpy(mat_h, arg_mat.mat->data, nnz * sizeof(ValueType),
          cudaMemcpyDeviceToHost));
    for ( int i = 0; i < nnz; i++ ) {
      printf("%g ", mat_h[i]);
    }
    printf("\n");
    free(mat_h);
  }
  cutilSafeCall(cudaFree(boffset_d));
  cutilSafeCall(cudaFree(nelems_d));
}
