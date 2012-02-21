#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
__device__
#include "mass_cuda.h"

__global__ void op_cuda_mass(float *mat_data,
                             int *rowptr,
                             int *colptr,
                             int nrow,
                             int *map,
                             int map_dim,
                             int *nelems,
                             int *blockoffset,
                             int nblocks,
                             float *data,
                             int dim)
{
  extern __shared__ char shared[];

  __shared__ float *data_s;
  __shared__ int *map_s;
  if ( blockIdx.x >= nblocks ) return;

  int nelem = nelems[blockIdx.x];
  int boffset = blockoffset[blockIdx.x];

  if ( threadIdx.x == 0 ) {
    data_s = (float *)&shared[0];
    map_s = (int *)&shared[ROUND_UP(nelem * map_dim * dim * sizeof(float))];
  }

  __syncthreads();
  // Needs nelem * map_dim * dim floats for data.
  // Plus a further nelem * map_dim ints for the map data
  // So for P1 triangles this requires
  // 6 * nelem * sizeof(float) + 3 * nelem * sizeof(int) bytes (plus
  // slop for 16-byte alignment)
  for ( int n = threadIdx.x; n < nelem; n+= blockDim.x ) {
    for ( int k = 0; k < map_dim; k++ ) {
      data_s[dim*(n * map_dim + k)] = data[dim*map[(n+boffset)*map_dim + k]];
      data_s[dim*(n * map_dim + k) + 1] = data[dim*map[(n+boffset)*map_dim + k] + 1];
      map_s[n * map_dim + k] = map[(n + boffset)*map_dim + k];
    }
  }
  __syncthreads();
  float entry;
  for ( int k = threadIdx.x; k < nelem * 3 * 3 * 3; k+=blockDim.x ) {
    // k == q + 3*j + 9*i + 27*n
    int n = k / 27;
    int i = (k - 27*n) / 9;
    int j = (k - 27*n - 9*i) / 3;
    int q = k - 27*n - 9*i - 3*j;
    int mapi = map_s[n*map_dim + i];
    int mapj = map_s[n*map_dim + j];
    entry = 0.0f;
    // Compute a single matrix entry
    // Probably best if q is iterated over in the kernel, but iterate
    // over it in the parallel loop as an example.
    mass(&entry, (float (*)[2])(data_s + n*map_dim*dim), i, j, q);
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
    atomicAdd(mat_data + offset, entry);
  }
}

void op_par_loop_mass(const char *name, op_set elements, op_sparsity sparsity,
                     op_arg arg_dat)
{
  int *map_d;
  cutilSafeCall(cudaMalloc((void **)&map_d,
                           sizeof(int) * arg_dat.map->dim * elements->size));
  cutilSafeCall(cudaMemcpy(map_d, arg_dat.map->map,
                           sizeof(int) * arg_dat.map->dim * elements->size,
                           cudaMemcpyHostToDevice));

  int nthread = 128;
  int nblocks = 128;
  int nblock = 128;
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

  int nshared;

  // This all needs to be wrapped in cuda versions of op_decl_sparsity
  // and op_decl_mat
  int *rowptr_d;
  int nrow = sparsity->nrows;
  int nnz = sparsity->rowptr[nrow];
  cutilSafeCall(cudaMalloc((void **)&rowptr_d, (nrow+1) * sizeof(int)));
  cutilSafeCall(cudaMemcpy(rowptr_d, sparsity->rowptr, (nrow+1) * sizeof(int),
                           cudaMemcpyHostToDevice));

  int *colptr_d;
  cutilSafeCall(cudaMalloc((void **)&colptr_d, nnz * sizeof(int)));
  cutilSafeCall(cudaMemcpy(colptr_d, sparsity->colidx, nnz * sizeof(int),
                           cudaMemcpyHostToDevice));
  float *data_d;
  cutilSafeCall(cudaMalloc((void **)&data_d, nnz * sizeof(float)));
  cutilSafeCall(cudaMemset(data_d, 0, nnz * sizeof(float)));

  nshared = nelems_h[0] * arg_dat.map->dim * arg_dat.dat->dim * sizeof(float)
    + nelems_h[0] * arg_dat.map->dim * sizeof(int);


  op_cuda_mass<<<nblocks, nthread, nshared>>>(data_d,
                                              rowptr_d,
                                              colptr_d,
                                              nrow,
                                              map_d,
                                              arg_dat.map->dim,
                                              nelems_d,
                                              boffset_d,
                                              nblock,
                                              (float *)arg_dat.data_d,
                                              arg_dat.dat->dim);

  // Print out resulting matrix if it comes from 2-element problem
  if ( elements->size == 2 ) {
      float *mat_h = (float *)malloc(nnz * sizeof(float));
      cutilSafeCall(cudaMemcpy(mat_h, data_d, nnz * sizeof(float),
                               cudaMemcpyDeviceToHost));
      for ( int i = 0; i < nrow; i++ ) {
          printf("row %d: ", i);
          for ( int j = sparsity->rowptr[i]; j < sparsity->rowptr[i+1]; j++ ) {
              printf("(%d, %g) ", sparsity->colidx[j], mat_h[j]);
          }
          printf("\n");
      }
      free(mat_h);
  }
  cutilSafeCall(cudaFree(data_d));
  cutilSafeCall(cudaFree(rowptr_d));
  cutilSafeCall(cudaFree(colptr_d));
  cutilSafeCall(cudaFree(map_d));
  cutilSafeCall(cudaFree(boffset_d));
  cutilSafeCall(cudaFree(nelems_d));
}
