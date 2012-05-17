#include "op_lib_core.h"
#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
#include "op_cuda_utils.h"

__device__ int pos(int row, int col, int* rowptr, int* colidx)
{
  for ( int k = rowptr[row]; k < rowptr[row+1]; k++ )
    if ( colidx[k] == col )
      return k;
  return INT_MAX;
}

template<class T>
__global__ void op_lma_to_csr_dev (T *lma, T *data,
                                   int *rowptr,
                                   int *colidx,
                                   int *rmap,
                                   int rmapdim,
                                   int *cmap,
                                   int cmapdim,
                                   int nelems)
{
  if ( threadIdx.x + blockIdx.x * blockDim.x > nelems * rmapdim * cmapdim )
    return;

  int n;
  int e;
  int i;
  int j;

  int entry_per_ele = rmapdim * cmapdim;

  int offset;
  int row;
  int col;
  for ( n = threadIdx.x; n < nelems * entry_per_ele; n += blockDim.x )
  {
    e = n / entry_per_ele;
    i = (n - e * entry_per_ele) / rmapdim;
    j = (n - e * entry_per_ele - i * cmapdim);

    row = rmap[e * rmapdim + i];
    col = cmap[e * cmapdim + j];

    offset = pos(row, col, rowptr, colidx);
    op_atomic_add(data + offset, lma[n]);
  }
}

template<class T>
__host__ void op_mat_lma_to_csr(op_arg arg, op_set set)
{
  op_mat mat = arg.mat;
  op_sparsity sparsity = mat->sparsity;
  op_map rmap = sparsity->rowmap;
  int rmapdim = rmap->dim;
  op_map cmap = sparsity->colmap;
  int cmapdim = cmap->dim;
  int * rowptr = sparsity->rowptr;
  int * colidx = sparsity->colidx;
  int nelems = set->size;

  if ( rmap->map_d == NULL ) {
    op_cpHostToDevice ((void **)&(rmap->map_d),
        (void **)&(rmap->map),
        sizeof(int) * rmapdim * nelems);
  }

  if ( cmap->map_d == NULL ) {
    op_cpHostToDevice((void **)&(cmap->map_d),
        (void **)&(cmap->map),
        sizeof(int) * cmapdim * nelems);
  }
  op_lma_to_csr_dev<<<128, 128>>> (((T *)mat->lma_data),
                                   ((T *)mat->data),
                                   rowptr,
                                   colidx,
                                   rmap->map_d,
                                   rmapdim,
                                   cmap->map_d,
                                   cmapdim,
                                   nelems);
}

__host__ void op_mat_lma_to_csr(float *dummy, op_arg arg, op_set set)
{
  op_mat_lma_to_csr<float>(arg, set);
}

__host__ void op_mat_lma_to_csr(double *dummy, op_arg arg, op_set set)
{
  op_mat_lma_to_csr<double>(arg, set);
}
