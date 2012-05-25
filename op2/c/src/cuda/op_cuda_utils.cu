#include <cusp/csr_matrix.h>
#include <cusp/precond/diagonal.h>
#include <cusp/krylov/cg.h>

#include "op_lib_core.h"
#include "op_lib_mat.h"
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
  int n;
  int e;
  int i;
  int j;
  int entry_per_ele;
  int offset;
  int row;
  int col;
  entry_per_ele = rmapdim * cmapdim;

  n = threadIdx.x + blockIdx.x * blockDim.x;
  if ( n >= nelems * entry_per_ele ) return;

  e = n / entry_per_ele;
  i = (n - e * entry_per_ele) / rmapdim;
  j = (n - e * entry_per_ele - i * cmapdim);

  row = rmap[e * rmapdim + i];
  col = cmap[e * cmapdim + j];

  offset = pos(row, col, rowptr, colidx);
  op_atomic_add(data + offset, lma[n]);
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
  int nthread = 128;
  int nblock = (nelems * rmapdim * cmapdim) / nthread + 1;
  op_lma_to_csr_dev<<<nblock, nthread>>> (((T *)mat->lma_data),
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

template<class T>
__host__ void op_solve_impl(const op_mat mat, const op_dat b_dat, op_dat x_dat)
{
  // use array1d_view to wrap the individual arrays
  typedef typename cusp::array1d_view< thrust::device_ptr<int> > DeviceIndexArrayView;
  typedef typename cusp::array1d_view< thrust::device_ptr<T> >   DeviceValueArrayView;

  // combine the three array1d_views into a csr_matrix_view
  typedef cusp::csr_matrix_view<DeviceIndexArrayView,
                                DeviceIndexArrayView,
                                DeviceValueArrayView> DeviceView;

  op_sparsity sparsity = mat->sparsity;
  thrust::device_ptr<int> d_rowptr(sparsity->rowptr);
  thrust::device_ptr<int> d_colidx(sparsity->colidx);
  thrust::device_ptr<T>   d_data((T*)mat->data);
  thrust::device_ptr<T>   d_b((T*)b_dat->data_d);
  thrust::device_ptr<T>   d_x((T*)x_dat->data_d);

  DeviceIndexArrayView row_offsets   (d_rowptr, d_rowptr + sparsity->nrows + 1);
  DeviceIndexArrayView column_indices(d_colidx, d_colidx + sparsity->total_nz);
  DeviceValueArrayView values        (d_data,   d_data   + sparsity->total_nz);
  DeviceValueArrayView b             (d_b,      d_b      + b_dat->set->size);
  DeviceValueArrayView x             (d_x,      d_x      + x_dat->set->size);

  DeviceView A(sparsity->nrows, sparsity->ncols, sparsity->total_nz,
               row_offsets, column_indices, values);

  // set stopping criteria (iteration_limit = 1000, relative_tolerance = 1e-10)
  cusp::verbose_monitor<T> monitor(b, 1000, 1e-10);

  // setup preconditioner
  cusp::precond::diagonal<T, cusp::device_memory> M(A);

  // solve
  cusp::krylov::cg(A, x, b, monitor, M);
}

__host__ void op_solve(const op_mat mat, const op_dat b, op_dat x)
{
  assert( mat && b && x );
  if(strncmp(mat->type,"float",5)==0)
    op_solve_impl<float>(mat, b, x);
  if(strncmp(mat->type,"double",5)==0)
    op_solve_impl<double>(mat, b, x);
}
