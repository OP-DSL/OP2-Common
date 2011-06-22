#ifndef __OP_CUDA_REDUCTION_H
#define __OP_CUDA_REDUCTION_H

/*
 * This file provides an optimised implementation for reduction of OP2 global variables.
 * It is separated from the op_cuda_rt_support.h file because the reduction code
 * is based on C++ templates, while the other file only includes C routines.
 */

#include <cuda.h>


//
// reduction routine for arbitrary datatypes
//

template < op_access reduction, class T >
__inline__ __device__ void op_reduction(volatile T *dat_g, T dat_l)
{
  int tid = threadIdx.x;
  int d   = blockDim.x>>1;
  extern __shared__ T temp[];

  __syncthreads();  // important to finish all previous activity

  temp[tid] = dat_l;

  for (; d>warpSize; d>>=1) {
    __syncthreads();
    if (tid<d) {
      switch (reduction) {
        case OP_INC:
          temp[tid] = temp[tid] + temp[tid+d];
          break;
        case OP_MIN:
          if(temp[tid+d]<temp[tid]) temp[tid] = temp[tid+d];
          break;
        case OP_MAX:
          if(temp[tid+d]>temp[tid]) temp[tid] = temp[tid+d];
          break;
      }
    }
  }

  __syncthreads();

  volatile T *vtemp = temp;   // see Fermi compatibility guide

  if (tid<warpSize) {
    for (; d>0; d>>=1) {
      if (tid<d) {
        switch (reduction) {
          case OP_INC:
            vtemp[tid] = vtemp[tid] + vtemp[tid+d];
            break;
          case OP_MIN:
            if(vtemp[tid+d]<vtemp[tid]) vtemp[tid] = vtemp[tid+d];
            break;
          case OP_MAX:
            if(vtemp[tid+d]>vtemp[tid]) vtemp[tid] = vtemp[tid+d];
            break;
        }
      }
    }
  }

  if (tid==0) {
    switch (reduction) {
      case OP_INC:
        *dat_g = *dat_g + vtemp[0];
        break;
      case OP_MIN:
        if(temp[0]<*dat_g) *dat_g = vtemp[0];
        break;
      case OP_MAX:
        if(temp[0]>*dat_g) *dat_g = vtemp[0];
        break;
    }
  }
}

#endif