#ifndef __OP_CUDA_REDUCTION_H
#define __OP_CUDA_REDUCTION_H

/*
 * This file provides an optimised implementation for reduction of OP2 global variables.
 * It is separated from the op_cuda_rt_support.h file because the reduction code
 * is based on C++ templates, while the other file only includes C routines.
 */

#include <cuda.h>


/*
 * reduction routine for arbitrary datatypes
 */

template < op_access reduction, class T >
__inline__ __device__ void op_reduction( volatile T * dat_g, T dat_l )
{
  extern __shared__ volatile T temp[];
  T   dat_t;

  __syncthreads();     /* important to finish all previous activity */

  int tid   = threadIdx.x;
  temp[tid] = dat_l;

  // first, cope with blockDim.x perhaps not being a power of 2

  __syncthreads();

  int d = 1 << (31 - __clz(((int)blockDim.x-1)) );
  // d = blockDim.x/2 rounded up to nearest power of 2

  if ( tid+d < blockDim.x ) {
    dat_t = temp[tid+d];

    switch ( reduction ) {
      case OP_INC:
        dat_l = dat_l + dat_t;
        break;
      case OP_MIN:
        if ( dat_t < dat_l ) dat_l = dat_t;
        break;
      case OP_MAX:
        if ( dat_t > dat_l ) dat_l = dat_t;
        break;
    }

    temp[tid] = dat_l;
  }

  // second, do reductions involving more than one warp

  for (d >>= 1 ; d > warpSize; d >>= 1 ) {
    __syncthreads();

    if ( tid < d ) {
      dat_t = temp[tid+d];

      switch ( reduction ) {
        case OP_INC:
          dat_l = dat_l + dat_t;
          break;
        case OP_MIN:
          if ( dat_t < dat_l ) dat_l = dat_t;
          break;
        case OP_MAX:
          if ( dat_t > dat_l ) dat_l = dat_t;
          break;
      }

      temp[tid] = dat_l;
    }
  }

  // third, do reductions involving just one warp

  __syncthreads();

  if ( tid < warpSize ) {
    for ( ; d > 0; d >>= 1 ) {
      if ( tid < d ) {
        dat_t = temp[tid+d];

        switch ( reduction ) {
          case OP_INC:
            dat_l = dat_l + dat_t;
            break;
          case OP_MIN:
            if ( dat_t < dat_l ) dat_l = dat_t;
            break;
          case OP_MAX:
            if ( dat_t > dat_l ) dat_l = dat_t;
            break;
  }

        temp[tid] = dat_l;
      }
    }

    // finally, update global reduction variable

    if ( tid == 0 ) {
      switch ( reduction ) {
        case OP_INC:
          *dat_g = *dat_g + dat_l;
          break;
        case OP_MIN:
          if ( dat_l < *dat_g ) *dat_g = dat_l;
          break;
        case OP_MAX:
          if ( dat_l > *dat_g ) *dat_g = dat_l;
          break;
      }
    }
  }
}

/*
 * reduction routine for arbitrary datatypes
 * (alternative version using just one warp)
 *
 */

template < op_access reduction, class T >
__inline__ __device__ void op_reduction_alt ( volatile T * dat_g, T dat_l )
{
  extern __shared__ volatile T temp[];
  T   dat_t;

  __syncthreads();  /* important to finish all previous activity */

  int tid   = threadIdx.x;
  temp[tid] = dat_l;

  __syncthreads();

  // set number of active threads

  int d = warpSize;

  if ( blockDim.x < warpSize )
    d = 1 << (31 - __clz((int)blockDim.x) );
  // this gives blockDim.x rounded down to nearest power of 2

  if ( tid < d ) {

    // first, do reductions for each thread

    for (int t = tid+d; t < blockDim.x ; t += d) {
      dat_t = temp[t];

      switch ( reduction ) {
        case OP_INC:
          dat_l = dat_l + dat_t;
          break;
        case OP_MIN:
          if ( dat_t < dat_l ) dat_l = dat_t;
          break;
        case OP_MAX:
          if ( dat_t > dat_l ) dat_l = dat_t;
          break;
      }
    }

    temp[tid] = dat_l;

    // second, do reductions to combine thread reductions

    for (d >>= 1 ; d > 0; d >>= 1 ) {
      if ( tid < d ) {
        dat_t = temp[tid+d];

        switch ( reduction ) {
          case OP_INC:
            dat_l = dat_l + dat_t;
            break;
          case OP_MIN:
            if ( dat_t < dat_l ) dat_l = dat_t;
            break;
          case OP_MAX:
            if ( dat_t > dat_l ) dat_l = dat_t;
            break;
        }

        temp[tid] = dat_l;
      }
    }

    // finally, update global reduction variable

    if ( tid == 0 ) {
      switch ( reduction ) {
        case OP_INC:
          *dat_g = *dat_g + dat_l;
          break;
        case OP_MIN:
          if ( dat_l < *dat_g ) *dat_g = dat_l;
          break;
        case OP_MAX:
          if ( dat_l > *dat_g ) *dat_g = dat_l;
          break;
      }
    }
  }
}

#endif /* __OP_CUDA_REDUCTION_H */

