/*
 * Open source copyright declaration based on BSD open source template:
 * http://www.opensource.org/licenses/bsd-license.php
 *
 * This file is part of the OP2 distribution.
 *
 * Copyright (c) 2011, Mike Giles and others. Please see the AUTHORS file in
 * the main source directory for a full list of copyright holders.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * The name of Mike Giles may not be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __OP_CUDA_UTILS_H
#define __OP_CUDA_UTILS_H

/*
 * This header file declares CUDA utility functions.
 */

/*
 * Device-only atomic routines
 */

#ifdef __CUDACC__

__device__ void op_atomic_add(double *address, double val)
{
  unsigned long long int new_val, old;
  unsigned long long int old2 = __double_as_longlong(*address);

  do
  {
    old = old2;
    new_val = __double_as_longlong(__longlong_as_double(old) + val);
    old2 = atomicCAS((unsigned long long int *)address, old, new_val);
  } while(old2!=old);
}

__device__ void op_atomic_add(float *address, float val)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 200
  atomicAdd(address, val);
#else
  unsigned int new_val, old;
  unsigned int old2 = __float_as_int(*address);

  do
  {
    old = old2;
    new_val = __float_as_int(__int_as_float(old) + val);
    old2 = atomicCAS((unsigned int *)address, old, new_val);
  } while(old2!=old);
#endif
}

#endif

#endif /* __OP_CUDA_UTILS_H */

