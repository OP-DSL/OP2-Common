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

#ifndef __OP_SYCL_REDUCTION_H
#define __OP_SYCL_REDUCTION_H

#include <op_sycl_rt_support.h>

/*
 * This file provides an optimised implementation for reduction of OP2 global
 * variables.
 * It is separated from the op_sycl_rt_support.h file because the reduction code
 * is based on C++ templates, while the other file only includes C routines.
 */

template <op_access reduction, int intel, class T, class out_acc, class local_acc>
void op_reduction(out_acc dat_g, int offset, T dat_l, local_acc temp, cl::sycl::nd_item<1> &item_id) {
  T dat_t;
#ifdef __SYCL_COMPILER_VERSION
  sycl::sub_group sg = item_id.get_sub_group();
  if (intel)
    sg.barrier();
  else
#endif
    item_id.barrier(cl::sycl::access::fence_space::local_space); /* important to finish all previous activity */

  size_t tid = item_id.get_local_id(0);
  temp[tid] = dat_l;

  for (size_t d = item_id.get_local_range()[0] / 2; d > 0; d >>= 1) {
#ifdef __SYCL_COMPILER_VERSION
  if (intel)
    sg.barrier();
  else
#endif
    item_id.barrier(cl::sycl::access::fence_space::local_space);
    if (tid < d) {
      dat_t = temp[tid + d];

      switch (reduction) {
        case OP_INC:
          dat_l = dat_l + dat_t;
          break;
        case OP_MIN:
          if (dat_t < dat_l)
            dat_l = dat_t;
          break;
        case OP_MAX:
          if (dat_t > dat_l)
            dat_l = dat_t;
          break;
      }
      temp[tid] = dat_l;
    }
  }

  if (tid == 0) {
    switch (reduction) {
      case OP_INC:
        dat_g[offset] = dat_g[offset] + dat_l;
        break;
      case OP_MIN:
        if (dat_l < dat_g[offset])
          dat_g[offset] = dat_l;
        break;
      case OP_MAX:
        if (dat_l > dat_g[offset])
          dat_g[offset] = dat_l;
        break;
    }
  }
}


#endif /* __OP_SYCL_REDUCTION_H */
