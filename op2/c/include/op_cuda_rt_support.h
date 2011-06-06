#ifndef __OP_CUDA_RT_SUPPORT_H
#define __OP_CUDA_RT_SUPPORT_H


/*
 Open source copyright declaration based on BSD open source template:
 http://www.opensource.org/licenses/bsd-license.php
 
 * Copyright (c) 2009-2011, Mike Giles
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

//
// header files
//

#include "op_lib_core.h"
#include "op_rt_support.h"

// define CUDA warpsize

#define OP_WARPSIZE 32


//
// personal stripped-down version of cutil_inline.h 
//

#define cutilSafeCall(err) __cudaSafeCall(err,__FILE__,__LINE__)
#define cutilCheckMsg(msg) __cutilCheckMsg(msg,__FILE__,__LINE__)

inline void __cudaSafeCall ( cudaError err, 
                             const char *file, const int line );

inline void __cutilCheckMsg ( const char *errorMessage,
                             const char *file, const int line );
                            
inline void cutilDeviceInit ( int argc, char **argv );

//
// routines to move arrays to/from GPU device
//

void op_mvHostToDevice ( void **map, int size );

void op_cpHostToDevice ( void **data_d, void **data_h, int size );

void op_fetch_data ( op_dat dat );

op_plan *op_plan_get ( char const *name, op_set set, int part_size,
                       int nargs, op_arg *args, int ninds, int *inds );

void op_exit ();

//
// routines to resize constant/reduct arrays, if necessary
//

void reallocConstArrays ( int consts_bytes );

void reallocReductArrays ( int reduct_bytes );

//
// routines to move constant/reduct arrays
//

void mvConstArraysToDevice ( int consts_bytes );

void mvReductArraysToDevice ( int reduct_bytes );

void mvReductArraysToHost ( int reduct_bytes );


//
// reduction routine for arbitrary datatypes
//

template < op_access reduction, class T >
__inline__ __device__ void op_reduction ( volatile T *dat_g, T dat_l );

#endif
