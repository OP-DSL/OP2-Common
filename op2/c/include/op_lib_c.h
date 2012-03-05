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

#ifndef __OP_LIB_C_H
#define __OP_LIB_C_H

/*
 * This header file defines the user-level OP2 library for
 * the case of C programs.
 * To optimise software engineering activities it is also
 * used by the fortran backends
 */

#include <op_lib_core.h>

/* identity mapping and global identifier */

#define OP_ID  (op_map) NULL
#define OP_GBL (op_map) NULL

/*
 * external variables declared in op_lib_core.cpp
 */

extern int OP_diags, OP_part_size, OP_block_size;

extern int OP_set_index, OP_set_max,
           OP_map_index, OP_map_max,
           OP_dat_index, OP_dat_max,
           OP_mat_index, OP_sparsity_index,
           OP_plan_index, OP_plan_max,
           OP_kern_max;

extern op_set * OP_set_list;
extern op_map * OP_map_list;
extern op_dat * OP_dat_list;
extern op_mat * OP_mat_list;
extern op_sparsity * OP_sparsity_list;
extern op_kernel * OP_kernels;

/*
 * declaration of C routines wrapping lower layer implementations (e.g. CUDA, reference, etc..)
 */

#ifdef __cplusplus
extern "C" {
#endif

void op_init ( int, char **, int);

op_set op_decl_set ( int, char const * );

op_map op_decl_map ( op_set, op_set, int, int *, char const * );

op_dat op_decl_dat_char ( op_set, int, char const *, int, char *, char const * );

void op_decl_const_char ( int, char const *, int, char *, char const * );

op_arg op_arg_dat ( op_dat, int, op_map, int, char const *, op_access );

op_arg op_arg_gbl_char ( char * , int, const char*, int, op_access);

void op_fetch_data ( op_dat );

void op_exit (  );

void op_timing_output();

int op_get_size(op_set set);

void op_printf(const char* format, ...);

void op_timers( double *cpu, double *et );

#ifdef __cplusplus
}
#endif

#endif /* __OP_LIB_C_H */

#include "op_rt_support.h"

#include "op_hdf5.h"
