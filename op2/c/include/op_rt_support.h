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

#ifndef __OP_RT_SUPPORT_H
#define __OP_RT_SUPPORT_H

/*
 * This header file defines the data structures required in the OP2 run-time support,
 * i.e. those related to OP2 plans, and it declares the low-level routines used to
 * build OP2 plans
 */

#include <op_lib_core.h>

typedef struct {
  /* input arguments */
  char const  *name;
  op_set       set;
  int          nargs, ninds, part_size;
  op_map      *maps;
  op_dat      *dats;
  int         *idxs;
  op_access   *accs;

  /* execution plan */
  int        *nthrcol;    /* number of thread colors for each block */
  int        *thrcol;     /* thread colors */
  int        *offset;     /* offset for primary set */
  int       **ind_maps;   /* pointers for indirect datasets */
  int        *ind_offs;   /* block offsets for indirect datasets */
  int        *ind_sizes;  /* block sizes for indirect datasets */
  int        *nindirect;  /* total sizes for indirect datasets */
  short     **loc_maps;   /* maps to local indices, renumbered as needed */
  int         nblocks;    /* number of blocks */
  int        *nelems;     /* number of elements in each block */
  int         ncolors_core;/* mumber of core colors in MPI */
  int         ncolors;    /* number of block colors */
  int        *ncolblk;    /* number of blocks for each color */
  int        *blkmap;     /* block mapping */
  int         nshared;    /* bytes of shared memory required */
  float       transfer;   /* bytes of data transfer per kernel call */
  float       transfer2;  /* bytes of cache line per kernel call */
  int         count;      /* number of times called */
} op_plan;


#ifdef __cplusplus
extern "C" {
#endif

op_plan * op_plan_old_core ( char const *, op_set, int, int, op_dat *,
                             int *, op_map *, int *, char const **, op_access *, int, int * );

op_plan * op_plan_core ( char const *, op_set, int, int, op_arg *, int, int * );

op_plan * op_plan_get ( char const * name, op_set set, int part_size,
                        int nargs, op_arg * args, int ninds, int * inds );

void op_plan_check ( op_plan OP_plan, int ninds, int * inds );

void op_rt_exit ( void );

#ifdef __cplusplus
}
#endif

#endif /* __OP_RT_SUPPORT_H */
