/*
 * This header file defines the data structures required in the OP2 run-time support,
 * i.e. those related to OP2 plans, and it declares the low-level routines used to
 * build OP2 plans
 */

#ifndef __OP_RT_SUPPORT_H
#define __OP_RT_SUPPORT_H

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
  int        *nthrcol;   /* number of thread colors for each block */
  int        *thrcol;    /* thread colors */
  int        *offset;    /* offset for primary set */
  int         set_offset;/* offset to use within the set */
  int       **ind_maps;  /* pointers for indirect datasets */
  int        *ind_offs;  /* block offsets for indirect datasets */
  int        *ind_sizes; /* block sizes for indirect datasets */
  int        *nindirect; /* total sizes for indirect datasets */
  short     **loc_maps;  /* maps to local indices, renumbered as needed */
  int         nblocks;   /* number of blocks */
  int        *nelems;    /* number of elements in each block */
  int         ncolors;   /* number of block colors */
  int        *ncolblk;   /* number of blocks for each color */
  int        *blkmap;    /* block mapping */
  int         nshared;   /* bytes of shared memory required */
  float       transfer;  /* bytes of data transfer per kernel call */
  float       transfer2; /* bytes of cache line per kernel call */
  int         count;     /* number of times called */
} op_plan;


#ifdef __cplusplus
extern "C" {
#endif

op_plan * op_plan_old_core ( char const *, op_set, int, int, op_dat *,
                             int *, op_map *, int *, char const **, op_access *, int, int * );


op_plan * op_plan_core ( char const *, op_set, int, int, int, op_arg *, int, int * );

op_plan * op_plan_get ( char const * name, op_set set, int part_size,
                        int nargs, op_arg * args, int ninds, int * inds );

op_plan * op_plan_get_offset ( char const * name, op_set set, int set_offset, int part_size,
                        int nargs, op_arg * args, int ninds, int * inds );

void op_plan_check ( op_plan OP_plan, int ninds, int * inds );

void op_timers ( double *cpu, double *et );

void op_rt_exit ();

#ifdef __cplusplus
}
#endif

#endif /* __OP_RT_SUPPORT_H */
