#ifndef __OP_RT_SUPPORT_H
#define __OP_RT_SUPPORT_H

typedef struct {
  // input arguments
  char const  *name;
  op_set       set;
  int          nargs, ninds, part_size;
  op_map      *maps;
  op_dat      *dats;
  int         *idxs;
  op_access   *accs;

  // execution plan
  int        *nthrcol;  // number of thread colors for each block
  int        *thrcol;   // thread colors
  int        *offset;   // offset for primary set
  int       **ind_maps; // pointers for indirect datasets
  int        *ind_offs; // block offsets for indirect datasets
  int        *ind_sizes;// block sizes for indirect datasets
  int        *nindirect;// total sizes for indirect datasets
  short     **loc_maps; // maps to local indices, renumbered as needed
  int         nblocks;  // number of blocks
  int        *nelems;   // number of elements in each block
  int         ncolors;  // number of block colors
  int        *ncolblk;  // number of blocks for each color
  int        *blkmap;   // block mapping
  int         nshared;  // bytes of shared memory required
  float       transfer; // bytes of data transfer per kernel call
  float       transfer2;// bytes of cache line per kernel call
  int         count;    // number of times called
} op_plan;


extern "C"
op_plan * op_plan_old_core(char const *, op_set, int, int, op_dat *,
                           int *, op_map *, int *, char const **, op_access *, int, int *);


extern "C"
op_plan * op_plan_core(char const *, op_set, int, int, op_arg *, int, int *);

void op_rt_exit ();

#endif
