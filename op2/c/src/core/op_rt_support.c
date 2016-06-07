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

/*
 * This file implements the OP2 run-time support used by different
 * OP2 back-ends, like CUDA and OpenMP. It provides and implementation
 * of the plan building function for colouring and partitioning of
 * unstructured meshes.
 */

#include "op_rt_support.h"

/*
 * Global variables
 */

int OP_plan_index = 0, OP_plan_max = 0;
op_plan * OP_plans;
double OP_plan_time = 0;

extern op_kernel * OP_kernels;
extern int OP_kern_max;

void
op_rt_exit (  )
{
  /* free storage for plans */
  for ( int ip = 0; ip < OP_plan_index; ip++ )
  {
    free ( OP_plans[ip].dats );
    free ( OP_plans[ip].idxs );
    free ( OP_plans[ip].maps );
    free ( OP_plans[ip].accs );
    free ( OP_plans[ip].inds_staged );
    free ( OP_plans[ip].nthrcol );
    free ( OP_plans[ip].thrcol );
    free ( OP_plans[ip].offset );
    free ( OP_plans[ip].ind_offs );
    free ( OP_plans[ip].ind_sizes );
    free ( OP_plans[ip].nelems );
    free ( OP_plans[ip].blkmap );
    free ( OP_plans[ip].ind_map );
    free ( OP_plans[ip].ind_maps );
    free ( OP_plans[ip].nindirect );
    free ( OP_plans[ip].loc_map );
    free ( OP_plans[ip].loc_maps );
    free ( OP_plans[ip].ncolblk );
    free ( OP_plans[ip].nsharedCol);
    op_free(OP_plans[ip].col_reord);
    if (OP_plans[ip].col_offsets != NULL) {
      op_free(OP_plans[ip].col_offsets[0]);
      op_free(OP_plans[ip].col_offsets);
    }
  }

  OP_plan_index = 0;
  OP_plan_max = 0;

  free ( OP_plans );
  OP_plans = NULL;
}

/*
 * comparison function for integer quicksort in op_plan
 */

static int
comp ( const void * a2, const void * b2 )
{
  int *a = ( int * ) a2;
  int *b = ( int * ) b2;

  if ( *a == *b )
    return 0;
  else if ( *a < *b )
    return -1;
  else
    return 1;
}

/*
 * comparison function for key-value quicksort in op_plan
 */

typedef struct {
  int key;
  int value;
} op_keyvalue;

int comp2 (const void * a, const void * b)
{
  if ( ((op_keyvalue*)a)->key <  ((op_keyvalue*)b)->key ) return -1;
  if ( ((op_keyvalue*)a)->key == ((op_keyvalue*)b)->key ) return 0;
  if ( ((op_keyvalue*)a)->key >  ((op_keyvalue*)b)->key ) return 1;
  return 0;
}

/*
 * plan check routine
 */

void op_plan_check( op_plan OP_plan, int ninds, int * inds)
{
  //compute exec_length - which include the exec halo given certain conditions (MPI)
  int exec_length = OP_plan.set->size;
  for ( int m = 0; m < OP_plan.nargs; m++ )
  {
    if(OP_plan.idxs[m] != -1 && OP_plan.accs[m] != OP_READ ) //if it needs exchaning
    {
      exec_length += OP_plan.set->exec_size;
      break;
    }
  }

  int err, ntot;

  int nblock = 0;
  for ( int col = 0; col < OP_plan.ncolors; col++ )
  {
    nblock += OP_plan.ncolblk[col];
  }

  /*
   * check total size
   */

  int nelem = 0;
  for ( int n = 0; n < nblock; n++ )
    nelem += OP_plan.nelems[n];

  if ( nelem != exec_length )
  {
    printf ( " *** OP_plan_check: nelems error \n" );
  }
  else if ( OP_diags > 6 )
  {
    printf ( " *** OP_plan_check: nelems   OK \n" );
  }

  /*
   * check offset and nelems are consistent
   */

  err = 0;
  ntot = 0;

  for ( int n = 0; n < nblock; n++ )
  {
    err += ( OP_plan.offset[n] != ntot );
    ntot += OP_plan.nelems[n];
  }

  if ( err != 0 )
  {
    printf ( " *** OP_plan_check: offset error \n" );
  }
  else if ( OP_diags > 6 )
  {
    printf ( " *** OP_plan_check: offset   OK \n" );
  }

  /*
   * check blkmap permutation
   */

  int *blkmap = ( int * ) op_malloc ( nblock * sizeof ( int ) );
  for ( int n = 0; n < nblock; n++ )
    blkmap[n] = OP_plan.blkmap[n];
  qsort ( blkmap, nblock, sizeof ( int ), comp );

  err = 0;
  for ( int n = 0; n < nblock; n++ )
    err += ( blkmap[n] != n );

  free ( blkmap );

  if ( err != 0 )
  {
    printf ( " *** OP_plan_check: blkmap error \n" );
  }
  else if ( OP_diags > 6 )
  {
    printf ( " *** OP_plan_check: blkmap   OK \n" );
  }

  /*
   * check ind_offs and ind_sizes are consistent
   */

  err = 0;

  for ( int i = 0; i < ninds; i++ )
  {
    ntot = 0;

    for ( int n = 0; n < nblock; n++ )
    {
      err += ( OP_plan.ind_offs[i + n * ninds] != ntot );
      ntot += OP_plan.ind_sizes[i + n * ninds];
    }
  }

  if ( err != 0 )
  {
    printf ( " *** OP_plan_check: ind_offs error \n" );
  }
  else if ( OP_diags > 6 )
  {
    printf ( " *** OP_plan_check: ind_offs OK \n" );
  }

  /*
   * check ind_maps correctly ordered within each block
   * and indices within range
   */

  err = 0;


  for ( int m = 0; m < ninds; m++ )
  {
    int m2 = 0;
    while ( inds[m2] != m )
      m2++;
    if (OP_plan.maps[m2] == NULL) continue; //it is a deactivated optional argument
    int halo_size = (OP_plan.maps[m2]->to)->exec_size +
      (OP_plan.maps[m2]->to)->nonexec_size;
    int set_size = OP_plan.maps[m2]->to->size + halo_size;

    ntot = 0;

    for ( int n = 0; n < nblock; n++ )
    {
      int last = -1;
      for ( int e = ntot; e < ntot + OP_plan.ind_sizes[m + n * ninds]; e++ )
      {
        err += ( OP_plan.ind_maps[m][e] <= last );
        last = OP_plan.ind_maps[m][e];
      }
      err += ( last >= set_size );
      ntot += OP_plan.ind_sizes[m + n * ninds];
    }
  }

  if ( err != 0 )
  {
    printf ( " *** OP_plan_check: ind_maps error \n" );
  }
  else if ( OP_diags > 6 )
  {
    printf ( " *** OP_plan_check: ind_maps OK \n" );
  }

  /*
   *check maps (most likely source of errors)
   */



  err = 0;

  for ( int m = 0; m < OP_plan.nargs; m++ )
  {
    if ( OP_plan.maps[m] != NULL && OP_plan.optflags[m] && OP_plan.loc_maps[m] != NULL)
    {
      op_map map = OP_plan.maps[m];
      int m2 = inds[m];

      ntot = 0;
      for ( int n = 0; n < nblock; n++ )
      {
        for ( int e = ntot; e < ntot + OP_plan.nelems[n]; e++ )
        {
          int p_local = OP_plan.loc_maps[m][e];
          int p_global = OP_plan.ind_maps[m2][p_local + OP_plan.ind_offs[m2 + n * ninds]];
          err += ( p_global != map->map[OP_plan.idxs[m] + e * map->dim] );
        }
        ntot += OP_plan.nelems[n];
      }
    }
  }

  if ( err != 0 )
  {
    printf ( " *** OP_plan_check: %d maps error(s) \n", err );
  }
  else if ( OP_diags > 6 )
  {
    printf ( " *** OP_plan_check: maps     OK \n" );
  }

  /*
   * check thread and block coloring
   */

  return;
}

/*
 * OP plan construction
 */

op_plan *op_plan_core(char const *name, op_set set, int part_size,
                      int nargs, op_arg *args, int ninds, int *inds, int staging )
{
  //set exec length
  int exec_length = set->size;
  for(int i = 0; i< nargs; i++)
  {
    if(args[i].opt && args[i].idx != -1 && args[i].acc != OP_READ )
    {
      exec_length += set->exec_size;
      break;
    }
  }

  /* first look for an existing execution plan */

  int ip = 0, match = 0;

  while ( match == 0 && ip < OP_plan_index )
  {
    if ( ( strcmp ( name, OP_plans[ip].name ) == 0 )
        && ( set == OP_plans[ip].set )
        && ( nargs == OP_plans[ip].nargs )
        && ( ninds == OP_plans[ip].ninds )
        && ( part_size == OP_plans[ip].part_size ) )
    {
      match = 1;
      for ( int m = 0; m < nargs; m++ )
      {
        if (args[m].dat != NULL && OP_plans[ip].dats[m] != NULL)
          match = match
          && ( args[m].dat->size == OP_plans[ip].dats[m]->size )
          && ( args[m].dat->dim == OP_plans[ip].dats[m]->dim )
          && ( args[m].map == OP_plans[ip].maps[m] )
          && ( args[m].idx == OP_plans[ip].idxs[m] )
          && ( args[m].acc == OP_plans[ip].accs[m] );
        else
          match = match
          && ( args[m].dat == OP_plans[ip].dats[m] )
          && ( args[m].map == OP_plans[ip].maps[m] )
          && ( args[m].idx == OP_plans[ip].idxs[m] )
          && ( args[m].acc == OP_plans[ip].accs[m] );
      }
    }
    ip++;
  }

  if ( match )
  {
    ip--;
    if ( OP_diags > 3 )
      printf ( " old execution plan #%d\n", ip );
    OP_plans[ip].count++;
    return &( OP_plans[ip] );
  }
  else
  {
    if ( OP_diags > 1 )
      printf ( " new execution plan #%d for kernel %s\n", ip, name );
  }
  double wall_t1, wall_t2, cpu_t1, cpu_t2;
  op_timers_core(&cpu_t1, &wall_t1);
  /* work out worst case shared memory requirement per element */


  int halo_exchange = 0;
  for(int i = 0; i< nargs; i++)
  {
    if(args[i].opt && args[i].idx != -1 && args[i].acc != OP_WRITE && args[i].acc != OP_INC  )
    {
      halo_exchange = 1;
      break;
    }
  }

  int maxbytes = 0;
  for ( int m = 0; m < nargs; m++ )
  {
    if ( args[m].opt && inds[m] >= 0 ) {
      if ((staging == OP_STAGE_INC && args[m].acc == OP_INC) ||
          (staging == OP_STAGE_ALL || staging == OP_STAGE_PERMUTE))
          maxbytes += args[m].dat->size;
    }
  }

  /* set blocksize and number of blocks; adaptive size based on 48kB of shared memory */

  int bsize = part_size;        // blocksize
  if ( bsize == 0 && maxbytes > 0 )
    bsize = MAX(( 24 * 1024 / ( 64 * maxbytes ) ) * 64, 256); //48kB exactly is too much, make it 24
  else if (bsize == 0 && maxbytes == 0)
    bsize = 256;

  //If we do 1 level of coloring, do it in one go
  if (staging == OP_COLOR2) bsize = exec_length;
  
  int nblocks = 0;

  int indirect_reduce = 0;
  for ( int m = 0; m < nargs; m++ ) {
    indirect_reduce |= (args[m].acc != OP_READ && args[m].argtype == OP_ARG_GBL);
  }
  indirect_reduce &= (ninds>0);

  /* Work out indirection arrays for OP_INCs */
  int ninds_staged = 0; //number of distinct (unique dat) indirect incs
  int *inds_staged = (int *)op_malloc(nargs*sizeof(int));
  int *inds_to_inds_staged = (int *)op_malloc(ninds*sizeof(int));

  for (int i = 0; i < nargs; i++) inds_staged[i] = -1;
  for (int i = 0; i < ninds; i++) inds_to_inds_staged[i] = -1;
  for (int i = 0; i < nargs; i++) {
    if (inds[i]>=0 && ((staging == OP_STAGE_INC && args[i].acc == OP_INC) ||
        (staging == OP_STAGE_ALL || staging == OP_STAGE_PERMUTE))) {
      if (inds_to_inds_staged[inds[i]] == -1) {
        inds_to_inds_staged[inds[i]] = ninds_staged;
        inds_staged[i] = ninds_staged;
        ninds_staged++;
      } else {
        inds_staged[i] = inds_to_inds_staged[inds[i]];
      }
    }
  }

  int *invinds_staged = (int *)op_malloc(ninds_staged*sizeof(int));
  for (int i = 0; i < ninds_staged; i++) invinds_staged[i] = -1;
  for (int i = 0; i < nargs; i++)
    if (inds[i]>=0 && ((staging == OP_STAGE_INC && args[i].acc == OP_INC) ||
        (staging == OP_STAGE_ALL || staging == OP_STAGE_PERMUTE)) && invinds_staged[inds_staged[i]] == -1)
      invinds_staged[inds_staged[i]] = i;

  int prev_offset = 0;
  int next_offset = 0;

  while (next_offset < exec_length)
  {
    prev_offset = next_offset;
    if (prev_offset + bsize >= set->core_size && prev_offset < set->core_size) {
      next_offset = set->core_size;
    } else if (prev_offset + bsize >= set->size && prev_offset < set->size && indirect_reduce) {
      next_offset = set->size;
    } else if (prev_offset + bsize >= exec_length && prev_offset < exec_length) {
      next_offset = exec_length;
    } else {
      next_offset = prev_offset + bsize;
    }
    nblocks++;
  }
  
  //If we do 1 level of coloring, we have a single "block"
  if (staging == OP_COLOR2) {nblocks = 1; prev_offset = 0; next_offset = exec_length;};

  /* enlarge OP_plans array if needed */

  if ( ip == OP_plan_max )
  {
    //printf("allocating more memory for OP_plans %d\n", OP_plan_max);
    OP_plan_max += 10;
    OP_plans = ( op_plan * ) op_realloc ( OP_plans, OP_plan_max * sizeof ( op_plan ) );
    if ( OP_plans == NULL )
    {
      printf ( " op_plan error -- error reallocating memory for OP_plans\n" );
      exit ( -1 );
    }
  }

  /* allocate memory for new execution plan and store input arguments */

  OP_plans[ip].dats = ( op_dat * ) op_malloc ( nargs * sizeof ( op_dat ) );
  OP_plans[ip].idxs = ( int * ) op_malloc ( nargs * sizeof ( int ) );
  OP_plans[ip].optflags = ( int * ) op_malloc ( nargs * sizeof ( int ) );
  OP_plans[ip].maps = ( op_map * ) op_malloc ( nargs * sizeof ( op_map ) );
  OP_plans[ip].accs = ( op_access * ) op_malloc ( nargs * sizeof ( op_access ) );
  OP_plans[ip].inds_staged = ( op_access * ) op_malloc ( ninds_staged * sizeof ( op_access ) );

  OP_plans[ip].nthrcol = ( int * ) op_malloc ( nblocks * sizeof ( int ) );
  OP_plans[ip].thrcol = ( int * ) op_malloc ( exec_length * sizeof ( int ) );
  OP_plans[ip].col_reord = ( int * ) op_malloc ( (exec_length+16) * sizeof ( int ) );
  OP_plans[ip].col_offsets = NULL;
  OP_plans[ip].offset = ( int * ) op_malloc ( nblocks * sizeof ( int ) );
  OP_plans[ip].ind_maps = ( int ** ) op_malloc ( ninds_staged * sizeof ( int * ) );
  OP_plans[ip].ind_offs = ( int * ) op_malloc ( nblocks * ninds_staged * sizeof ( int ) );
  OP_plans[ip].ind_sizes = ( int * ) op_malloc ( nblocks * ninds_staged * sizeof ( int ) );
  OP_plans[ip].nindirect = ( int * ) op_calloc ( ninds, sizeof ( int ) );
  OP_plans[ip].loc_maps = ( short ** ) op_malloc ( nargs * sizeof ( short * ) );
  OP_plans[ip].nelems = ( int * ) op_malloc ( nblocks * sizeof ( int ) );
  OP_plans[ip].ncolblk = ( int * ) op_calloc ( exec_length, sizeof ( int ) );  /* max possibly needed */
  OP_plans[ip].blkmap = ( int * ) op_calloc ( nblocks, sizeof ( int ) );

  int *offsets = (int *)op_malloc((ninds_staged+1)*sizeof(int));
  offsets[0] = 0;
  for ( int m = 0; m < ninds_staged; m++ ) {
    int count = 0;
    for ( int m2 = 0; m2 < nargs; m2++ )
      if ( inds_staged[m2] == m )
        count++;
      offsets[m+1] = offsets[m] + count;
  }
  OP_plans[ip].ind_map = ( int * ) op_malloc ( offsets[ninds_staged] * exec_length * sizeof ( int ) );
  for ( int m = 0; m < ninds_staged; m++ ) {
    OP_plans[ip].ind_maps[m] = &OP_plans[ip].ind_map[exec_length*offsets[m]];
  }
  free(offsets);

  int counter = 0;
  for ( int m = 0; m < nargs; m++ ) {
    if ( inds_staged[m] >= 0 )
      counter++;
    else
      OP_plans[ip].loc_maps[m] = NULL;

    OP_plans[ip].dats[m] = args[m].dat;
    OP_plans[ip].idxs[m] = args[m].idx;
    OP_plans[ip].optflags[m] = args[m].opt;
    OP_plans[ip].maps[m] = args[m].map;
    OP_plans[ip].accs[m] = args[m].acc;
  }

  OP_plans[ip].loc_map = ( short * ) op_malloc ( counter * exec_length * sizeof ( short ) );
  counter = 0;
  for ( int m = 0; m < nargs; m++ ) {
    if ( inds_staged[m] >= 0 ) {
      OP_plans[ip].loc_maps[m] = &OP_plans[ip].loc_map[exec_length*(counter)];
      counter++;
    }
  }


  OP_plans[ip].name = name;
  OP_plans[ip].set = set;
  OP_plans[ip].nargs = nargs;
  OP_plans[ip].ninds = ninds;
  OP_plans[ip].ninds_staged = ninds_staged;
  OP_plans[ip].part_size = part_size;
  OP_plans[ip].nblocks = nblocks;
  OP_plans[ip].ncolors_core = 0;
  OP_plans[ip].ncolors_owned = 0;
  OP_plans[ip].count = 1;
  OP_plans[ip].inds_staged = inds_staged;

  OP_plan_index++;

  /* define aliases */

  op_dat * dats = OP_plans[ip].dats;
  int * idxs = OP_plans[ip].idxs;
  op_map * maps = OP_plans[ip].maps;
  op_access * accs = OP_plans[ip].accs;

  int * offset = OP_plans[ip].offset;
  int * nelems = OP_plans[ip].nelems;
  int ** ind_maps = OP_plans[ip].ind_maps;
  int * ind_offs = OP_plans[ip].ind_offs;
  int * ind_sizes = OP_plans[ip].ind_sizes;
  int * nindirect = OP_plans[ip].nindirect;

  /* allocate working arrays */
  uint **work;
  work = (uint **)op_malloc(ninds * sizeof(uint *));

  for ( int m = 0; m < ninds; m++ )
  {
    int m2 = 0;
    while ( inds[m2] != m )
      m2++;
    if (args[m2].opt == 0) {
      work[m] = NULL;
      continue;
    }

    int to_size = (maps[m2]->to)->exec_size + (maps[m2]->to)->nonexec_size + (maps[m2]->to)->size;
    work[m] = ( uint * )op_malloc( to_size * sizeof (uint));
  }

  int *work2;
  work2 = ( int * ) op_malloc ( nargs * bsize * sizeof ( int ) );  /* max possibly needed */

  /* process set one block at a time */

  float total_colors = 0;

  prev_offset = 0;
  next_offset = 0;
  for ( int b = 0; b < nblocks; b++ )
  {
    prev_offset = next_offset;
    if (prev_offset + bsize >= set->core_size && prev_offset < set->core_size) {
      next_offset = set->core_size;
    } else if (prev_offset + bsize >= set->size && prev_offset < set->size && indirect_reduce) {
      next_offset = set->size;
    } else if (prev_offset + bsize >= exec_length && prev_offset < exec_length) {
      next_offset = exec_length;
    } else {
      next_offset = prev_offset + bsize;
    }
    
    if (staging == OP_COLOR2) {prev_offset = 0; next_offset = exec_length;};
    int bs = next_offset-prev_offset;

    offset[b] = prev_offset;  /* offset for block */
    nelems[b] = bs;   /* size of block */

    /* loop over indirection sets */
    for ( int m = 0; m < ninds; m++ )
    {
      int m2 = 0;
      while ( inds[m2] != m )
        m2++;
      int m3 = inds_staged[m2];
      if (m3 < 0) continue;
      if (args[m2].opt == 0) {
        if (b==0) {
          ind_offs[m3+b*ninds_staged] = 0;
          ind_sizes[m3+b*ninds_staged] = 0;
        } else {
          ind_offs[m3+b*ninds_staged] = ind_offs[m3+(b-1)*ninds_staged];
          ind_sizes[m3+b*ninds_staged] = 0;
        }
        continue;
      }
      /* build the list of elements indirectly referenced in this block */

      int ne = 0;/* number of elements */
      for ( int m2 = 0; m2 < nargs; m2++ )
      {
        if ( inds[m2] == m )
        {
          for ( int e = prev_offset; e < next_offset; e++ )
            work2[ne++] = maps[m2]->map[idxs[m2] + e * maps[m2]->dim];
        }
      }

      /* sort them, then eliminate duplicates */

      qsort(work2, ne, sizeof(int), comp);

      int nde = 0;
      int p = 0;
      while ( p < ne )
      {
        work2[nde] = work2[p];
        while ( p < ne && work2[p] == work2[nde] )
          p++;
        nde++;
      }
      ne = nde; /* number of distinct elements */

      /*
         if (OP_diags > 5) { printf(" indirection set %d: ",m); for (int e=0; e<ne; e++) printf("
         %d",work2[e]); printf(" \n"); } */


      /* store mapping and renumbered mappings in execution plan */

      for ( int e = 0; e < ne; e++ )
      {
        ind_maps[m3][nindirect[m]++] = work2[e];
        work[m][work2[e]] = e;  // inverse mapping
      }

      for ( int m2 = 0; m2 < nargs; m2++ )
      {
        if ( inds[m2] == m )
        {
          for ( int e = prev_offset; e < next_offset; e++ )
            OP_plans[ip].loc_maps[m2][e] = (short)(work[m][maps[m2]->map[idxs[m2] + e * maps[m2]->dim]]);
        }
      }

      if ( b == 0 )
      {
        ind_offs[m3 + b * ninds_staged] = 0;
        ind_sizes[m3 + b * ninds_staged] = nindirect[m];
      }
      else
      {
        ind_offs[m3 + b * ninds_staged] = ind_offs[m3 + ( b - 1 ) * ninds_staged]
          + ind_sizes[m3 + ( b - 1 ) * ninds_staged];
        ind_sizes[m3 + b * ninds_staged] = nindirect[m] - ind_offs[m3 + b * ninds_staged];
      }
    }

    /* now colour main set elements */

    for ( int e = prev_offset; e < next_offset; e++ )
      OP_plans[ip].thrcol[e] = -1;

    int repeat = 1;
    int ncolor = 0;
    int ncolors = 0;

    while ( repeat )
    {
      repeat = 0;

      for ( int m = 0; m < nargs; m++ )
      {
        if ( inds[m] >= 0 && args[m].opt )
          for ( int e = prev_offset; e < next_offset; e++ )
            work[inds[m]][maps[m]->map[idxs[m] + e * maps[m]->dim]] = 0; /* zero out color array */
      }

      for ( int e = prev_offset; e < next_offset; e++ )
      {
        if ( OP_plans[ip].thrcol[e] == -1 )
        {
          int mask = 0;
          if (staging==OP_COLOR2 && halo_exchange && e >= set->core_size && ncolor==0) mask = 1;
          for ( int m = 0; m < nargs; m++ )
            if ( inds[m] >= 0 && (accs[m] == OP_INC || accs[m] == OP_RW) && args[m].opt )
              mask |= work[inds[m]][maps[m]->map[idxs[m] + e * maps[m]->dim]]; /* set bits of mask */

          int color = ffs ( ~mask ) - 1;  /* find first bit not set */
          if ( color == -1 )
          {                     /* run out of colors on this pass */
            repeat = 1;
          }
          else
          {
            OP_plans[ip].thrcol[e] = ncolor + color;
            mask = 1 << color;
            ncolors = MAX ( ncolors, ncolor + color + 1 );

            for ( int m = 0; m < nargs; m++ )
              if ( inds[m] >= 0 && (accs[m] == OP_INC || accs[m] == OP_RW) && args[m].opt )
                work[inds[m]][maps[m]->map[idxs[m] + e * maps[m]->dim]] |= mask; /* set color bit */
          }
        }
      }

      ncolor += 32;             /* increment base level */
    }

    OP_plans[ip].nthrcol[b] = ncolors;  /* number of thread colors in this block */
    total_colors += ncolors;

    //if(ncolors>1) printf(" number of colors in this block = %d \n",ncolors);
  }

  /* create element permutation by color */
  if (staging == OP_STAGE_PERMUTE || staging == OP_COLOR2 ) {
    int size_of_col_offsets = 0;
    for (int b = 0; b < nblocks; b++) {
      size_of_col_offsets += OP_plans[ip].nthrcol[b]+1;
    }
    //allocate
    OP_plans[ip].col_offsets = (int **)op_malloc(nblocks*sizeof(int*));
    int *col_offsets = (int *)op_malloc(size_of_col_offsets*sizeof(int*));
    
    size_of_col_offsets = 0;
    op_keyvalue *kv = (op_keyvalue *)op_malloc(bsize * sizeof(op_keyvalue));
    for (int b = 0; b < nblocks; b++) {
      int ncolor = OP_plans[ip].nthrcol[b];
      for (int e = 0; e < nelems[b]; e++) {
        kv[e].key = OP_plans[ip].thrcol[offset[b]+e];
        kv[e].value = e;
      }
      qsort ( kv, nelems[b], sizeof ( op_keyvalue ), comp2 );
      OP_plans[ip].col_offsets[b] = col_offsets+size_of_col_offsets;
      OP_plans[ip].col_offsets[b][0] = 0;
      size_of_col_offsets+=(ncolor+1);
      
      //Set up permutation and pointers to beginning of each color
      ncolor = 0;
      for (int e = 0; e < nelems[b]; e++) {
        OP_plans[ip].thrcol[offset[b]+e] = kv[e].key;
        OP_plans[ip].col_reord[offset[b]+e] = kv[e].value;
        if (e > 0)
          if (kv[e].key > kv[e-1].key) {ncolor++; OP_plans[ip].col_offsets[b][ncolor]=e;}
      }
      OP_plans[ip].col_offsets[b][ncolor+1] = nelems[b];
    }
    for (int i = exec_length; i < exec_length+16; i++) OP_plans[ip].col_reord[i] = 0;
  }

  /* color the blocks, after initialising colors to 0 */

  int * blk_col;

  blk_col = ( int * ) op_malloc ( nblocks * sizeof ( int ) );
  for ( int b = 0; b < nblocks; b++ )
    blk_col[b] = -1;

  int repeat = 1;
  int ncolor = 0;
  int ncolors = 0;

  while ( repeat )
  {
    repeat = 0;

    for ( int m = 0; m < nargs; m++ )
    {
      if ( inds[m] >= 0 && args[m].opt)
      {
        int to_size = (maps[m]->to)->exec_size + (maps[m]->to)->nonexec_size + (maps[m]->to)->size;
        for ( int e = 0; e < to_size; e++ )
          work[inds[m]][e] = 0; // zero out color arrays
      }
    }
    prev_offset = 0;
    next_offset = 0;
    for ( int b = 0; b < nblocks; b++ )
    {
      prev_offset = next_offset;

      if (prev_offset + bsize >= set->core_size && prev_offset < set->core_size) {
        next_offset = set->core_size;
      } else if (prev_offset + bsize >= set->size && prev_offset < set->size && indirect_reduce) {
        next_offset = set->size;
      } else if (prev_offset + bsize >= exec_length && prev_offset < exec_length) {
        next_offset = exec_length;
      } else {
        next_offset = prev_offset + bsize;
      }
      if ( blk_col[b] == -1 )
      { // color not yet assigned to block
        uint mask = 0;
        if (next_offset > set->core_size) { //should not use block colors from the core set when doing the non_core ones
          if (prev_offset <= set->core_size) OP_plans[ip].ncolors_core = ncolors;
          for (int shifter = 0; shifter < OP_plans[ip].ncolors_core; shifter++) mask |= 1<<shifter;
          if (prev_offset == set->size && indirect_reduce) OP_plans[ip].ncolors_owned = ncolors;
          for (int shifter = OP_plans[ip].ncolors_core; indirect_reduce && shifter < OP_plans[ip].ncolors_owned; shifter++) mask |= 1<<shifter;
        }

        for ( int m = 0; m < nargs; m++ )
        {
          if ( inds[m] >= 0 && (accs[m] == OP_INC || accs[m] == OP_RW) && args[m].opt )
            for ( int e = prev_offset; e < next_offset; e++ )
              mask |= work[inds[m]][maps[m]->map[idxs[m] +
                e * maps[m]->dim]]; // set bits of mask
        }

        int color = ffs( ~mask ) - 1; // find first bit not set
        if ( color == -1 )
        { //run out of colors on this pass
          repeat = 1;
        }
        else
        {
          blk_col[b] = ncolor + color;
          mask = 1 << color;
          ncolors = MAX( ncolors, ncolor + color + 1 );

          for ( int m = 0; m < nargs; m++ )
          {
            if ( inds[m] >= 0 && (accs[m] == OP_INC || accs[m] == OP_RW) && args[m].opt )
              for ( int e = prev_offset; e < next_offset; e++ )
                work[inds[m]][maps[m]->map[idxs[m] +
                  e * maps[m]->dim]] |= mask;
          }
        }
      }
    }

    ncolor += 32;               // increment base level
  }

  /* store block mapping and number of blocks per color */

  if (indirect_reduce && OP_plans[ip].ncolors_owned == 0) OP_plans[ip].ncolors_owned = ncolors; //no MPI, so get the reduction arrays after everyting is done
  OP_plans[ip].ncolors = ncolors;
  if (staging==OP_COLOR2) OP_plans[ip].ncolors = OP_plans[ip].nthrcol[0];

  /*for(int col = 0; col = OP_plans[ip].ncolors;col++) //should initialize to zero because op_calloc returns garbage!!
    {
    OP_plans[ip].ncolblk[col] = 0;
    }*/

  for ( int b = 0; b < nblocks; b++ )
    OP_plans[ip].ncolblk[blk_col[b]]++; // number of blocks of each color

  for ( int c = 1; c < ncolors; c++ )
    OP_plans[ip].ncolblk[c] += OP_plans[ip].ncolblk[c - 1]; // cumsum

  for ( int c = 0; c < ncolors; c++ )
    work2[c] = 0;

  for ( int b = 0; b < nblocks; b++ )
  {
    int c = blk_col[b];
    int b2 = work2[c]; // number of preceding blocks of this color
    if ( c > 0 )
      b2 += OP_plans[ip].ncolblk[c - 1];  // plus previous colors

    OP_plans[ip].blkmap[b2] = b;

    work2[c]++; //increment counter
  }

  for ( int c = ncolors - 1; c > 0; c-- )
    OP_plans[ip].ncolblk[c] -= OP_plans[ip].ncolblk[c - 1]; // undo cumsum


  /* reorder blocks by color? */

  /* work out shared memory requirements */
  OP_plans[ip].nsharedCol = (int *)op_malloc(ncolors * sizeof(int));
  float total_shared = 0;
  for (int col = 0; col < ncolors; col++) {
    OP_plans[ip].nsharedCol[col] = 0;
    for ( int b = 0; b < nblocks; b++ ) {
      if (blk_col[b] ==  col) {
        int nbytes = 0;
        for ( int m = 0; m < ninds_staged; m++ )  {
          int m2 = 0;
          while ( inds_staged[m2] != m )
            m2++;
          if (args[m2].opt==0) continue;

          nbytes += ROUND_UP_64 ( ind_sizes[m + b * ninds_staged] * dats[m2]->size );
        }
        OP_plans[ip].nsharedCol[col] = MAX ( OP_plans[ip].nsharedCol[col], nbytes );
        total_shared += nbytes;
      }
    }
  }

  OP_plans[ip].nshared = 0;
  total_shared = 0;

  for ( int b = 0; b < nblocks; b++ )
  {
    int nbytes = 0;
    for ( int m = 0; m < ninds_staged; m++ )
    {
      int m2 = 0;
      while ( inds_staged[m2] != m )
        m2++;
      if (args[m2].opt==0) continue;

      nbytes += ROUND_UP_64 ( ind_sizes[m + b * ninds_staged] * dats[m2]->size );
    }
    OP_plans[ip].nshared = MAX ( OP_plans[ip].nshared, nbytes );
    total_shared += nbytes;
  }

  /* work out total bandwidth requirements */

  OP_plans[ip].transfer = 0;
  OP_plans[ip].transfer2 = 0;
  float transfer3 = 0;

  if (staging != OP_COLOR2 && staging != OP_STAGE_INC) {
    for ( int b = 0; b < nblocks; b++ )
    {
      for ( int m = 0; m < nargs; m++ ) //for each argument
      {
        if (args[m].opt) {
          if ( inds[m] < 0 ) //if it is directly addressed
          {
            float fac = 2.0f;
            if ( accs[m] == OP_READ || accs[m] == OP_WRITE ) //if you only read or write it
              fac = 1.0f;
            if ( dats[m] != NULL )
            {
              OP_plans[ip].transfer += fac * nelems[b] * dats[m]->size; //cost of reading it all
              OP_plans[ip].transfer2 += fac * nelems[b] * dats[m]->size;
              transfer3 += fac * nelems[b] * dats[m]->size;
            }
          }
          else //if it is indirectly addressed: cost of reading the pointer to it
          {
            OP_plans[ip].transfer += nelems[b] * sizeof ( short );
            OP_plans[ip].transfer2 += nelems[b] * sizeof ( short );
            transfer3 += nelems[b] * sizeof ( short );
          }
        }
      }
      for ( int m = 0; m < ninds; m++ ) //for each indirect mapping
      {
        int m2 = 0;
        while ( inds[m2] != m ) //find the first argument that uses this mapping
          m2++;
        if ( args[m2].opt == 0 ) continue;
        float fac = 2.0f;
        if ( accs[m2] == OP_READ || accs[m2] == OP_WRITE) //only read it
          fac = 1.0f;
        if (staging == OP_STAGE_INC && accs[m2] != OP_INC){
          OP_plans[ip].transfer += 1;
          OP_plans[ip].transfer2 += 1;
          continue;  
        }
        OP_plans[ip].transfer += fac * ind_sizes[m + b * ninds] * dats[m2]->size; //simply read all data one by one
  
        /* work out how many cache lines are used by indirect addressing */
  
        int i_map, l_new, l_old;
        int e0 = ind_offs[m + b * ninds]; //where it starts
        int e1 = e0 + ind_sizes[m + b * ninds]; //where it ends
  
        l_old = -1;
  
        for ( int e = e0; e < e1; e++ ) //iterate through every indirectly accessed data element
        {
          i_map = ind_maps[m][e]; //the pointer to the data element
          l_new = ( i_map * dats[m2]->size ) / OP_cache_line_size; //which cache line it is on (full size, dim*sizeof(type))
          if ( l_new > l_old ) //if it is on a further cache line (that is not yet loaded, - i_map is ordered)
            OP_plans[ip].transfer2 += fac * OP_cache_line_size; //load the cache line
          l_old = l_new;
          l_new = ( ( i_map + 1 ) * dats[m2]->size - 1 ) / OP_cache_line_size; //the last byte of the data
          OP_plans[ip].transfer2 += fac * ( l_new - l_old ) * OP_cache_line_size; //again, if not loaded, load it (can be multiple cache lines)
          l_old = l_new;
        }
  
        l_old = -1;
  
        for ( int e = e0; e < e1; e++ )
        {
          i_map = ind_maps[m][e]; //pointer to the data element
          l_new = ( i_map * dats[m2]->size ) / ( dats[m2]->dim * OP_cache_line_size ); //which cache line the first dimension of the data is on
          if ( l_new > l_old )
            transfer3 += fac * dats[m2]->dim * OP_cache_line_size; //if not loaded yet, load all cache lines
          l_old = l_new;
          l_new = ( ( i_map + 1 ) * dats[m2]->size - 1 ) / ( dats[m2]->dim * OP_cache_line_size ); //primitve type's last byte
          transfer3 += fac * ( l_new - l_old ) * dats[m2]->dim * OP_cache_line_size; //load it
          l_old = l_new;
        }
  
        /* also include mappings to load/store data */
  
        fac = 1.0f;
        if ( accs[m2] == OP_RW )
          fac = 2.0f;
        OP_plans[ip].transfer += fac * ind_sizes[m + b * ninds] * sizeof ( int );
        OP_plans[ip].transfer2 += fac * ind_sizes[m + b * ninds] * sizeof ( int );
        transfer3 += fac * ind_sizes[m + b * ninds] * sizeof ( int );
      }
    }
  }

  /* print out useful information */

  if ( OP_diags > 1 )
  {
    printf( " number of blocks       = %d \n", nblocks );
    printf( " number of block colors = %d \n", OP_plans[ip].ncolors );
    printf( " maximum block size     = %d \n", bsize );
    printf( " average thread colors  = %.2f \n", total_colors / nblocks );
    printf( " shared memory required = ");
    for (int i = 0; i < ncolors-1; i++) printf(" %.2f KB,", OP_plans[ip].nsharedCol[i] / 1024.0f );
    printf(" %.2f KB\n", OP_plans[ip].nsharedCol[ncolors-1] / 1024.0f );
    printf( " average data reuse     = %.2f \n", maxbytes * ( exec_length / total_shared ) );
    printf( " data transfer (used)   = %.2f MB \n",
        OP_plans[ip].transfer / ( 1024.0f * 1024.0f ) );
    printf( " data transfer (total)  = %.2f MB \n",
        OP_plans[ip].transfer2 / ( 1024.0f * 1024.0f ) );
    printf( " SoA/AoS transfer ratio = %.2f \n\n", transfer3 / OP_plans[ip].transfer2 );
  }

  /* validate plan info */

  op_plan_check ( OP_plans[ip], ninds_staged, inds_staged );

  /* free work arrays */

  for ( int m = 0; m < ninds; m++ )
    free ( work[m] );
  free ( work );
  free ( work2 );
  free ( blk_col );
  free(inds_to_inds_staged);
  free(invinds_staged);
  op_timers_core(&cpu_t2, &wall_t2);
  for (int i = 0; i < OP_kern_max; i++) {
    if (strcmp(name, OP_kernels[i].name)==0) {
      OP_kernels[i].plan_time += wall_t2-wall_t1;
      break;
    }
  }
  /* return pointer to plan */
  OP_plan_time += wall_t2-wall_t1;
  return &( OP_plans[ip] );
}
