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

#include <mpi.h>

#include <op_lib_core.h>
#include <op_lib_cpp.h>
#include <op_util.h>
#include <op_mpi_core.h>
#include <vector>

#ifdef HAVE_PTSCOTCH
#include <scotch.h>
#endif

typedef struct {
int a;
int b;
} map2;

int compare (const void * a, const void * b)
{
  return ( (*(map2*)a).a - (*(map2*)b).a );
}



//propage renumbering based on a map that points to an already reordered set
void propagate_reordering(op_set from, op_set to, std::vector<std::vector<int> >& set_permutations, std::vector<std::vector<int> >& set_ipermutations) {
  if (set_permutations[to->index].size()) return;
  //find a map that is (to)->(from), reorder (to)
  for (int mapidx = 0; mapidx < OP_map_index; mapidx++) {
    op_map map = OP_map_list[mapidx];
    if (map->to == from && map->from == to) {
      std::vector<map2> renum(to->size);
      for (int i = 0; i < to->size; i++) {
        renum[i].a = set_permutations[from->index][map->map[map->dim*i]];
        renum[i].b = i;
      }
      qsort(&renum[0], renum.size(), sizeof(map2), compare);
      set_permutations[to->index].resize(to->size);
      for (int i = 0; i < to->size; i++) set_permutations[to->index][renum[i].b] = i;
      break;
    }
  }
  if (set_permutations[to->index].size()==0) {
    printf("Could not find suitable mapping to renumber %s based on %s\n", to->name, from->name);
    return;
  }
  //find any maps that is (*)->to, propagate reordering
  for (int mapidx = 0; mapidx < OP_map_index; mapidx++) {
    op_map map = OP_map_list[mapidx];
    if (map->to == to) {
      propagate_reordering(to, map->from, set_permutations, set_ipermutations);
    }
  }
}

void reorder_set(op_set set, std::vector<std::vector<int> >& set_permutations, std::vector<std::vector<int> >& set_ipermutations) {
  if (set_permutations[set->index].size()==0) {
    printf("No reordering for set %s, skipping...\n", set->name);
    return;
  }
  for (int mapidx = 0; mapidx < OP_map_index; mapidx++) {
    op_map map = OP_map_list[mapidx];
    if (map->from == set) {
      int *tempmap = (int *)malloc(set->size*sizeof(int)*map->dim);
      for (int i = 0; i < set->size; i++) std::copy(map->map + map->dim*i, map->map + map->dim*(i+1), tempmap + map->dim*set_permutations[set->index][i]);
      free(map->map);
      map->map = tempmap;
    } else if (map->to == set) {
      for (int i = 0; i < map->from->size*map->dim; i++) map->map[i] = set_permutations[set->index][map->map[i]];
    }
  }
  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    if (dat->set == set) {
      char *tempdata = (char *)malloc(set->size*dat->size);
      for (int i = 0; i < set->size; i++) std::copy(dat->data + dat->size*i, dat->data + dat->size*(i+1), tempdata + dat->size*set_permutations[set->index][i]);
      free(dat->data);
      dat->data = tempdata;
    }
  }
}

void op_renumber(op_map base) {
#ifndef HAVE_PTSCOTCH
  printf("OP2 was not compiled with Scotch, no reordering.\n");
#else
  printf("Renumbering using base map %s\n", base->name);
  int nranks;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  if (nranks>1) {
    printf("Renumbering only works with 1 rank\n");
    exit(-1);
  }

//-----------------------------------------------------------------------------------------
// Build adjacency list
//-----------------------------------------------------------------------------------------
  std::vector<int> row_offsets(base->to->size+1);
  std::vector<int> col_indices;
  if (base->to == base->from) {
    //if map is self-referencing, just create row_offsets with dim stride, and copy over col_indices
    std::copy(base->map, base->map+base->dim*base->from->size, col_indices.begin());
    row_offsets[0] = 0;
    for (int i = 1; i < base->from->size+1; i++) row_offsets[i] = i*base->dim;
  } else {
    //otherwise, construct self-referencing map
    col_indices.resize(base->from->size * (base->dim-1) * (base->dim)); //Worst case memory requirement

    //construct map pointing back
    std::vector<map2> loopback(base->from->size * base->dim);
    for (int i = 0; i < base->from->size*base->dim; i++) {
      loopback[i].a = base->map[i];
      loopback[i].b = i/base->dim;
    }
    qsort(&loopback[0], loopback.size(), sizeof(map2), compare);

    row_offsets[0] = 0;
    row_offsets[1] = 0;
    row_offsets[base->to->size] = 0;
    for (int i = 0; i < base->dim; i++) if (base->map[base->dim*loopback[0].b+i] != 0) col_indices[row_offsets[1]++] = base->map[base->dim*loopback[0].b+i];
    int nodectr = 0;
    for (int i = 1; i < base->from->size * base->dim; i++) {
      if (loopback[i].a != loopback[i-1].a) {nodectr++; row_offsets[nodectr+1] = row_offsets[nodectr];}
      for (int d1 = 0; d1 < base->dim; d1++) {
        int id = base->map[base->dim*loopback[i].b+d1];
        int add = 1;
        for (int d2 = row_offsets[nodectr]; (d2 < row_offsets[nodectr+1]) && add; d2++) {
          if (col_indices[d2] == id || nodectr == id) add = 0;
        }
        if (add) col_indices[row_offsets[nodectr+1]++] = id;
      }
    }
    if (row_offsets[base->to->size] == 0) {
      printf("Map %s is not an onto map from %s to %s, aborting renumbering...\n", base->name, base->from, base->to);
      return;
    }
    col_indices.resize(row_offsets[base->to->size]);
    printf("Loopback map %s->%s constructed: %d, from set %s (%d)\n", base->to->name, base->to->name, (int)col_indices.size(), base->from->name, base->from->size);
  }

  //
  // Using SCOTCH for reordering
  //
  SCOTCH_Num baseval = 0; // start numbering from 0
  SCOTCH_Num vertnbr = base->to->size; // number of vertices in graph = number of cells in mesh
  SCOTCH_Num edgenbr = row_offsets[base->to->size];

  SCOTCH_Graph *graphptr = SCOTCH_graphAlloc();
  SCOTCH_graphInit(graphptr);

  SCOTCH_Num *verttab = &row_offsets[0];

  SCOTCH_Num *vendtab = &verttab[1]; // = NULL; // Used to calculate vertex degree = verttab[i+1] - verttab[i]
  SCOTCH_Num *velotab = NULL; // Vertex load = vertex weight
  SCOTCH_Num *vlbltab = NULL;
  SCOTCH_Num *edgetab = &col_indices[0];

  SCOTCH_Num *edlotab = NULL; // Edge load = edge weight
  SCOTCH_Num *permutation = (SCOTCH_Num*) malloc(base->to->size*sizeof(SCOTCH_Num));
  SCOTCH_Num *ipermutation = (SCOTCH_Num*) malloc(base->to->size*sizeof(SCOTCH_Num));
  SCOTCH_Num *cblkptr = (SCOTCH_Num*) malloc(base->to->size*sizeof(SCOTCH_Num));
  SCOTCH_Num *rangtab = NULL;//(SCOTCH_Num*) malloc(1 + ncell*sizeof(SCOTCH_Num));
  SCOTCH_Num *treetab = NULL;//(SCOTCH_Num*) malloc(ncell*sizeof(SCOTCH_Num));

  int mesg = 0;
  mesg = SCOTCH_graphBuild(graphptr, baseval, vertnbr, verttab, vendtab, velotab, vlbltab, edgenbr, edgetab, edlotab);
  if(mesg != 0){
    op_printf("Error during SCOTCH_graphBuild() \n");
    exit(-1);
  }

  SCOTCH_Strat *straptr = SCOTCH_stratAlloc();
  SCOTCH_stratInit(straptr);

  const char * strategyString = "g";
  //    char * strategyString = "(g{pass=100})";
  mesg = SCOTCH_stratGraphOrder(straptr, strategyString);
  if(mesg != 0){
    op_printf("Error during setting strategy string. \n");
    exit(-1);
  }

  mesg = SCOTCH_graphOrder(graphptr, straptr, permutation, ipermutation, cblkptr, rangtab, treetab);
  if(mesg != 0){
    op_printf("Error during SCOTCH_graphOrder() \n");
    exit(-1);
  }
  SCOTCH_graphExit(graphptr);
  SCOTCH_stratExit(straptr);

  std::vector<std::vector<int> > set_permutations(OP_set_index);
  std::vector<std::vector<int> > set_ipermutations(OP_set_index);
  set_permutations[base->to->index].resize(base->to->size);
  std::copy(permutation, permutation+base->to->size, set_permutations[base->to->index].begin());
  set_ipermutations[base->to->index].resize(base->to->size);
  std::copy(ipermutation, ipermutation+base->to->size, set_ipermutations[base->to->index].begin());
  if (base->from == base->to) {
    //find some set that maps to the reordered set
    for (int i = 0; i < OP_map_index; i++) {
      if (OP_map_list[i]->to == base->to && OP_map_list[i]->from != base->to) {
        propagate_reordering(base->to, OP_map_list[i]->from, set_permutations, set_ipermutations);
        break;
      }
    }
  } else {
    propagate_reordering(base->to, base->from, set_permutations, set_ipermutations);
  }
  for (int i = 0; i < OP_set_index; i++) {
    reorder_set(OP_set_list[i], set_permutations, set_ipermutations);
  }
  #endif
}
