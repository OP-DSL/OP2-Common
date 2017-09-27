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

#include <op_lib_core.h>
#include <op_lib_cpp.h>
#include <op_util.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <op_lib_mpi.h>
#include <op_mpi_core.h>

#ifdef HAVE_PTSCOTCH
#include <scotch.h>
#endif

#ifdef PARMETIS_VER_4
#include <metis.h>
typedef idx_t idxtype;
#endif
extern part *OP_part_list;

typedef struct {
  int a;
  int b;
} map2;

int compare(const void *a, const void *b) {
  return ((*(map2 *)a).a - (*(map2 *)b).a);
}

void check_permutation(int *perm, int size) {
  std::vector<int> flags(size,0);
  for (int i = 0; i < size; i++)
    flags[perm[i]] = 1;
  int acc = 0;
  for (int i = 0; i < size; i++)
    acc += flags[i];
  if (acc != size) printf("Permutation map error\n");
}

// propage renumbering based on a map that points to an already reordered set
void propagate_reordering(op_set from, op_set to,
                          std::vector<std::vector<int> > &set_permutations,
                          std::vector<std::vector<int> > &set_ipermutations) {

  if (to->size == 0)
    return;

  // find a map that is (to)->(from), reorder (to)
  if (set_permutations[to->index].size() == 0) {
    for (int mapidx = 0; mapidx < OP_map_index; mapidx++) {
      op_map map = OP_map_list[mapidx];
      if (map->to == from && map->from == to) {
        std::vector<map2> renum(to->core_size);
        for (int i = 0; i < to->core_size; i++) {
          renum[i].a = set_permutations[from->index][map->map[map->dim * i]];
          renum[i].b = i;
        }
        qsort(&renum[0], renum.size(), sizeof(map2), compare);
        set_permutations[to->index].resize(to->size+to->exec_size+to->nonexec_size);
        for (int i = 0; i < to->core_size; i++)
          set_permutations[to->index][renum[i].b] = i;
        for (int i = to->core_size; i < to->size + to->exec_size + to->nonexec_size; i++)
          set_permutations[to->index][i] = i;
        check_permutation(&set_permutations[to->index][0],to->size + to->exec_size + to->nonexec_size);
        break;
      }
    }
  }
  // find a map that is (from)->(to), reorder (to), if it's an onto map
  if (set_permutations[to->index].size() == 0) {
    for (int mapidx = 0; mapidx < OP_map_index; mapidx++) {
      op_map map = OP_map_list[mapidx];
      if (map->to == to && map->from == from) {
        int counter = 0;
        set_permutations[to->index].resize(to->size+to->exec_size+to->nonexec_size,-1);
        for (int i = 0; i < from->size; i++) {
          for (int d = 0; d < map->dim; d++) {
            int idx = map->map[set_ipermutations[from->index][i]*map->dim + d];
            if (idx < to->core_size && set_permutations[to->index][idx]==-1)
              set_permutations[to->index][idx]=counter++;
          }
        }
        int onto = 1;
        for (int i = 0; i < to->core_size; i++) if (set_permutations[to->index][i]==-1) {onto = 0; break;}
        if (!onto) {
          set_permutations[to->index].resize(0);
          continue;
        }
        for (int i = to->core_size; i < to->size + to->exec_size + to->nonexec_size; i++)
          set_permutations[to->index][i] = i;
        check_permutation(&set_permutations[to->index][0],to->size + to->exec_size + to->nonexec_size);
        break;
      }
    }
  }
  if (set_permutations[to->index].size() == 0) {
    return;
  }
  else {
    set_ipermutations[to->index].resize(to->size+to->exec_size+to->nonexec_size);
    for (int i = 0; i < to->size+to->exec_size+to->nonexec_size; i++) {
      set_ipermutations[to->index][set_permutations[to->index][i]] = i;
    }
  }

  // find any maps that is (*)->to, propagate reordering
  for (int mapidx = 0; mapidx < OP_map_index; mapidx++) {
    op_map map = OP_map_list[mapidx];
    if (map->to == to && set_permutations[map->from->index].size() == 0) {
      propagate_reordering(to, map->from, set_permutations, set_ipermutations);
    }
  }
  // find any maps that is to->(*), propagate reordering
  for (int mapidx = 0; mapidx < OP_map_index; mapidx++) {
    op_map map = OP_map_list[mapidx];
    if (map->from == to && set_permutations[map->to->index].size() == 0) {
      propagate_reordering(to, map->to, set_permutations, set_ipermutations);
    }
  }
}

void reorder_set(op_set set, std::vector<std::vector<int> > &set_permutations,
                 std::vector<std::vector<int> > &set_ipermutations) {

  if (set_permutations[set->index].size() == 0 && set->core_size > 0) {
    printf("No reordering for set %s, skipping...\n", set->name);
    return;
  }

  if (set->size == 0)
    return;

  //Reorder maps
  for (int mapidx = 0; mapidx < OP_map_index; mapidx++) {
    op_map map = OP_map_list[mapidx];
    if (map->from == set) {
      int *tempmap = (int *)malloc((set->size+set->exec_size) * sizeof(int) * map->dim);

      for (int i = 0; i < set->size+set->exec_size; i++)
        std::copy(map->map + map->dim * i, map->map + map->dim * (i + 1),
                  tempmap + map->dim * set_permutations[set->index][i]);
      free(map->map);
      map->map = tempmap;

    } else if (map->to == set) {
      for (int i = 0; i < (map->from->size+map->from->exec_size) * map->dim; i++)
        map->map[i] = set_permutations[set->index][map->map[i]];
    }
  }

  //Reorder datasets
  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    if (dat->set == set && dat->data != NULL) {
      char *tempdata = (char *)malloc((size_t)(set->size+set->exec_size+set->nonexec_size) * (size_t)dat->size);
      for (unsigned long int i = 0; i < (unsigned long int)(set->size+set->exec_size+set->nonexec_size); i++)
        std::copy(dat->data + (unsigned long int)dat->size * i,
                  dat->data + (unsigned long int)dat->size * (i + 1),
                  tempdata +
                      (unsigned long int)dat->size *
                          (unsigned long int)set_permutations[set->index][i]);
      free(dat->data);
      dat->data = tempdata;
    }
  }

  //Renumber halos
  halo_list exp_exec_list = OP_export_exec_list[set->index];
  halo_list exp_nonexec_list = OP_export_nonexec_list[set->index];
  if (exp_exec_list->ranks_size > 0) {
    int halolen = exp_exec_list->disps[exp_exec_list->ranks_size-1] + exp_exec_list->sizes[exp_exec_list->ranks_size-1];
    for (int i = 0; i < halolen; i++) {
      exp_exec_list->list[i] = set_permutations[set->index][exp_exec_list->list[i]];
    }
  }
  if (exp_nonexec_list->ranks_size > 0) {
    int halolen = exp_nonexec_list->disps[exp_nonexec_list->ranks_size-1] + exp_nonexec_list->sizes[exp_nonexec_list->ranks_size-1];
    for (int i = 0; i < halolen; i++) {
      exp_nonexec_list->list[i] = set_permutations[set->index][exp_nonexec_list->list[i]];
    }
  }

  //Reorder mapping back to original (unpartitioned indexing)
  int *new_g_index = (int*)malloc(set->size*sizeof(int));
  for (int i = 0; i < set->size; i++)
    new_g_index[set_permutations[set->index][i]] = OP_part_list[set->index]->g_index[i];
  free(OP_part_list[set->index]->g_index);
  OP_part_list[set->index]->g_index = new_g_index; 
}

void op_renumber(op_map base) {
#ifndef HAVE_PTSCOTCH
  op_printf("OP2 was not compiled with Scotch, no reordering.\n");
#else
  op_printf("Renumbering using base map %s\n", base->name);
  int generated_partvec = 0;
/*
  if (FILE *file = fopen("partvec0001_0001", "r")) {
    fclose(file);
    int possible[] = {1,  2,  4,   6,   8,   12,  16,   22,   24,
                      32, 64, 128, 192, 256, 512, 1024, 2048, 4096};
    for (int i = 0; i < 18; i++) {
      char buffer[64];
      sprintf(buffer, "partvec0001_%04d", possible[i]);
      if (!(file = fopen(buffer, "r")))
        continue;
      printf("Processing partitioning for %d partitions\n", possible[i]);
      int *partvec = (int *)malloc(base->to->size * sizeof(int));
      int *order = (int *)malloc(base->to->size * sizeof(int));
      int total_ctr = 0;
      for (int f = 0; f < possible[i]; f++) {
        if (f > 0) {
          fclose(file);
          sprintf(buffer, "partvec%04d_%04d", f + 1, possible[i]);
          file = fopen(buffer, "r");
        }
        int counter = 0;
        int id, part;
        while (fscanf(file, "%d %d", &id, &part) != EOF) {
          partvec[id] = part - 1;
          order[id] = counter;
          counter++;
          total_ctr++;
        }
      }
      fclose(file);
      sprintf(buffer, "partvec%04d", possible[i]);
      op_decl_dat(base->to, 1, "int", partvec, buffer);
      sprintf(buffer, "ordering%04d", possible[i]);
      op_decl_dat(base->to, 1, "int", order, buffer);
      if (total_ctr != base->to->size)
        printf("Size mismatch %d %d\n", total_ctr, base->to->size);
      generated_partvec = 1;
    }
  }
*/
  //-----------------------------------------------------------------------------------------
  // Build adjacency list
  //-----------------------------------------------------------------------------------------
  //base->to->size does not include exec halo
  std::vector<int> row_offsets(base->to->core_size + 1);
  std::vector<int> col_indices;
  if (base->to == base->from) {
    col_indices.resize(base->dim * base->from->size);
    // if map is self-referencing, mapping is the same, except we have to remove references 
    // the halo
    row_offsets[0] = 0;
    for (int i = 0; i < base->from->size; i++) {
      int rowlen = 0;
      for (int j = 0; j < base->dim; j++)
        if (base->map[i*base->dim+j] < base->to->core_size)
          col_indices[row_offsets[i]+rowlen++] = base->map[i*base->dim+j];
      row_offsets[i+1] = row_offsets[i] + rowlen;
    }
    //reduce size to actual size
    col_indices.resize(row_offsets[base->to->core_size]);
  } else {
    // otherwise, construct self-referencing map
    col_indices.resize(base->from->size * (base->dim - 1) *
                       (base->dim)); // Worst case memory requirement

    // construct map pointing back, dropping indices referring to the halo
    std::vector<map2> loopback(base->from->size * base->dim);
    int sizectr = 0;
    for (int i = 0; i < base->from->size ; i++) {
      for (int j = 0; j < base->dim; j++) {
        if (base->map[i*base->dim+j] < base->to->core_size) {
          loopback[sizectr].a = base->map[i*base->dim+j];
          loopback[sizectr].b = i;
          sizectr++;
        }
      }
    }

    loopback.resize(sizectr);
    qsort(&loopback[0], loopback.size(), sizeof(map2), compare);
    
    row_offsets[0] = 0;
    row_offsets[1] = 0;
    row_offsets[base->to->core_size] = 0;
    for (int i = 0; i < base->dim; i++) {
      //Drop self-references
      if (base->map[base->dim * loopback[0].b + i] != 0 &&
          base->map[base->dim * loopback[0].b + i] < base->to->core_size)
        col_indices[row_offsets[1]++] =
            base->map[base->dim * loopback[0].b + i];
    }
    int nodectr = 0;
    for (int i = 1; i < loopback.size(); i++) {
      if (loopback[i].a != loopback[i - 1].a) {
        nodectr++;
        row_offsets[nodectr + 1] = row_offsets[nodectr];
      }

      for (int d1 = 0; d1 < base->dim; d1++) {
        int id = base->map[base->dim * loopback[i].b + d1];
        int add = (id != nodectr && id < base->to->core_size);
        for (int d2 = row_offsets[nodectr];
             (d2 < row_offsets[nodectr + 1]) && add; d2++) {
          if (col_indices[d2] == id)
            add = 0;
        }
        if (add)
          col_indices[row_offsets[nodectr + 1]++] = id;
      }
    }
    if (row_offsets[base->to->core_size] == 0) {
      printf(
          "Map %s is not an onto map from %s to %s, or bad partitioning, aborting renumbering...\n",
          base->name, base->from->name, base->to->name);
      return;
    }
    col_indices.resize(row_offsets[base->to->core_size]);
    if (OP_diags>2) op_printf("Loopback map %s->%s constructed: %d, from set %s (%d)\n",
           base->to->name, base->to->name, (int)col_indices.size(),
           base->from->name, base->from->size);
  }
  //sanity check
  for (int row = 0; row < row_offsets.size()-1; row++) {
    if (row_offsets[row] == row_offsets[row+1]) printf("Zero length row\n");
    for (int col = row_offsets[row]; col < row_offsets[row+1]; col++) {
      if (col_indices[col]<0 || col_indices[col]>=row_offsets.size()-1)
        printf("Error col idx %d, but num rows is %lu\n", col_indices[col],
               row_offsets.size() - 1);
      else {
        //Symmetry
        int found = 0;
        for (int c2 = row_offsets[col_indices[col]]; c2 < row_offsets[col_indices[col]+1]; c2++) {
          if (col_indices[c2] == row) found = 1;
        }
        if (!found) printf("Error, symmetry broken at row %d col %d\n",row,col_indices[col]);
      }
    }
  }
  //Statistics
  int max_dist = 0;
  long avg_dist = 0;
  for (int i = 0; i < base->from->size; i++) {
    int dist = 0;
    for (int d1 = 0; d1 < base->dim; d1++)
      for (int d2 = 0; d2 < base->dim; d2++)
        dist = std::max(dist,std::abs(base->map[i*base->dim+d1]-base->map[i*base->dim+d2]));
    max_dist = std::max(max_dist,dist);
    avg_dist += dist;
  }
  avg_dist /= base->from->size;
/*
  if (generated_partvec == 0) {
#ifdef PARMETIS_VER_4
    int possible[] = {2,  4,   6,   8,   12,  16,   22,   24,  32,
                      64, 128, 192, 256, 512, 1024, 2048, 4096};
    for (int i = 0; i < 17; i++) {
      int *partvec = (int *)malloc(base->to->size * sizeof(int));
      idxtype nconstr = 1;
      idxtype edgecut;
      idxtype nparts = possible[i];
      idxtype options[METIS_NOPTIONS];
      METIS_SetDefaultOptions(options);
      options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
      options[METIS_OPTION_NCUTS] = 3;
      options[METIS_OPTION_NSEPS] = 3;
      options[METIS_OPTION_NUMBERING] = 0;
      options[METIS_OPTION_MINCONN] = 1;

      METIS_PartGraphKway((idxtype *)&base->to->size, &nconstr,
                          (idxtype *)&row_offsets[0],
                          (idxtype *)&col_indices[0], NULL, NULL, NULL, &nparts,
                          NULL, NULL, options, &edgecut, (idxtype *)partvec);
      printf("Metis partitioning precomputed for %d partitions. Edgecut: %d\n",
             nparts, edgecut);
      char buffer[50];
      sprintf(buffer, "partvec%04d", possible[i]);
      op_decl_dat(base->to, 1, "int", partvec, buffer);
    }
#endif
  }
*/
  //
  // Using SCOTCH for reordering
  //
  SCOTCH_Num baseval = 0; // start numbering from 0
  SCOTCH_Num vertnbr =
      base->to->core_size; // number of vertices in graph = number of cells in mesh
  SCOTCH_Num edgenbr = row_offsets[base->to->core_size];

  SCOTCH_Graph *graphptr = SCOTCH_graphAlloc();
  SCOTCH_graphInit(graphptr);

  SCOTCH_Num *verttab = &row_offsets[0];

  SCOTCH_Num *vendtab = &verttab[1]; // = NULL; // Used to calculate vertex
                                     // degree = verttab[i+1] - verttab[i]
  SCOTCH_Num *velotab = NULL;        // Vertex load = vertex weight
  SCOTCH_Num *vlbltab = NULL;
  SCOTCH_Num *edgetab = &col_indices[0];

  SCOTCH_Num *edlotab = NULL; // Edge load = edge weight
  SCOTCH_Num *permutation =
      (SCOTCH_Num *)malloc(base->to->core_size * sizeof(SCOTCH_Num));
  SCOTCH_Num *ipermutation =
      (SCOTCH_Num *)malloc(base->to->core_size * sizeof(SCOTCH_Num));
  SCOTCH_Num *cblkptr =
      (SCOTCH_Num *)malloc(base->to->core_size * sizeof(SCOTCH_Num));
  SCOTCH_Num *rangtab =
      NULL; //(SCOTCH_Num*) malloc(1 + ncell*sizeof(SCOTCH_Num));
  SCOTCH_Num *treetab = NULL; //(SCOTCH_Num*) malloc(ncell*sizeof(SCOTCH_Num));

  int mesg = 0;
  mesg = SCOTCH_graphBuild(graphptr, baseval, vertnbr, verttab, vendtab,
                           velotab, vlbltab, edgenbr, edgetab, edlotab);
  if (mesg != 0) {
    op_printf("Error during SCOTCH_graphBuild() \n");
    exit(-1);
  }

  SCOTCH_Strat *straptr = SCOTCH_stratAlloc();
  SCOTCH_stratInit(straptr);

  const char *strategyString = "g";
  //    char * strategyString = "(g{pass=100})";
  mesg = SCOTCH_stratGraphOrder(straptr, strategyString);
  if (mesg != 0) {
    op_printf("Error during setting strategy string. \n");
    exit(-1);
  }

  mesg = SCOTCH_graphOrder(graphptr, straptr, permutation, ipermutation,
                           cblkptr, rangtab, treetab);
  if (mesg != 0) {
    op_printf("Error during SCOTCH_graphOrder() \n");
    exit(-1);
  }
  SCOTCH_graphExit(graphptr);
  SCOTCH_stratExit(straptr);

  std::vector<std::vector<int> > set_permutations(OP_set_index);
  std::vector<std::vector<int> > set_ipermutations(OP_set_index);
  set_permutations[base->to->index].resize(base->to->size + base->to->exec_size + base->to->nonexec_size);
  std::copy(permutation, permutation + base->to->size,
            set_permutations[base->to->index].begin());
  for (int i = base->to->core_size; i < base->to->size + base->to->exec_size + base->to->nonexec_size; i++)
    set_permutations[base->to->index][i] = i;
  check_permutation(&set_permutations[base->to->index][0],base->to->size + base->to->exec_size + base->to->nonexec_size);
  set_ipermutations[base->to->index].resize(base->to->size + base->to->exec_size + base->to->nonexec_size);
  std::copy(ipermutation, ipermutation + base->to->size,
            set_ipermutations[base->to->index].begin());
  for (int i = base->to->core_size; i < base->to->size + base->to->exec_size + base->to->nonexec_size; i++)
    set_permutations[base->to->index][i] = i;
  propagate_reordering(base->to, base->to, set_permutations, set_ipermutations);
  for (int i = 0; i < OP_set_index; i++) {
    reorder_set(OP_set_list[i], set_permutations, set_ipermutations);
  }

  op_move_to_device();

  op_printf("Before renumbering: maximum bandwidth = %d average bandwidth = %d\n",max_dist,avg_dist);
  //Statistics
  max_dist = 0;
  avg_dist = 0;
  for (int i = 0; i < base->from->size; i++) {
    int dist = 0;
    for (int d1 = 0; d1 < base->dim; d1++)
      for (int d2 = 0; d2 < base->dim; d2++)
        dist = std::max(dist,std::abs(base->map[i*base->dim+d1]-base->map[i*base->dim+d2]));
    max_dist = std::max(max_dist,dist);
    avg_dist += dist;
  }
  avg_dist /= base->from->size;
  op_printf("After renumbering: maximum bandwidth = %d average bandwidth = %d\n",max_dist,avg_dist);

#endif
}
