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
 * op_mpi_core.c
 *
 * Implements the OP2 Distributed memory (MPI) halo creation, halo exchange and
 * support utility routines/functions
 *
 * written by: Gihan R. Mudalige, (Started 01-03-2011)
 */

// mpi header
#include <mpi.h>

//#include <op_lib_core.h>
#include <cassert>
#include <memory>
#include <span>

#include <op_mpi_comm.h>
#include <op_lib_c.h>
#include <op_lib_mpi.h>
#include <op_util.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <numeric>

#include <op_mpi_core.h>

//
// MPI Halo related global variables
//

std::vector<SetHalo> OP_set_halos;

//
// Partial halo exchange lists
//

int *OP_map_partial_exchange; // flag for each map
std::vector<MapHalo> OP_map_halos;
//
// global array to hold dirty_bits for op_dats
//

/*table holding MPI performance of each loop
  (accessed via a hash of loop name) */
std::unordered_map<std::string, op_mpi_kernel *> op_mpi_kernel_map;

//
// global variables to hold partition information on an MPI rank
//

int OP_part_index = 0;
part *OP_part_list;

//
// Save original partition ranges
//

idx_g_t **orig_part_range = NULL;

// Timing
double t1, t2, c1, c2;

#ifdef __cplusplus
extern "C" {
#endif


/*
 * Wrappers of core lib
 */

op_set op_decl_set(idx_l_t size, char const *name) {
  return op_decl_set_core(size, name);
}

op_map op_decl_map(op_set from, op_set to, int dim, int *imap,
                   char const *name) {

  idx_g_t *imap_g = (idx_g_t *)malloc((idx_g_t)from->size * dim * sizeof(idx_g_t));
  for (idx_g_t i = 0; i < (idx_g_t)from->size * dim; i++) {
    imap_g[i] = (idx_g_t)imap[i];
  }

  op_map map = op_decl_map_long(from, to, dim, imap_g, name);
  free(imap_g);
  return map;
}

op_map op_decl_map_long(op_set from, op_set to, int dim, idx_g_t *imap_g,
                   char const *name) {

  op_map map = op_decl_map_core(from, to, dim, NULL, name);

  idx_g_t *imap_g2 = (idx_g_t *)malloc((idx_g_t)from->size * dim * sizeof(idx_g_t));
  for (idx_g_t i = 0; i < (idx_g_t)from->size * dim; i++) {
    imap_g2[i] = (idx_g_t)imap_g[i];
  }

  if (OP_maps_base_index == 1) {
    // convert imap to 0 based indexing -- i.e. reduce each imap value by 1
    for (idx_g_t i = 0; i < (idx_g_t)from->size * dim; i++)
      // imap[i]--;
      imap_g2[i]--; // modify op2's copy
  }
  //Convert from long global indices to shorter local indices. Set size
  //per process has to be less than INT_MAX
  map->map_gbl = imap_g2;
  map->user_managed = 0;

  op_register_map_ptr((int *)imap_g, map);
  return map;
}

op_arg op_arg_dat(op_dat dat, int idx, op_map map, int dim, char const *type,
                  op_access acc) {
  return op_arg_dat_core(dat, idx, map, dim, type, acc);
}

op_arg op_opt_arg_dat(int opt, op_dat dat, int idx, op_map map, int dim,
                      char const *type, op_access acc) {
  return op_opt_arg_dat_core(opt, dat, idx, map, dim, type, acc);
}

op_arg op_arg_gbl_char(char *data, int dim, const char *type, int size,
                       op_access acc) {
  return op_arg_gbl_core(1, data, dim, type, size, acc);
}

op_arg op_opt_arg_gbl_char(int opt, char *data, int dim, const char *type,
                           int size, op_access acc) {
  return op_arg_gbl_core(opt, data, dim, type, size, acc);
}

/*******************************************************************************
 * Routine to declare partition information for a given set
 *******************************************************************************/

void decl_partition(op_set set, idx_g_t *g_index, int *partition) {
  part p = (part)xmalloc(sizeof(part_core));
  p->set = set;
  p->g_index = g_index;
  p->elem_part = partition;
  p->is_partitioned = 0;
  OP_part_list[set->index] = p;
  OP_part_index++;
}

/*******************************************************************************
 * Routine to get partition range on all mpi ranks for all sets
 *******************************************************************************/

void get_part_range(idx_g_t **part_range, int my_rank, int comm_size,
                    MPI_Comm Comm) {
  (void)my_rank;
  for (int s = 0; s < OP_set_index; s++) {
    op_set set = OP_set_list[s];

    idx_g_t *sizes = (idx_g_t *)xmalloc(sizeof(idx_g_t) * comm_size);
    idx_g_t set_size = set->size;
    MPI_Allgather(&set_size, 1, get_mpi_type(&set_size),  sizes, 1, get_mpi_type(sizes), Comm);

    part_range[set->index] = (idx_g_t *)xmalloc(2 * comm_size * sizeof(idx_g_t));

    idx_g_t disp = 0;
    for (int i = 0; i < comm_size; i++) {
      part_range[set->index][2 * i] = disp;
      disp = disp + sizes[i] - 1;
      part_range[set->index][2 * i + 1] = disp;
      disp++;
#ifdef DEBUG
      if (my_rank == MPI_ROOT && OP_diags > 5)
        printf("range of %10s in rank %d: %lld-%lld\n", set->name, i,
               part_range[set->index][2 * i],
               part_range[set->index][2 * i + 1]);
#endif
    }
    op_free(sizes);
  }
}

/*******************************************************************************
 * Routine to get partition (i.e. mpi rank) where global_index is located and
 * its local index
 *******************************************************************************/

int get_partition(idx_g_t global_index, idx_g_t *part_range, int *local_index,
                  int comm_size, op_set set) {
  int low = 0;
  int high = comm_size - 1;
  
  while (low <= high) {
    int mid = low + (high - low) / 2;
    
    // Check if global_index is within the range of this partition
    if (global_index >= part_range[2 * mid] && global_index <= part_range[2 * mid + 1]) {
      *local_index = global_index - part_range[2 * mid];
      return mid;
    }
    
    // If global_index is smaller than the start of this partition's range
    if (global_index < part_range[2 * mid]) {
      high = mid - 1;
    } 
    // If global_index is larger than the end of this partition's range
    else {
      low = mid + 1;
    }
  }
  
  printf("Error: orphan global index %lld in set %s\n", global_index, set->name);
  MPI_Abort(OP_MPI_WORLD, 2);
  return -1;
}

/*******************************************************************************
 * Routine to convert a local index in to a global index
 *******************************************************************************/

idx_g_t get_global_index(idx_l_t local_index, int partition, idx_g_t *part_range,
                     int comm_size) {
  (void)comm_size;
  idx_g_t g_index = part_range[2 * partition] + local_index;
#ifdef DEBUG
  if (g_index > part_range[2 * (comm_size - 1) + 1] && OP_diags > 2)
    printf("Global index larger than set size\n");
#endif
  return g_index;
}

/*******************************************************************************
 * Halo list constructors
 *
 * C++ linkage, unlike the rest of this file: they take and return C++ types, and
 * the header declares them outside its extern "C" block.
 *******************************************************************************/

extern "C++" {

HaloList halo_list_from_groups(op_set set, std::vector<int> ranks,
                               std::vector<idx_l_t> sizes,
                               std::unique_ptr<idx_l_t[]> list) {
  assert(ranks.size() == sizes.size());
  HaloList h;
  h.set = set;
  h.disps.resize(ranks.size());
  idx_l_t total = 0;
  for (std::size_t i = 0; i < ranks.size(); i++) {
    assert(sizes[i] > 0 && (i == 0 || ranks[i - 1] < ranks[i]));
    h.disps[i] = total;
    total += sizes[i];
  }
  assert(total == 0 || list != nullptr);
  h.ranks = std::move(ranks);
  h.sizes = std::move(sizes);
  h.list = std::move(list);
  return h;
}

HaloList halo_list_from_pairs(op_set set, const int *pairs, int n_ints) {
  /* Each pair packed as rank:index, so one sort groups by rank and orders every
     rank's indices, and unique drops the repeats. No comm_size-long array: the
     cost is the pairs', whatever the number of ranks. */
  const int n = n_ints / 2;
  std::vector<std::uint64_t> keys(n);
  for (int i = 0; i < n; i++) {
    assert(pairs[2 * i] >= 0 && pairs[2 * i + 1] >= 0);
    keys[i] = (std::uint64_t)pairs[2 * i] << 32 | (std::uint32_t)pairs[2 * i + 1];
  }
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

  std::vector<int> ranks;
  std::vector<idx_l_t> sizes;
  std::unique_ptr<idx_l_t[]> list;
  if (!keys.empty())
    list = std::make_unique_for_overwrite<idx_l_t[]>(keys.size());
  for (std::size_t k = 0; k < keys.size(); k++) {
    const int rank = (int)(keys[k] >> 32);
    if (ranks.empty() || ranks.back() != rank) {
      ranks.push_back(rank);
      sizes.push_back(0);
    }
    sizes.back()++;
    list[k] = (idx_l_t)(std::uint32_t)keys[k];
  }
  return halo_list_from_groups(set, std::move(ranks), std::move(sizes),
                               std::move(list));
}

/* A halo list from what an exchange delivered: one entry per sending rank,
   ranks ascending, each holding what that rank sent. Takes the exchange's
   buffers over rather than copying them. */
static HaloList halo_list_from_received(op_set set, op::mpi::Received<int> &&got) {
  return halo_list_from_groups(set, std::move(got.ranks), std::move(got.counts),
                               std::move(got.data));
}

HaloList halo_list_transpose(op_set set, const HaloList &list, MPI_Comm comm) {
  /* One message per neighbour, each viewing its block of the list in place. */
  std::vector<op::mpi::msg::BlockView<int>> messages;
  messages.reserve(list.ranks_size());
  for (int i = 0; i < list.ranks_size(); i++)
    messages.emplace_back(list.ranks[i], list.list.get() + list.disps[i],
                          (std::size_t)list.sizes[i]);

  return halo_list_from_received(set, op::mpi::sparse::exchange(comm, messages));
}

}  // extern "C++"



/*******************************************************************************
 * Check whether a map reaches every element of its to-set: each rank tells the
 * owner of every to-set element its entries reach, and the owners count what
 * nobody reached. Expects map entries and set elements in the same global
 * numbering, as before partitioning. Collective over OP_MPI_WORLD.
 *******************************************************************************/

int is_onto_map(op_map map) {
  int my_rank, comm_size;
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_WORLD, &comm_size);

  idx_g_t **part_range = (idx_g_t **)xmalloc(OP_set_index * sizeof(idx_g_t *));
  get_part_range(part_range, my_rank, comm_size, OP_MPI_WORLD);
  idx_g_t *range = part_range[map->to->index];

  std::vector<idx_g_t> reached(map->map_gbl, map->map_gbl + static_cast<std::size_t>(map->from->size) * map->dim);
  std::sort(reached.begin(), reached.end());
  reached.erase(std::unique(reached.begin(), reached.end()), reached.end());
  auto told = op::mpi::sparse::exchange_by(OP_MPI_WORLD, reached, [&](idx_g_t g) {
    int local;
    return get_partition(g, range, &local, comm_size, map->to);
  });

  std::vector<char> hit(map->to->size, 0);
  for (idx_g_t g : told)
    hit[g - range[2 * my_rank]] = 1;
  long long missed = std::count(hit.begin(), hit.end(), 0), missed_anywhere = 0;
  MPI_Allreduce(&missed, &missed_anywhere, 1, MPI_LONG_LONG, MPI_SUM, OP_MPI_WORLD);

  for (int i = 0; i < OP_set_index; i++)
    op_free(part_range[i]);
  op_free(part_range);
  return missed_anywhere == 0;
}

/*******************************************************************************
 * Main MPI halo creation routine
 *******************************************************************************/

void op_halo_create() {
  // declare timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  double time;
  double max_time;
  op_timers(&cpu_t1, &wall_t1); // timer start for list create

  // create new communicator for OP mpi operation
  int my_rank, comm_size;
  // MPI_Comm_dup(OP_MPI_WORLD, &OP_MPI_WORLD);
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_WORLD, &comm_size);

  /* Compute global partition range information for each set*/
  idx_g_t **part_range = (idx_g_t **)xmalloc(OP_set_index * sizeof(idx_g_t *));
  get_part_range(part_range, my_rank, comm_size, OP_MPI_WORLD);

  // save this partition range information if it is not already saved during
  // a call to some partitioning routine
  if (orig_part_range == NULL) {
    orig_part_range = (idx_g_t **)xmalloc(OP_set_index * sizeof(idx_g_t *));
    for (int s = 0; s < OP_set_index; s++) {
      op_set set = OP_set_list[s];
      orig_part_range[set->index] = (idx_g_t *)xmalloc(2 * comm_size * sizeof(idx_g_t));
      for (int j = 0; j < comm_size; j++) {
        orig_part_range[set->index][2 * j] = part_range[set->index][2 * j];
        orig_part_range[set->index][2 * j + 1] =
            part_range[set->index][2 * j + 1];
      }
    }
  }

  OP_set_halos = std::vector<SetHalo>(OP_set_index);

  /*----- STEP 1 - Construct export lists for execute set elements and related
    mapping table entries -----*/

  // declare temporaty scratch variables to hold set export lists and mapping
  // table export lists
  idx_g_t s_i;
  int *set_list;

  idx_g_t cap_s = 1000; // keep track of the temp array capacities

  // Find all elements of other sets that are pointed to from this set
  // and are owned by other partitions, and construct export lists
  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];

    // create a temporaty scratch space to hold export list for this set
    s_i = 0;
    cap_s = 1000;
    set_list = (int *)xmalloc(cap_s * sizeof(int));

    for (int e = 0; e < set->size; e++) {      // for each elment of this set
      for (int m = 0; m < OP_map_index; m++) { // for each maping table
        op_map map = OP_map_list[m];

        if (compare_sets(map->from, set) == 1) { // need to select mappings
                                                 // FROM this set
          int part, local_index;
          for (int j = 0; j < map->dim; j++) { // for each element
                                               // pointed at by this entry
            part = get_partition(map->map_gbl[e * map->dim + j],
                                 part_range[map->to->index], &local_index,
                                 comm_size, map->to);
            if (s_i >= cap_s) {
              cap_s = cap_s * 2;
              set_list = (int *)xrealloc(set_list, cap_s * sizeof(int));
            }

            if (part != my_rank) {
              set_list[s_i++] = part; // add to set export list
              set_list[s_i++] = e;
            }
          }
        }
      }
    }

    // create set export list
    // printf("creating set export list for set %10s of size %d\n",
    // set->name,s_i);
    OP_set_halos[set->index].export_exec = halo_list_from_pairs(set, set_list, s_i);
    op_free(set_list); // free temp list
  }

  /*---- STEP 2 - construct import lists for mappings and execute sets------*/

  MPI_Request *request_send;

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];

    OP_set_halos[set->index].import_exec =
        halo_list_transpose(set, OP_set_halos[set->index].export_exec, OP_MPI_WORLD);
  }

  /*--STEP 3 -Exchange mapping table entries using the import/export lists--*/

  for (int m = 0; m < OP_map_index; m++) { // for each maping table
    op_map map = OP_map_list[m];
    const HaloList &i_list = OP_set_halos[map->from->index].import_exec;
    const HaloList &e_list = OP_set_halos[map->from->index].export_exec;

    request_send = (MPI_Request *)xmalloc(e_list.ranks_size() * sizeof(MPI_Request));

    // prepare bits of the mapping tables to be exported
    idx_g_t **sbuf = (idx_g_t **)xmalloc(e_list.ranks_size() * sizeof(idx_g_t *));

    for (int i = 0; i < e_list.ranks_size(); i++) {
      sbuf[i] = (idx_g_t *)xmalloc((size_t)e_list.sizes[i] * map->dim * sizeof(idx_g_t));
      for (int j = 0; j < e_list.sizes[i]; j++) {
        for (int p = 0; p < map->dim; p++) {
          sbuf[i][j * map->dim + p] =
              map->map_gbl[map->dim * (e_list.list[e_list.disps[i] + j]) + p];
        }
      }
      // printf("\n export from %d to %d map %10s, number of elements of size %d
      // | sending:\n ",
      //    my_rank,e_list.ranks[i],map.name,e_list.sizes[i]);
      MPI_Isend(sbuf[i], map->dim * e_list.sizes[i], get_mpi_type(sbuf[i]), e_list.ranks[i],
                m, OP_MPI_WORLD, &request_send[i]);
    }

    // prepare space for the incomming mapping tables - realloc each
    // mapping tables in each mpi process
    OP_map_list[map->index]->map_gbl = (idx_g_t *)xrealloc(
        OP_map_list[map->index]->map_gbl,
        (map->dim * (size_t)(map->from->size + i_list.size())) * sizeof(idx_g_t));

    int init = map->dim * (map->from->size);
    for (int i = 0; i < i_list.ranks_size(); i++) {
      // printf("\n imported on to %d map %10s, number of elements of size %d |
      // recieving: ",
      // my_rank, map->name, i_list.size());
      MPI_Recv(
          &(OP_map_list[map->index]->map_gbl[init + i_list.disps[i] * map->dim]),
          map->dim * i_list.sizes[i],
          get_mpi_type(OP_map_list[map->index]->map_gbl), i_list.ranks[i], m,
          OP_MPI_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Waitall(e_list.ranks_size(), request_send, MPI_STATUSES_IGNORE);
    op_free(request_send);

    for (int i = 0; i < e_list.ranks_size(); i++)
      op_free(sbuf[i]);
    op_free(sbuf);
  }

  /*-- STEP 4 - Create import lists for non-execute set elements using mapping
  table entries including the additional mapping table entries --*/

  // declare temporaty scratch variables to hold non-exec set export lists
  s_i = 0;
  set_list = NULL;
  cap_s = 1000; // keep track of the temp array capacity

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    const HaloList &exec_set_list = OP_set_halos[set->index].import_exec;

    // create a temporaty scratch space to hold nonexec export list for this set
    s_i = 0;
    set_list = (int *)xmalloc(cap_s * sizeof(int));

    for (int m = 0; m < OP_map_index; m++) { // for each maping table
      op_map map = OP_map_list[m];
      const HaloList &exec_map_list = OP_set_halos[map->from->index].import_exec;

      if (compare_sets(map->to, set) == 1) { // need to select
                                             // mappings TO this set

        // for each entry in this mapping table: original+execlist
        int len = map->from->size + exec_map_list.size();
        for (int e = 0; e < len; e++) {
          int part;
          int local_index = 0;
          for (int j = 0; j < map->dim; j++) { // for each element pointed
                                               // at by this entry
            part = get_partition(map->map_gbl[e * map->dim + j],
                                 part_range[map->to->index], &local_index,
                                 comm_size, map->to);

            if (s_i >= cap_s) {
              cap_s = cap_s * 2;
              set_list = (int *)xrealloc(set_list, cap_s * sizeof(int));
            }

            if (part != my_rank) {
              int found = -1;
              // check in exec list
              int rank = binary_search(exec_set_list.ranks.data(), part, 0,
                                       exec_set_list.ranks_size() - 1);

              if (rank >= 0) {
                found = binary_search(exec_set_list.list.get(), local_index,
                                      exec_set_list.disps[rank],
                                      exec_set_list.disps[rank] +
                                          exec_set_list.sizes[rank] - 1);
              }

              if (found < 0) {
                // not in this partition and not found in
                // exec list
                // add to non-execute set_list
                set_list[s_i++] = part;
                set_list[s_i++] = local_index;
              }
            }
          }
        }
      }
    }

    // Create the non-exec set import list. It is built from (rank, index) pairs
    // like an export list: these are the elements this rank needs, by owner.
    OP_set_halos[set->index].import_nonexec = halo_list_from_pairs(set, set_list, s_i);
    op_free(set_list); // free temp list
  }

  /*----------- STEP 5 - construct non-execute set export lists -------------*/

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];

    /* The nonexec import list was built unilaterally from the received map
       entries, so its owners do not know about it; sending it back is what
       gives them their export list. */
    OP_set_halos[set->index].export_nonexec =
        halo_list_transpose(set, OP_set_halos[set->index].import_nonexec, OP_MPI_WORLD);
  }

  /*-STEP 6 - Exchange execute set elements/data using the import/export
   * lists--*/

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    const HaloList &i_list = OP_set_halos[set->index].import_exec;
    const HaloList &e_list = OP_set_halos[set->index].export_exec;

    // for each data array
    op_dat_entry *item;
    int d = -1; // d is just simply the tag for mpi comms
    TAILQ_FOREACH(item, &OP_dat_list, entries) {
      d++; // increase tag to do mpi comm for the next op_dat
      op_dat dat = item->dat;

      if (compare_sets(set, dat->set) == 1) { // if this data array
                                              // is defined on this set

        // printf("on rank %d, The data array is %10s\n",my_rank,dat->name);
        request_send = (MPI_Request *)xmalloc(e_list.ranks_size() * sizeof(MPI_Request));

        // prepare execute set element data to be exported
        char **sbuf = (char **)xmalloc(e_list.ranks_size() * sizeof(char *));

        for (int i = 0; i < e_list.ranks_size(); i++) {
          sbuf[i] = (char *)xmalloc((size_t)e_list.sizes[i] * (size_t)dat->size);
          for (int j = 0; j < e_list.sizes[i]; j++) {
            int set_elem_index = e_list.list[e_list.disps[i] + j];
            memcpy(&sbuf[i][j * (size_t)dat->size],
                   (void *)&dat->data[(size_t)dat->size * (set_elem_index)], dat->size);
          }
          // printf("export from %d to %d data %10s, number of elements of size
          // %d | sending:\n ",
          //    my_rank,e_list.ranks[i],dat->name,e_list.sizes[i]);
          MPI_Isend(sbuf[i], (size_t)dat->size * e_list.sizes[i], MPI_CHAR,
                    e_list.ranks[i], d, OP_MPI_WORLD, &request_send[i]);
        }

        // prepare space for the incomming data - realloc each
        // data array in each mpi process
        dat->data = (char *)xrealloc(
            dat->data, (size_t)(set->size + i_list.size()) * (size_t)dat->size);

        size_t init = set->size * (size_t)dat->size;
        for (int i = 0; i < i_list.ranks_size(); i++) {
          MPI_Recv(&(dat->data[init + i_list.disps[i] * (size_t)dat->size]),
                   (size_t)dat->size * i_list.sizes[i], MPI_CHAR,
                   i_list.ranks[i], d, OP_MPI_WORLD, MPI_STATUS_IGNORE);
        }

        MPI_Waitall(e_list.ranks_size(), request_send, MPI_STATUSES_IGNORE);
        op_free(request_send);

        for (int i = 0; i < e_list.ranks_size(); i++)
          op_free(sbuf[i]);
        op_free(sbuf);
        // printf("imported on to %d data %10s, number of elements of size %d |
        // recieving:\n ",
        //    my_rank, dat->name, i_list.size());
      }
    }
  }

  /*-STEP 7 - Exchange non-execute set elements/data using the import/export
   * lists--*/

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    const HaloList &i_list = OP_set_halos[set->index].import_nonexec;
    const HaloList &e_list = OP_set_halos[set->index].export_nonexec;

    // for each data array
    op_dat_entry *item;
    int d = -1; // d is just simply the tag for mpi comms
    TAILQ_FOREACH(item, &OP_dat_list, entries) {
      d++; // increase tag to do mpi comm for the next op_dat
      op_dat dat = item->dat;

      if (compare_sets(set, dat->set) == 1) { // if this data array is
                                              // defined on this set

        // printf("on rank %d, The data array is %10s\n",my_rank,dat->name);
        request_send = (MPI_Request *)xmalloc(e_list.ranks_size() * sizeof(MPI_Request));

        // prepare non-execute set element data to be exported
        char **sbuf = (char **)xmalloc(e_list.ranks_size() * sizeof(char *));

        for (int i = 0; i < e_list.ranks_size(); i++) {
          sbuf[i] = (char *)xmalloc(e_list.sizes[i] * (size_t)dat->size);
          for (int j = 0; j < e_list.sizes[i]; j++) {
            int set_elem_index = e_list.list[e_list.disps[i] + j];
            memcpy(&sbuf[i][j * (size_t)dat->size],
                   (void *)&dat->data[(size_t)dat->size * (set_elem_index)], dat->size);
          }
          MPI_Isend(sbuf[i], (size_t)dat->size * e_list.sizes[i], MPI_CHAR,
                    e_list.ranks[i], d, OP_MPI_WORLD, &request_send[i]);
        }

        // prepare space for the incomming nonexec-data - realloc each
        // data array in each mpi process
        const HaloList &exec_i_list = OP_set_halos[set->index].import_exec;

        dat->data = (char *)xrealloc(
            dat->data,
            (size_t)(set->size + exec_i_list.size() + i_list.size()) * (size_t)dat->size);

        size_t init = (size_t)(set->size + exec_i_list.size()) * (size_t)dat->size;
        for (int i = 0; i < i_list.ranks_size(); i++) {
          MPI_Recv(&(dat->data[init + i_list.disps[i] * (size_t)dat->size]),
                   (size_t)dat->size * i_list.sizes[i], MPI_CHAR, i_list.ranks[i], d,
                   OP_MPI_WORLD, MPI_STATUS_IGNORE);
        }

        MPI_Waitall(e_list.ranks_size(), request_send, MPI_STATUSES_IGNORE);
        op_free(request_send);

        for (int i = 0; i < e_list.ranks_size(); i++)
          op_free(sbuf[i]);
        op_free(sbuf);
      }
    }
  }

  /*-STEP 8 ----------------- Renumber Mapping tables-----------------------*/

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];

    for (int m = 0; m < OP_map_index; m++) { // for each maping table
      op_map map = OP_map_list[m];

      if (compare_sets(map->to, set) == 1) { // need to select
                                             // mappings TO this set

        const HaloList &exec_set_list = OP_set_halos[set->index].import_exec;
        const HaloList &nonexec_set_list = OP_set_halos[set->index].import_nonexec;

        const HaloList &exec_map_list = OP_set_halos[map->from->index].import_exec;

        // for each entry in this mapping table: original+execlist
        int len = map->from->size + exec_map_list.size();
        map->map = (int *)xmalloc(len * map->dim * sizeof(int));
        for (int e = 0; e < len; e++) {
          for (int j = 0; j < map->dim; j++) { // for each element
                                               // pointed at by this entry
            int part;
            int local_index = 0;
            part = get_partition(map->map_gbl[e * map->dim + j],
                                 part_range[map->to->index], &local_index,
                                 comm_size, map->to);

            if (part == my_rank) {
              OP_map_list[map->index]->map[e * map->dim + j] = local_index;
            } else {
              int found = -1;
              // check in exec list
              int rank1 = binary_search(exec_set_list.ranks.data(), part, 0,
                                        exec_set_list.ranks_size() - 1);
              // check in nonexec list
              int rank2 = binary_search(nonexec_set_list.ranks.data(), part, 0,
                                        nonexec_set_list.ranks_size() - 1);

              if (rank1 >= 0) {
                found = binary_search(exec_set_list.list.get(), local_index,
                                      exec_set_list.disps[rank1],
                                      exec_set_list.disps[rank1] +
                                          exec_set_list.sizes[rank1] - 1);
                if (found >= 0) {
                  OP_map_list[map->index]->map[e * map->dim + j] =
                      found + map->to->size;
                }
              }

              if (rank2 >= 0 && found < 0) {
                found = binary_search(nonexec_set_list.list.get(), local_index,
                                      nonexec_set_list.disps[rank2],
                                      nonexec_set_list.disps[rank2] +
                                          nonexec_set_list.sizes[rank2] - 1);
                if (found >= 0) {
                  OP_map_list[map->index]->map[e * map->dim + j] =
                      found + set->size + exec_set_list.size();
                }
              }

              if (found < 0)
                printf("ERROR: Set %10s Element %d needed on rank %d \
                from partition %d\n",
                       set->name, local_index, my_rank, part);
            }
          }
        }
        free(map->map_gbl);
        map->map_gbl = NULL;
      }
    }
  }

  // set dirty bits of all data arrays to 0
  // for each data array
  op_dat_entry *item = NULL;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    dat->dirtybit = 0;
  }

  /*-STEP 10 -------------------- Separate core
   * elements------------------------*/

  int **core_elems = (int **)xmalloc(OP_set_index * sizeof(int *));
  int **exp_elems = (int **)xmalloc(OP_set_index * sizeof(int *));

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];

    HaloList &exec = OP_set_halos[set->index].export_exec;
    HaloList &nonexec = OP_set_halos[set->index].export_nonexec;

    if (exec.size() > 0) {
      exp_elems[set->index] = (int *)xmalloc(exec.size() * sizeof(int));
      memcpy(exp_elems[set->index], exec.list.get(), exec.size() * sizeof(int));
      op_sort(exp_elems[set->index], exec.size());

      int num_exp = removeDups(exp_elems[set->index], exec.size());
      core_elems[set->index] = (int *)xmalloc(set->size * sizeof(int));
      int count = 0;
      for (int e = 0; e < set->size; e++) { // for each elment of this set

        if ((binary_search(exp_elems[set->index], e, 0, num_exp - 1) < 0)) {
          core_elems[set->index][count++] = e;
        }
      }
      op_sort(core_elems[set->index], count);

      if (count + num_exp != set->size)
        printf("sizes not equal\n");
      set->core_size = count;

      // for each data array defined on this set seperate its elements
      op_dat_entry *item;
      TAILQ_FOREACH(item, &OP_dat_list, entries) {
        op_dat dat = item->dat;

        if (compare_sets(set, dat->set) == 1) // if this data array is
        // defined on this set
        {
          char *new_dat = (char *)xmalloc((size_t)set->size * (size_t)dat->size);
          for (int i = 0; i < count; i++) {
            memcpy(&new_dat[i * (size_t)dat->size],
                   &dat->data[core_elems[set->index][i] * (size_t)dat->size],
                   dat->size);
          }
          for (int i = 0; i < num_exp; i++) {
            memcpy(&new_dat[(count + i) * (size_t)dat->size],
                   &dat->data[exp_elems[set->index][i] * (size_t)dat->size], dat->size);
          }
          memcpy(&dat->data[0], &new_dat[0], set->size * (size_t)dat->size);
          op_free(new_dat);
        }
      }

      // for each mapping defined from this set seperate its elements
      for (int m = 0; m < OP_map_index; m++) { // for each set
        op_map map = OP_map_list[m];

        if (compare_sets(map->from, set) == 1) { // if this mapping is
                                                 // defined from this set
          int *new_map = (int *)xmalloc((size_t)set->size * map->dim * sizeof(int));
          for (int i = 0; i < count; i++) {
            memcpy(&new_map[i * (size_t)map->dim],
                   &map->map[core_elems[set->index][i] * (size_t)map->dim],
                   map->dim * sizeof(int));
          }
          for (int i = 0; i < num_exp; i++) {
            memcpy(&new_map[(count + i) * (size_t)map->dim],
                   &map->map[exp_elems[set->index][i] * (size_t)map->dim],
                   map->dim * sizeof(int));
          }
          memcpy(&map->map[0], &new_map[0], set->size * (size_t)map->dim * sizeof(int));
          op_free(new_map);
        }
      }

      for (int i = 0; i < exec.size(); i++) {
        int index =
            binary_search(exp_elems[set->index], exec.list[i], 0, num_exp - 1);
        if (index < 0)
          printf("Problem in seperating core elements - exec list\n");
        else
          exec.list[i] = count + index;
      }

      for (int i = 0; i < nonexec.size(); i++) {
        int index = binary_search(core_elems[set->index], nonexec.list[i], 0,
                                  count - 1);
        if (index < 0) {
          index = binary_search(exp_elems[set->index], nonexec.list[i], 0,
                                num_exp - 1);
          if (index < 0)
            printf("Problem in seperating core elements - nonexec list\n");
          else
            nonexec.list[i] = count + index;
        } else
          nonexec.list[i] = index;
      }
    } else {
      core_elems[set->index] = (int *)xmalloc(set->size * sizeof(int));
      exp_elems[set->index] = (int *)xmalloc(0 * sizeof(int));
      for (int e = 0; e < set->size; e++) { // for each elment of this set
        core_elems[set->index][e] = e;
      }
      set->core_size = set->size;
    }
  }

  // now need to renumber mapping tables as the elements are seperated
  for (int m = 0; m < OP_map_index; m++) { // for each set
    op_map map = OP_map_list[m];

    const HaloList &exec_map_list = OP_set_halos[map->from->index].import_exec;
    // for each entry in this mapping table: original+execlist
    int len = map->from->size + exec_map_list.size();
    for (int e = 0; e < len; e++) {
      for (int j = 0; j < map->dim; j++) { // for each element pointed
                                           // at by this entry
        if (map->map[e * map->dim + j] < map->to->size) {
          int index = binary_search(core_elems[map->to->index],
                                    map->map[e * map->dim + j], 0,
                                    map->to->core_size - 1);
          if (index < 0) {
            index = binary_search(exp_elems[map->to->index],
                                  map->map[e * map->dim + j], 0,
                                  (map->to->size) - (map->to->core_size) - 1);
            if (index < 0)
              printf("Problem in seperating core elements - \
              renumbering map\n");
            else
              OP_map_list[map->index]->map[e * (size_t)map->dim + j] =
                  map->to->core_size + index;
          } else
            OP_map_list[map->index]->map[e * (size_t)map->dim + j] = index;
        }
      }
    }
  }

  /* Step 10 moved owned elements and rewrote the export lists; the import lists
     other ranks hold still name the old positions. */
  op_halo_refresh_imports();

  /*-STEP 11 ----------- Save the original set element
   * indexes------------------*/

  // if OP_part_list is empty, (i.e. no previous partitioning done) then
  // create it and store the seperation of elements using core_elems
  // and exp_elems
  if (OP_part_index != OP_set_index) {
    // allocate memory for list
    OP_part_list = (part *)xmalloc(OP_set_index * sizeof(part));

    for (int s = 0; s < OP_set_index; s++) { // for each set
      op_set set = OP_set_list[s];
      // printf("set %s size = %d\n", set.name, set.size);
      idx_g_t *g_index = (idx_g_t *)xmalloc(sizeof(idx_g_t) * set->size);
      int *partition = (int *)xmalloc(sizeof(int) * set->size);
      for (int i = 0; i < set->size; i++) {
        g_index[i] =
            get_global_index(i, my_rank, part_range[set->index], comm_size);
        partition[i] = my_rank;
      }
      decl_partition(set, g_index, partition);

      // combine core_elems and exp_elems to one memory block
      idx_g_t *temp = (idx_g_t *)xmalloc(sizeof(idx_g_t) * set->size);
      // memcpy(&temp[0], core_elems[set->index], set->core_size * sizeof(idx_g_t));
      // memcpy(&temp[set->core_size], exp_elems[set->index],
      //  (set->size - set->core_size) * sizeof(idx_g_t));
      for (int i = 0; i < set->core_size; i++) {
        temp[i] = core_elems[set->index][i];
      }
      for (int i = 0; i < set->size - set->core_size; i++) {
        temp[set->core_size + i] = exp_elems[set->index][i];
      }

      // update OP_part_list[set->index]->g_index
      for (int i = 0; i < set->size; i++) {
        temp[i] = OP_part_list[set->index]->g_index[temp[i]];
      }
      op_free(OP_part_list[set->index]->g_index);
      OP_part_list[set->index]->g_index = temp;
    }
  } else { // OP_part_list exists (i.e. a partitioning has been done)
    // update the seperation of elements

    for (int s = 0; s < OP_set_index; s++) { // for each set
      op_set set = OP_set_list[s];

      // combine core_elems and exp_elems to one memory block
      idx_g_t *temp = (idx_g_t *)xmalloc(sizeof(idx_g_t) * set->size);

      if (set->core_size * sizeof(int) > 0) {
        //  memcpy(&temp[0], core_elems[set->index], set->core_size * sizeof(int));
        for (int i = 0; i < set->core_size; i++) {
          temp[i] = core_elems[set->index][i];
        }
      }

      if ((set->size - set->core_size) * sizeof(idx_g_t) > 0) {
        // memcpy(&temp[set->core_size], exp_elems[set->index],
        //        (set->size - set->core_size) * sizeof(idx_g_t));
        for (int i = 0; i < set->size - set->core_size; i++) {
          temp[set->core_size + i] = exp_elems[set->index][i];
        }
      }

      // update OP_part_list[set->index]->g_index
      for (int i = 0; i < set->size; i++) {
        temp[i] = OP_part_list[set->index]->g_index[temp[i]];
      }
      op_free(OP_part_list[set->index]->g_index);
      OP_part_list[set->index]->g_index = temp;
    }
  }

  /*for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];
    printf("Original Index for set %s\n", set->name);
    for(int i=0; i<set->size; i++ )
    printf(" %d",OP_part_list[set->index]->g_index[i]);
    }*/

  // set up exec and nonexec sizes
  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    set->exec_size = OP_set_halos[set->index].import_exec.size();
    set->nonexec_size = OP_set_halos[set->index].import_nonexec.size();
  }

  /*-STEP 12 ---------- Clean up and Compute rough halo size
   * numbers------------*/

  for (int i = 0; i < OP_set_index; i++) {
    op_free(part_range[i]);
    op_free(core_elems[i]);
    op_free(exp_elems[i]);
  }
  op_free(part_range);
  op_free(exp_elems);
  op_free(core_elems);

  op_timers(&cpu_t2, &wall_t2); // timer stop for list create
  // compute import/export lists creation time
  time = wall_t2 - wall_t1;
  MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);

  // compute avg/min/max set sizes and exec sizes accross the MPI universe
  for (int s = 0; s < OP_set_index; s++) {
    op_set set = OP_set_list[s];
    
    {
      idx_g_t avg_size = 0, min_size = 0, max_size = 0;
      idx_g_t size = set->size;
      // number of set elements first
      MPI_Reduce(&size, &avg_size, 1, get_mpi_type(&size), MPI_SUM, MPI_ROOT,
                OP_MPI_WORLD);
      MPI_Reduce(&size, &min_size, 1, get_mpi_type(&size), MPI_MIN, MPI_ROOT,
                OP_MPI_WORLD);
      MPI_Reduce(&size, &max_size, 1, get_mpi_type(&size), MPI_MAX, MPI_ROOT,
                OP_MPI_WORLD);

      if (my_rank == MPI_ROOT) {
        printf("Num of %8s (avg | min | max)\n", set->name);
        printf("total elems         %10d %10d %10d\n", (int)(avg_size / comm_size),
              (int)min_size, (int)max_size);
      }
    }

    {
      idx_g_t avg_size = 0, min_size = 0, max_size = 0;
      idx_g_t core_size = set->core_size;
      // number of OWNED elements second
      MPI_Reduce(&core_size, &avg_size, 1, get_mpi_type(&core_size), MPI_SUM, MPI_ROOT,
                OP_MPI_WORLD);
      MPI_Reduce(&core_size, &min_size, 1, get_mpi_type(&core_size), MPI_MIN, MPI_ROOT,
                OP_MPI_WORLD);
      MPI_Reduce(&core_size, &max_size, 1, get_mpi_type(&core_size), MPI_MAX, MPI_ROOT,
                OP_MPI_WORLD);
    
      if (my_rank == MPI_ROOT) {
        printf("core elems         %10d %10d %10d \n", (int)(avg_size / comm_size),
              (int)min_size, (int)max_size);
      }
    }

    {
      idx_g_t avg_size = 0, min_size = 0, max_size = 0;
      idx_g_t exec_size = OP_set_halos[set->index].import_exec.size();
      // number of exec halo elements third
      MPI_Reduce(&exec_size, &avg_size, 1, get_mpi_type(&exec_size), MPI_SUM, MPI_ROOT, OP_MPI_WORLD);
      MPI_Reduce(&exec_size, &min_size, 1, get_mpi_type(&exec_size), MPI_MIN, MPI_ROOT, OP_MPI_WORLD);
      MPI_Reduce(&exec_size, &max_size, 1, get_mpi_type(&exec_size), MPI_MAX, MPI_ROOT, OP_MPI_WORLD);
      if (my_rank == MPI_ROOT) {
        printf("exec halo elems     %10d %10d %10d \n", (int)(avg_size / comm_size),
              (int)min_size, (int)max_size);
      }
    }

    {
      idx_g_t avg_size = 0, min_size = 0, max_size = 0;
      idx_g_t nonexec_size = OP_set_halos[set->index].import_nonexec.size();
      // number of non-exec halo elements fourth
      MPI_Reduce(&nonexec_size, &avg_size, 1, get_mpi_type(&nonexec_size), MPI_SUM, MPI_ROOT, OP_MPI_WORLD);
      MPI_Reduce(&nonexec_size, &min_size, 1, get_mpi_type(&nonexec_size), MPI_MIN, MPI_ROOT, OP_MPI_WORLD);
      MPI_Reduce(&nonexec_size, &max_size, 1, get_mpi_type(&nonexec_size), MPI_MAX, MPI_ROOT, OP_MPI_WORLD);
      if (my_rank == MPI_ROOT) {
        printf("non-exec halo elems %10d %10d %10d \n", (int)(avg_size / comm_size),
              (int)min_size, (int)max_size);
      }
    }
    if (my_rank == MPI_ROOT) {
      printf("-----------------------------------------------------\n");
    }
  }

  if (my_rank == MPI_ROOT) {
    printf("\n\n");
  }

  // compute avg/min/max number of MPI neighbors per process accross the MPI
  // universe
  for (int s = 0; s < OP_set_index; s++) {
    op_set set = OP_set_list[s];
    {
      int neighbours = OP_set_halos[set->index].import_exec.ranks_size();
      int avg_size = 0, min_size = 0, max_size = 0;
      // number of exec halo neighbors first
      MPI_Reduce(&neighbours, &avg_size, 1,
               MPI_INT, MPI_SUM, MPI_ROOT, OP_MPI_WORLD);
      MPI_Reduce(&neighbours, &min_size, 1,
                MPI_INT, MPI_MIN, MPI_ROOT, OP_MPI_WORLD);
      MPI_Reduce(&neighbours, &max_size, 1,
                MPI_INT, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);
      if (my_rank == MPI_ROOT) {
        printf("MPI neighbors for exchanging %8s (avg | min | max)\n", set->name);
        printf("exec halo elems     %4d %4d %4d\n", avg_size / comm_size,
              (int)min_size, (int)max_size);
      }
    }

    {
      int neighbours = OP_set_halos[set->index].import_nonexec.ranks_size();
      int avg_size = 0, min_size = 0, max_size = 0;
      // number of non-exec halo neighbors second
      MPI_Reduce(&neighbours, &avg_size, 1,
               MPI_INT, MPI_SUM, MPI_ROOT, OP_MPI_WORLD);
      MPI_Reduce(&neighbours, &min_size, 1,
               MPI_INT, MPI_MIN, MPI_ROOT, OP_MPI_WORLD);
      MPI_Reduce(&neighbours, &max_size, 1,
               MPI_INT, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);
      if (my_rank == MPI_ROOT) {
        printf("MPI neighbors for exchanging %8s (avg | min | max)\n", set->name);
        printf("non-exec halo elems %4d %4d %4d\n", avg_size / comm_size,
              (int)min_size, (int)max_size);
      }
    }
    if (my_rank == MPI_ROOT) {
      printf("-----------------------------------------------------\n");
    }
  }

  // compute average worst case halo size in Bytes
  idx_g_t tot_halo_size = 0;
  for (int s = 0; s < OP_set_index; s++) {
    op_set set = OP_set_list[s];

    op_dat_entry *item;
    TAILQ_FOREACH(item, &OP_dat_list, entries) {
      op_dat dat = item->dat;

      if (compare_sets(dat->set, set) == 1) {
        const HaloList &exec_imp = OP_set_halos[set->index].import_exec;
        const HaloList &nonexec_imp = OP_set_halos[set->index].import_nonexec;
        tot_halo_size = tot_halo_size + exec_imp.size() * (size_t)dat->size +
                        nonexec_imp.size() * (size_t)dat->size;
      }
    }
  }
  idx_g_t avg_halo_size;
  MPI_Reduce(&tot_halo_size, &avg_halo_size, 1, get_mpi_type(&tot_halo_size), MPI_SUM, MPI_ROOT,
             OP_MPI_WORLD);

  // print performance results
  if (my_rank == MPI_ROOT) {
    printf("Max total halo creation time = %lf\n", max_time);
    printf("Average (worst case) Halo size = %d Bytes\n",
           (int)(avg_halo_size / comm_size));
  }
}

/*******************************************************************************
 * Bring the import lists up to date with their owners' numbering
 *
 * Every import list is the transpose of the export lists that feed it, block for
 * block and position for position, so transposing the current export lists
 * gives the same layout with current entries. Nothing in an exchange reads the
 * entries - received data lands at disps - but op_mpi_probe_halo_index hands
 * them to callers as the element's index on its owner.
 *******************************************************************************/

void op_halo_refresh_imports() {
  for (int s = 0; s < OP_set_index; s++) {
    op_set set = OP_set_list[s];
    SetHalo &halo = OP_set_halos[set->index];
    for (auto [exp, imp] : {std::pair{&halo.export_exec, &halo.import_exec},
                            std::pair{&halo.export_nonexec, &halo.import_nonexec}}) {
      HaloList current = halo_list_transpose(set, *exp, OP_MPI_WORLD);
      assert(current.ranks == imp->ranks && current.sizes == imp->sizes);
      *imp = std::move(current);
    }
  }
}

/*******************************************************************************
 * Create map-specific halo exchange tables
 *
 * A map is exchanged partially when, summed over all ranks, its entries into the
 * halo of the set it points to number fewer than 30% of that halo. For each such
 * map every rank builds two lists:
 *
 *   import  the halo elements the map reaches from [core_size, size + exec_size),
 *           as local indices, grouped by owning rank and in halo order
 *   export  for each rank importing from this one, the local indices of the
 *           owned elements it needs
 *
 * An importer names each element it needs by its position in the halo it
 * receives from that owner - the exec block, then the nonexec block - and the
 * owner translates positions through its own export lists, so the result does
 * not depend on the values the import lists hold.
 *******************************************************************************/

/* Where rank sits in a list's ranks, or -1. */
static int rank_slot(const HaloList &h, int rank) {
  auto it = std::lower_bound(h.ranks.begin(), h.ranks.end(), rank);
  return it != h.ranks.end() && *it == rank ? (int)(it - h.ranks.begin()) : -1;
}

void op_halo_permap_create() {
  /* Which maps are partially exchanged: halo references against halo size. */
  std::vector<idx_g_t> halo_sizes(OP_set_index), map_halo_sizes(OP_map_index);
  for (int s = 0; s < OP_set_index; s++)
    halo_sizes[s] = OP_set_list[s]->exec_size + OP_set_list[s]->nonexec_size;
  for (int m = 0; m < OP_map_index; m++) {
    op_map map = OP_map_list[m];
    for (int e = map->from->core_size; e < map->from->size + map->from->exec_size; e++)
      for (int j = 0; j < map->dim; j++)
        if (map->map[e * map->dim + j] >= map->to->size)
          map_halo_sizes[m]++;
  }
  MPI_Allreduce(MPI_IN_PLACE, halo_sizes.data(), OP_set_index, get_mpi_type<idx_g_t>(),
                MPI_SUM, OP_MPI_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, map_halo_sizes.data(), OP_map_index, get_mpi_type<idx_g_t>(),
                MPI_SUM, OP_MPI_WORLD);

  OP_map_partial_exchange = (int *)xmalloc(OP_map_index * sizeof(int));
  for (int m = 0; m < OP_map_index; m++) {
    op_map map = OP_map_list[m];
    OP_map_partial_exchange[m] = OP_partial_exchange &&
        (double)map_halo_sizes[m] < (double)halo_sizes[map->to->index] * 0.3;
    if (OP_partial_exchange)
      op_printf("Mapping %s partially exchanged: %d (%lld < 0.3*%lld)\n", map->name,
                OP_map_partial_exchange[m], (long long)map_halo_sizes[m],
                (long long)halo_sizes[map->to->index]);
  }

  /* A map without a partial exchange keeps empty lists. */
  OP_map_halos = std::vector<MapHalo>(OP_map_index);

  for (int m = 0; m < OP_map_index; m++) {
    if (!OP_map_partial_exchange[m])
      continue;
    op_map map = OP_map_list[m];
    op_set to = map->to;
    const SetHalo &halo = OP_set_halos[to->index];
    const HaloList *imp[2] = {&halo.import_exec, &halo.import_nonexec};
    const HaloList *exp[2] = {&halo.export_exec, &halo.export_nonexec};

    std::vector<char> reached(to->exec_size + to->nonexec_size, 0);
    for (int e = map->from->core_size; e < map->from->size + map->from->exec_size; e++)
      for (int j = 0; j < map->dim; j++)
        if (map->map[e * map->dim + j] >= to->size)
          reached[map->map[e * map->dim + j] - to->size] = 1;

    /* Every rank either halo region imports from, ascending, and how much of
       each one's halo is its exec block. */
    std::vector<int> owners(imp[0]->ranks);
    owners.insert(owners.end(), imp[1]->ranks.begin(), imp[1]->ranks.end());
    std::sort(owners.begin(), owners.end());
    owners.erase(std::unique(owners.begin(), owners.end()), owners.end());
    auto slot = [&](int rank) {
      return (int)(std::lower_bound(owners.begin(), owners.end(), rank) - owners.begin());
    };
    std::vector<idx_l_t> exec_from(owners.size(), 0);
    for (int b = 0; b < imp[0]->ranks_size(); b++)
      exec_from[slot(imp[0]->ranks[b])] = imp[0]->sizes[b];

    /* Each reached element with its owner's slot, its position in the halo from
       that owner, and its local index - in halo order, so every owner's
       elements come out in halo order too. */
    auto for_each_reached = [&](auto &&visit) {
      for (int r = 0; r < 2; r++) {
        const idx_l_t region = r == 0 ? 0 : to->exec_size;
        for (int b = 0; b < imp[r]->ranks_size(); b++) {
          const int o = slot(imp[r]->ranks[b]);
          const idx_l_t first = r == 0 ? 0 : exec_from[o];
          for (idx_l_t k = 0; k < imp[r]->sizes[b]; k++) {
            const idx_l_t h = region + imp[r]->disps[b] + k;
            if (reached[h])
              visit(o, first + k, to->size + h);
          }
        }
      }
    };

    /* Counting sort by owner. */
    std::vector<idx_l_t> count(owners.size(), 0);
    for_each_reached([&](int o, idx_l_t, idx_l_t) { count[o]++; });
    std::vector<idx_l_t> start(owners.size() + 1, 0);
    for (std::size_t o = 0; o < owners.size(); o++)
      start[o + 1] = start[o] + count[o];
    std::vector<int> positions(start.back());
    std::unique_ptr<idx_l_t[]> indices;
    if (start.back() > 0)
      indices = std::make_unique_for_overwrite<idx_l_t[]>(start.back());
    std::vector<idx_l_t> next(start.begin(), start.end() - 1);
    for_each_reached([&](int o, idx_l_t position, idx_l_t index) {
      positions[next[o]] = position;
      indices[next[o]++] = index;
    });

    /* positions is complete, so the messages can borrow it. */
    std::vector<int> ranks;
    std::vector<idx_l_t> sizes;
    std::vector<op::mpi::msg::BlockView<int>> messages;
    for (std::size_t o = 0; o < owners.size(); o++) {
      if (count[o] == 0)
        continue;
      ranks.push_back(owners[o]);
      sizes.push_back(count[o]);
      messages.emplace_back(owners[o], positions.data() + start[o], (std::size_t)count[o]);
    }
    OP_map_halos[m].import_nonexec =
        halo_list_from_groups(to, std::move(ranks), std::move(sizes), std::move(indices));

    /* As an owner: turn each requested position into the element's local index. */
    op::mpi::Received<int> wanted = op::mpi::sparse::exchange(OP_MPI_WORLD, messages);
    for (std::size_t g = 0; g < wanted.ranks.size(); g++) {
      const int be = rank_slot(*exp[0], wanted.ranks[g]);
      const int bn = rank_slot(*exp[1], wanted.ranks[g]);
      const idx_l_t exec_to = be < 0 ? 0 : exp[0]->sizes[be];
      for (int k = 0; k < wanted.counts[g]; k++) {
        int &p = wanted.data[wanted.disps[g] + k];
        assert(p < exec_to + (bn < 0 ? 0 : exp[1]->sizes[bn]));
        p = p < exec_to ? exp[0]->list[exp[0]->disps[be] + p]
                        : exp[1]->list[exp[1]->disps[bn] + p - exec_to];
      }
    }
    OP_map_halos[m].export_nonexec = halo_list_from_received(to, std::move(wanted));
  }
}

/*******************************************************************************
 * Routine to Clean-up all MPI halos(called at the end of an OP2 MPI
 *application)
 *******************************************************************************/

void op_halo_destroy() {
  // remove halos from op_dats
  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    dat->data = (char *)xrealloc(dat->data, (size_t)dat->set->size * dat->size);
  }

  OP_set_halos = std::vector<SetHalo>();

  // MPI_Comm_free(&OP_MPI_WORLD);
}

/*******************************************************************************
 * Routine to set the dirty bit for an MPI Halo after halo exchange
 *******************************************************************************/

static void set_dirtybit(op_arg *arg, int hd) {
  op_dat dat = arg->dat;

  if ((arg->opt == 1) && (arg->argtype == OP_ARG_DAT) &&
      (arg->acc == OP_INC || arg->acc == OP_WRITE || arg->acc == OP_RW)) {
    dat->dirtybit = 1;
    dat->dirty_hd = hd;
  }
}

void op_mpi_reduce_combined(op_arg *args, int nargs) {
  if (OP_disable_mpi_reductions)
    return;

  op_timers_core(&c1, &t1);
  int nreductions = 0;
  for (int i = 0; i < nargs; i++) {
    if (args[i].argtype == OP_ARG_GBL && args[i].acc != OP_READ && args[i].acc != OP_WORK)
      nreductions++;
  }
  op_arg *arg_list = (op_arg *)xmalloc(nreductions * sizeof(op_arg));
  nreductions = 0;
  int nbytes = 0;
  for (int i = 0; i < nargs; i++) {
    if (args[i].argtype == OP_ARG_GBL && args[i].acc != OP_READ && args[i].acc != OP_WORK) {
      arg_list[nreductions++] = args[i];
      nbytes += args[i].size;
    }
  }
  char *data = (char *)xmalloc(nbytes * sizeof(char));
  int char_counter = 0;
  for (int i = 0; i < nreductions; i++) {
    for (int j = 0; j < arg_list[i].size; j++)
      data[char_counter++] = arg_list[i].data[j];
  }

  int comm_size, comm_rank;
  MPI_Comm_size(OP_MPI_WORLD, &comm_size);
  MPI_Comm_rank(OP_MPI_WORLD, &comm_rank);
  char *result = (char *)xmalloc(comm_size * nbytes * sizeof(char));
  MPI_Allgather(data, nbytes, MPI_CHAR, result, nbytes, MPI_CHAR, OP_MPI_WORLD);

  char_counter = 0;
  for (int i = 0; i < nreductions; i++) {
    if (strcmp(arg_list[i].type, "double") == 0 ||
        strcmp(arg_list[i].type, "r8") == 0) {
      double *output = (double *)arg_list[i].data;
      for (int rank = 0; rank < comm_size; rank++) {
        if (rank != comm_rank) {
          if (arg_list[i].acc == OP_INC) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] +=
                  ((double *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_MIN) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] <
                          ((double *)(result + char_counter + nbytes * rank))[j]
                      ? output[j]
                      : ((double *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_MAX) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] >
                          ((double *)(result + char_counter + nbytes * rank))[j]
                      ? output[j]
                      : ((double *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_WRITE) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] != 0.0
                      ? output[j]
                      : ((double *)(result + char_counter + nbytes * rank))[j];
            }
          }
        }
      }
    }
    if (strcmp(arg_list[i].type, "float") == 0 ||
        strcmp(arg_list[i].type, "r4") == 0 ||
        strcmp(arg_list[i].type, "real*4") == 0) {
      float *output = (float *)arg_list[i].data;
      for (int rank = 0; rank < comm_size; rank++) {
        if (rank != comm_rank) {
          if (arg_list[i].acc == OP_INC) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] +=
                  ((float *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_MIN) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] <
                          ((float *)(result + char_counter + nbytes * rank))[j]
                      ? output[j]
                      : ((float *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_MAX) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] >
                          ((float *)(result + char_counter + nbytes * rank))[j]
                      ? output[j]
                      : ((float *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_WRITE) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] != 0.0
                      ? output[j]
                      : ((float *)(result + char_counter + nbytes * rank))[j];
            }
          }
        }
      }
    }
    if (strcmp(arg_list[i].type, "int") == 0 ||
        strcmp(arg_list[i].type, "i4") == 0 ||
        strcmp(arg_list[i].type, "integer*4") == 0) {
      int *output = (int *)arg_list[i].data;
      for (int rank = 0; rank < comm_size; rank++) {
        if (rank != comm_rank) {
          if (arg_list[i].acc == OP_INC) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] += ((int *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_MIN) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] <
                          ((int *)(result + char_counter + nbytes * rank))[j]
                      ? output[j]
                      : ((int *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_MAX) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] >
                          ((int *)(result + char_counter + nbytes * rank))[j]
                      ? output[j]
                      : ((int *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_WRITE) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] != 0.0
                      ? output[j]
                      : ((int *)(result + char_counter + nbytes * rank))[j];
            }
          }
        }
      }
    }
    if (strcmp(arg_list[i].type, "bool") == 0 ||
        strcmp(arg_list[i].type, "logical") == 0) {
      bool *output = (bool *)arg_list[i].data;
      for (int rank = 0; rank < comm_size; rank++) {
        if (rank != comm_rank) {
          if (arg_list[i].acc == OP_INC) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] += ((bool *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_MIN) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] <
                          ((bool *)(result + char_counter + nbytes * rank))[j]
                      ? output[j]
                      : ((bool *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_MAX) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] >
                          ((bool *)(result + char_counter + nbytes * rank))[j]
                      ? output[j]
                      : ((bool *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_WRITE) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] != 0.0
                      ? output[j]
                      : ((bool *)(result + char_counter + nbytes * rank))[j];
            }
          }
        }
      }
    }
    char_counter += arg_list[i].size;
  }
  op_timers_core(&c2, &t2);
  if (OP_kern_max > 0)
    OP_kernels[OP_kern_curr].mpi_time += t2 - t1;
  op_free(arg_list);
  op_free(data);
  op_free(result);
}

extern "C++" {

/* One body for every element type of a global reduction; op_mpi_reduce_<type>
   below only names the MPI datatype. */
template <typename T>
static void reduce_gbl(op_arg *arg, MPI_Datatype type) {
  if (OP_disable_mpi_reductions || arg->data == NULL)
    return;
  op_timers_core(&c1, &t1);
  if (arg->argtype == OP_ARG_GBL) {
    T *data = (T *)arg->data;
    if (arg->acc == OP_INC || arg->acc == OP_MAX || arg->acc == OP_MIN) {
      MPI_Op op = arg->acc == OP_INC ? MPI_SUM : arg->acc == OP_MAX ? MPI_MAX : MPI_MIN;
      MPI_Allreduce(MPI_IN_PLACE, data, arg->dim, type, op, OP_MPI_WORLD);
    } else if (arg->acc == OP_WRITE) {
      // Any rank's value: the last rank's that is not zero, else rank 0's.
      int size;
      MPI_Comm_size(OP_MPI_WORLD, &size);
      std::unique_ptr<T[]> all(new T[(std::size_t)arg->dim * size]());
      MPI_Allgather(data, arg->dim, type, all.get(), arg->dim, type, OP_MPI_WORLD);
      for (int i = 1; i < size; i++)
        for (int j = 0; j < arg->dim; j++)
          if (all[i * arg->dim + j] != T(0))
            all[j] = all[i * arg->dim + j];
      std::copy_n(all.get(), arg->dim, data);
    }
  }
  op_timers_core(&c2, &t2);
  if (OP_kern_max > 0)
    OP_kernels[OP_kern_curr].mpi_time += t2 - t1;
}

}  // extern "C++"

void op_mpi_reduce_float(op_arg *arg, float *) { reduce_gbl<float>(arg, MPI_FLOAT); }

void op_mpi_reduce_double(op_arg *arg, double *) { reduce_gbl<double>(arg, MPI_DOUBLE); }

void op_mpi_reduce_int(op_arg *arg, int *) { reduce_gbl<int>(arg, MPI_INT); }

/* MPI_CHAR as it always has been: one byte, like bool, though the MPI standard
   does not list MPI_CHAR among the reduction types. */
void op_mpi_reduce_bool(op_arg *arg, bool *) { reduce_gbl<bool>(arg, MPI_CHAR); }

/*******************************************************************************
 * Routine to get a copy of the data held in a distributed op_dat
 *******************************************************************************/

op_dat op_mpi_get_data(op_dat dat) {
  // create new communicator for fetching
  int my_rank, comm_size;
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_WORLD, &comm_size);

  //
  // make a copy of the distributed op_dat on to a distributed temporary op_dat
  //
  op_dat temp_dat = (op_dat)xmalloc(sizeof(op_dat_core));
  char *data = (char *)xmalloc((size_t)dat->set->size * dat->size);
  memcpy(data, dat->data, dat->set->size * (size_t)dat->size);

  //
  // use orig_part_range to fill in OP_part_list[set->index]->elem_part with
  // original partitioning information
  //
  for (int i = 0; i < dat->set->size; i++) {
    int local_index;
    OP_part_list[dat->set->index]->elem_part[i] = get_partition(
        OP_part_list[dat->set->index]->g_index[i],
        orig_part_range[dat->set->index], &local_index, comm_size, dat->set);
  }


  //
  // create export list
  //
  part p = OP_part_list[dat->set->index];
  int count = 0;
  int cap = 1000;
  int *temp_list = (int *)xmalloc(cap * sizeof(int));

  for (int i = 0; i < dat->set->size; i++) {
    if (p->elem_part[i] != my_rank) {
      if (count >= cap) {
        cap = cap * 2;
        temp_list = (int *)xrealloc(temp_list, cap * sizeof(int));
      }
      temp_list[count++] = p->elem_part[i];
      temp_list[count++] = i;
    }
  }

  HaloList pe_list = halo_list_from_pairs(dat->set, temp_list, count);
  op_free(temp_list);

  //
  // create import list
  //
  HaloList pi_list = halo_list_transpose(dat->set, pe_list, OP_MPI_WORLD);

  /* Reused by both data migrations below. */
  MPI_Request *request_send =
      (MPI_Request *)xmalloc(pe_list.ranks_size() * sizeof(MPI_Request));

  //
  // migrate the temp "data" array to the original MPI ranks
  //

  // prepare bits of the data array to be exported
  char **sbuf_char = (char **)xmalloc(pe_list.ranks_size() * sizeof(char *));

  for (int i = 0; i < pe_list.ranks_size(); i++) {
    sbuf_char[i] = (char *)xmalloc((size_t)pe_list.sizes[i] * (size_t)dat->size);
    for (int j = 0; j < pe_list.sizes[i]; j++) {
      int index = pe_list.list[pe_list.disps[i] + j];
      memcpy(&sbuf_char[i][j * (size_t)dat->size], (void *)&data[(size_t)dat->size * (index)],
             dat->size);
    }
    MPI_Isend(sbuf_char[i], (size_t)dat->size * pe_list.sizes[i], MPI_CHAR,
              pe_list.ranks[i], dat->index, OP_MPI_WORLD, &request_send[i]);
  }

  char *rbuf_char = (char *)xmalloc((size_t)dat->size * pi_list.size());
  for (int i = 0; i < pi_list.ranks_size(); i++) {
    MPI_Recv(&rbuf_char[pi_list.disps[i] * (size_t)dat->size],
             (size_t)dat->size * pi_list.sizes[i], MPI_CHAR, pi_list.ranks[i],
             dat->index, OP_MPI_WORLD, MPI_STATUS_IGNORE);
  }

  MPI_Waitall(pe_list.ranks_size(), request_send, MPI_STATUSES_IGNORE);
  for (int i = 0; i < pe_list.ranks_size(); i++)
    op_free(sbuf_char[i]);
  op_free(sbuf_char);

  // delete the data entirs that has been sent and create a
  // modified data array
  char *new_dat = (char *)xmalloc((size_t)dat->size * (dat->set->size + pi_list.size()));

  count = 0;
  for (int i = 0; i < dat->set->size; i++) // iterate over old set size
  {
    if (OP_part_list[dat->set->index]->elem_part[i] == my_rank) {
      memcpy(&new_dat[count * (size_t)dat->size], (void *)&data[(size_t)dat->size * i],
             dat->size);
      count++;
    }
  }

  if ((size_t)dat->size * pi_list.size() > 0) {
    memcpy(&new_dat[count * (size_t)dat->size], (void *)rbuf_char,
           (size_t)dat->size * pi_list.size());
  }

  count = count + pi_list.size();
  new_dat = (char *)xrealloc(new_dat, (size_t)dat->size * count);
  op_free(rbuf_char);
  op_free(data);
  data = new_dat;

  //
  // make a copy of the original g_index and migrate that also to the original
  // MPI process
  //
  // prepare bits of the original g_index array to be exported
  idx_g_t **sbuf = (idx_g_t **)xmalloc(pe_list.ranks_size() * sizeof(idx_g_t *));

  // send original g_index values to relevant mpi processes
  for (int i = 0; i < pe_list.ranks_size(); i++) {
    sbuf[i] = (idx_g_t *)xmalloc(pe_list.sizes[i] * sizeof(idx_g_t));
    for (int j = 0; j < pe_list.sizes[i]; j++) {
      sbuf[i][j] = OP_part_list[dat->set->index]
                       ->g_index[pe_list.list[pe_list.disps[i] + j]];
    }
    MPI_Isend(sbuf[i], pe_list.sizes[i], get_mpi_type(sbuf[i]), pe_list.ranks[i],
              dat->index, OP_MPI_WORLD, &request_send[i]);
  }

  idx_g_t *rbuf = (idx_g_t *)xmalloc(sizeof(idx_g_t) * pi_list.size());

  // receive original g_index values from relevant mpi processes
  for (int i = 0; i < pi_list.ranks_size(); i++) {
    MPI_Recv(&rbuf[pi_list.disps[i]], pi_list.sizes[i], get_mpi_type(rbuf),
             pi_list.ranks[i], dat->index, OP_MPI_WORLD, MPI_STATUS_IGNORE);
  }
  MPI_Waitall(pe_list.ranks_size(), request_send, MPI_STATUSES_IGNORE);
  for (int i = 0; i < pe_list.ranks_size(); i++)
    op_free(sbuf[i]);
  op_free(sbuf);

  // delete the g_index entirs that has been sent and create a
  // modified g_index
  idx_g_t *new_g_index =
      (idx_g_t *)xmalloc(sizeof(idx_g_t) * (dat->set->size + pi_list.size()));

  count = 0;
  for (int i = 0; i < dat->set->size; i++) { // iterate over old
                                             // size of the g_index array
    if (OP_part_list[dat->set->index]->elem_part[i] == my_rank) {
      new_g_index[count] = OP_part_list[dat->set->index]->g_index[i];
      count++;
    }
  }

  if (sizeof(idx_g_t) * pi_list.size() > 0)
    memcpy(&new_g_index[count], (void *)rbuf, sizeof(idx_g_t) * pi_list.size());

  count = count + pi_list.size();
  new_g_index = (idx_g_t *)xrealloc(new_g_index, sizeof(idx_g_t) * count);
  op_free(rbuf);

  //
  // sort elements in temporaty data according to new_g_index
  //
  op_sort_dat(new_g_index, data, count, dat->size);

  // cleanup
  op_free(new_g_index);
  op_free(request_send);

  // remember that the original set size is now given by count
  op_set set = (op_set)xmalloc(sizeof(op_set_core));
  set->index = dat->set->index;
  set->size = count;
  set->name = dat->set->name;

  temp_dat->index = dat->index;
  temp_dat->set = set;
  temp_dat->dim = dat->dim;
  temp_dat->data = data;
  temp_dat->data_d = NULL;
  temp_dat->name = dat->name;
  temp_dat->type = dat->type;
  temp_dat->size = dat->size;

  return temp_dat;
}

/*******************************************************************************
 * Routine to put user data held in the original block partition into a
 * distributed op_dat (reverse of op_mpi_get_data)
 *******************************************************************************/

void op_mpi_put_data(op_dat dat, void *ptr, size_t local_size) {
  int my_rank, comm_size;
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_WORLD, &comm_size);

  char *src = (char *)ptr;

  // No partitioning information: data is already in declaration order
  if (orig_part_range == NULL || OP_part_list == NULL) {
    if (local_size != (size_t)dat->set->size) {
      printf("Error: op_mpi_put_data local_size %zu does not match set size %d "
             "for dat %s\n",
             local_size, dat->set->size, dat->name);
      MPI_Abort(OP_MPI_WORLD, 2);
    }
    if (local_size > 0)
      memcpy(dat->data, src, local_size * (size_t)dat->size);
    dat->dirtybit = 1;
    dat->dirty_hd = 1;
    return;
  }

  idx_g_t orig_start = orig_part_range[dat->set->index][2 * my_rank];
  idx_g_t orig_end = orig_part_range[dat->set->index][2 * my_rank + 1];
  size_t orig_size =
      (orig_end >= orig_start) ? (size_t)(orig_end - orig_start + 1) : 0;

  if (local_size != orig_size) {
    printf("Error: op_mpi_put_data local_size %zu does not match original "
           "partition size %zu for dat %s on rank %d\n",
           local_size, orig_size, dat->name, my_rank);
    MPI_Abort(OP_MPI_WORLD, 2);
  }

  //
  // for each currently owned element, find the original rank and local index
  //
  part p = OP_part_list[dat->set->index];
  int *orig_rank = (int *)xmalloc(sizeof(int) * dat->set->size);
  int *orig_local = (int *)xmalloc(sizeof(int) * dat->set->size);

  for (int i = 0; i < dat->set->size; i++) {
    orig_rank[i] = get_partition(p->g_index[i], orig_part_range[dat->set->index],
                                 &orig_local[i], comm_size, dat->set);
  }


  //
  // create export list: current-local elements that originally belonged
  // elsewhere (we need to receive data for these from the original ranks)
  //
  int count = 0;
  int cap = 1000;
  int *temp_list = (int *)xmalloc(cap * sizeof(int));

  for (int i = 0; i < dat->set->size; i++) {
    if (orig_rank[i] != my_rank) {
      if (count >= cap) {
        cap = cap * 2;
        temp_list = (int *)xrealloc(temp_list, cap * sizeof(int));
      }
      temp_list[count++] = orig_rank[i];
      temp_list[count++] = i;
    } else {
      memcpy(&dat->data[(size_t)dat->size * i],
             &src[(size_t)dat->size * orig_local[i]], dat->size);
    }
  }

  HaloList pe_list = halo_list_from_pairs(dat->set, temp_list, count);
  op_free(temp_list);

  //
  // create import list: send each original rank the original local indices it
  // must supply, so pi_list.list is directly the pack list for the data below
  //
  /* Unlike the other sites this sends a translated index, not the list entry
     itself, so the payload has to be materialised - but in pe_list order, so
     pe_list's own ranks and sizes describe it. */
  std::vector<int> want;
  want.reserve(pe_list.size());
  for (int i = 0; i < pe_list.ranks_size(); i++)
    for (int j = 0; j < pe_list.sizes[i]; j++)
      want.push_back(orig_local[pe_list.list[pe_list.disps[i] + j]]);

  std::vector<op::mpi::msg::BlockView<int>> messages;
  messages.reserve(pe_list.ranks_size());
  for (int i = 0; i < pe_list.ranks_size(); i++)
    messages.emplace_back(pe_list.ranks[i], want.data() + pe_list.disps[i],
                          (std::size_t)pe_list.sizes[i]);

  HaloList pi_list = halo_list_from_received(dat->set, op::mpi::sparse::exchange(OP_MPI_WORLD, messages));

  //
  // original ranks pack user data and send it to the current owners
  //
  char **sbuf_char = (char **)xmalloc(pi_list.ranks_size() * sizeof(char *));
  MPI_Request *request_send_data =
      (MPI_Request *)xmalloc(pi_list.ranks_size() * sizeof(MPI_Request));

  for (int i = 0; i < pi_list.ranks_size(); i++) {
    sbuf_char[i] =
        (char *)xmalloc((size_t)pi_list.sizes[i] * (size_t)dat->size);
    for (int j = 0; j < pi_list.sizes[i]; j++) {
      int ol = pi_list.list[pi_list.disps[i] + j];
      if (ol < 0 || (size_t)ol >= orig_size) {
        printf("Error: op_mpi_put_data original local index %d out of range "
               "(orig_size %zu) for dat %s on rank %d\n",
               ol, orig_size, dat->name, my_rank);
        MPI_Abort(OP_MPI_WORLD, 2);
      }
      memcpy(&sbuf_char[i][j * (size_t)dat->size],
             &src[(size_t)dat->size * ol], dat->size);
    }
    MPI_Isend(sbuf_char[i], (size_t)dat->size * pi_list.sizes[i], MPI_CHAR,
              pi_list.ranks[i], dat->index, OP_MPI_WORLD,
              &request_send_data[i]);
  }

  char *rbuf_char = (char *)xmalloc((size_t)dat->size * pe_list.size());
  for (int i = 0; i < pe_list.ranks_size(); i++) {
    MPI_Recv(&rbuf_char[pe_list.disps[i] * (size_t)dat->size],
             (size_t)dat->size * pe_list.sizes[i], MPI_CHAR, pe_list.ranks[i],
             dat->index, OP_MPI_WORLD, MPI_STATUS_IGNORE);
  }

  MPI_Waitall(pi_list.ranks_size(), request_send_data, MPI_STATUSES_IGNORE);
  for (int i = 0; i < pi_list.ranks_size(); i++)
    op_free(sbuf_char[i]);
  op_free(sbuf_char);
  op_free(request_send_data);

  // scatter received data into current local positions
  for (int i = 0; i < pe_list.size(); i++) {
    int index = pe_list.list[i];
    memcpy(&dat->data[(size_t)dat->size * index],
           &rbuf_char[(size_t)dat->size * i], dat->size);
  }
  op_free(rbuf_char);

  // cleanup
  op_free(orig_rank);
  op_free(orig_local);

  dat->dirtybit = 1;
  dat->dirty_hd = 1;
}

/*******************************************************************************
 * Debug/Diagnostics Routine to initialise import halo data to NaN
 *******************************************************************************/

static void op_reset_halo(op_arg *arg) {
  op_dat dat = arg->dat;

  if ((arg->opt) && (arg->argtype == OP_ARG_DAT) &&
      (arg->acc == OP_READ || arg->acc == OP_RW) && (dat->dirtybit == 1)) {
    // printf("Resetting Halo of data array %10s\n",dat->name);
    const HaloList &imp_exec_list = OP_set_halos[dat->set->index].import_exec;
    const HaloList &imp_nonexec_list = OP_set_halos[dat->set->index].import_nonexec;

    // initialise import halo data to NaN
    int double_count = imp_exec_list.size() * (size_t)dat->size / sizeof(double);
    double_count += imp_nonexec_list.size() * (size_t)dat->size / sizeof(double);
    double *NaN = (double *)xmalloc(double_count * sizeof(double));
    for (int i = 0; i < double_count; i++)
      NaN[i] = (double)NAN; // 0.0/0.0;

    int init = dat->set->size * (size_t)dat->size;
    memcpy(&(dat->data[init]), NaN, (size_t)dat->size * imp_exec_list.size() +
                                        (size_t)dat->size * imp_nonexec_list.size());
    op_free(NaN);
  }
}

void op_compute_moment(double t, double *first, double *second) {
  double times[2];
  double times_reduced[2];
  int comm_size;
  times[0] = t;
  times[1] = t * t;
  MPI_Comm_size(OP_MPI_WORLD, &comm_size);
  MPI_Reduce(times, times_reduced, 2, MPI_DOUBLE, MPI_SUM, 0, OP_MPI_WORLD);

  *first = times_reduced[0] / (double)comm_size;
  *second = times_reduced[1] / (double)comm_size;
}

void op_compute_moment_across_times(double* times, int ntimes, bool ignore_zeros, double *first, double *second) {
  double times_moment[2] = {0.0f, 0.0f};

  int num_times = 0;
  for (int i=0; i<ntimes; i++) {
    if (ignore_zeros && (times[i] == 0.0f)) {
      continue;
    }
    times_moment[0] += times[i];
    times_moment[1] += times[i] * times[i];
    num_times++;
  }

  double times_moment_world[2] = {0.0f, 0.0f};
  MPI_Reduce(times_moment, times_moment_world, 2, MPI_DOUBLE, MPI_SUM, 0, OP_MPI_WORLD);

  int num_times_world;
  MPI_Reduce(&num_times, &num_times_world, 1, get_mpi_type(&num_times), MPI_SUM, 0, OP_MPI_WORLD);

  if (num_times_world != 0) {
    *first = times_moment_world[0] / (double)num_times_world;
    *second = times_moment_world[1] / (double)num_times_world;
  }
}


/*******************************************************************************
 * Routine to output performance measures
 *******************************************************************************/
void mpi_timing_output() {
  int my_rank, comm_size;
  MPI_Comm OP_MPI_IO_WORLD;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_MPI_IO_WORLD);
  MPI_Comm_rank(OP_MPI_IO_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_IO_WORLD, &comm_size);

  int count, tot_count;
  count = op_mpi_kernel_map.size();
  MPI_Allreduce(&count, &tot_count, 1, get_mpi_type(&count), MPI_SUM, OP_MPI_IO_WORLD);

  if (tot_count > 0) {
    double tot_time;
    double avg_time;

    printf("___________________________________________________\n");
    printf("Performance information on rank %d\n", my_rank);
    printf("Kernel        Count  total time(sec)  Avg time(sec)  \n");

    op_mpi_kernel *k;
    for (auto it = op_mpi_kernel_map.begin(); it != op_mpi_kernel_map.end(); it++) {
      k = it->second;
      if (k->count > 0) {
        printf("%-10s  %6d       %10.4f      %10.4f    \n", k->name, k->count,
               k->time, k->time / k->count);

#ifdef COMM_PERF
        if (k->num_indices > 0) {
          printf("halo exchanges:  ");
          for (int i = 0; i < k->num_indices; i++)
            printf("%10s ", k->comm_info[i]->name);
          printf("\n");
          printf("       count  :  ");
          for (int i = 0; i < k->num_indices; i++)
            printf("%10d ", k->comm_info[i]->count);
          printf("\n");
          printf("total(Kbytes) :  ");
          for (int i = 0; i < k->num_indices; i++)
            printf("%10d ", k->comm_info[i]->bytes / 1024);
          printf("\n");
          printf("average(bytes):  ");
          for (int i = 0; i < k->num_indices; i++)
            printf("%10d ", k->comm_info[i]->bytes / k->comm_info[i]->count);
          printf("\n");
        } else {
          printf("halo exchanges:  %10s\n", "NONE");
        }
        printf("---------------------------------------------------\n");
#endif
      }
    }
    printf("___________________________________________________\n");

    if (my_rank == MPI_ROOT) {
      printf("___________________________________________________\n");
      printf("\nKernel        Count   Max time(sec)   Avg time(sec)  \n");
    }

    for (auto it = op_mpi_kernel_map.begin(); it != op_mpi_kernel_map.end(); it++) {
      k = it->second;
      MPI_Reduce(&(k->count), &count, 1, MPI_INT, MPI_MAX, MPI_ROOT,
                 OP_MPI_IO_WORLD);
      MPI_Reduce(&(k->time), &avg_time, 1, MPI_DOUBLE, MPI_SUM, MPI_ROOT,
                 OP_MPI_IO_WORLD);
      MPI_Reduce(&(k->time), &tot_time, 1, MPI_DOUBLE, MPI_MAX, MPI_ROOT,
                 OP_MPI_IO_WORLD);

      if (my_rank == MPI_ROOT && count > 0) {
        printf("%-10s  %6d       %10.4f      %10.4f    \n", k->name, count,
               tot_time, (avg_time) / comm_size);
      }
      tot_time = avg_time = 0.0;
    }
  }
  MPI_Comm_free(&OP_MPI_IO_WORLD);
}

/*******************************************************************************
 * Routine to measure timing for an op_par_loop / kernel
 *******************************************************************************/
void *op_mpi_perf_time(const char *name, double time) {
  op_mpi_kernel *kernel_entry;

  auto it = op_mpi_kernel_map.find(name);
  if (it == op_mpi_kernel_map.end()) {
    kernel_entry = (op_mpi_kernel *)xmalloc(sizeof(op_mpi_kernel));
    kernel_entry->num_indices = 0;
    kernel_entry->time = 0.0;
    kernel_entry->count = 0;
    strncpy((char *)kernel_entry->name, name, NAMESIZE);
    op_mpi_kernel_map[name] = kernel_entry;
  } else {
    kernel_entry = it->second;
  }

  kernel_entry->count += 1;
  kernel_entry->time += time;

  return (void *)kernel_entry;
}
#ifdef COMM_PERF

/*******************************************************************************
 * Routine to linear search comm_info array in an op_mpi_kernel for an op_dat
 *******************************************************************************/
int search_op_mpi_kernel(op_dat dat, op_mpi_kernel *kernal, int num_indices) {
  for (int i = 0; i < num_indices; i++) {
    if (strcmp((kernal->comm_info[i])->name, dat->name) == 0 &&
        (kernal->comm_info[i])->size == dat->size) {
      return i;
    }
  }

  return -1;
}

/*******************************************************************************
 * Routine to measure MPI message sizes exchanged in an op_par_loop / kernel
 *******************************************************************************/
void op_mpi_perf_comm(void *k_i, op_dat dat) {
  const HaloList &exp_exec_list = OP_set_halos[dat->set->index].export_exec;
  const HaloList &exp_nonexec_list = OP_set_halos[dat->set->index].export_nonexec;
  int tot_halo_size =
      (exp_exec_list.size() + exp_nonexec_list.size()) * (size_t)dat->size;

  op_mpi_kernel *kernel_entry = (op_mpi_kernel *)k_i;
  int num_indices = kernel_entry->num_indices;

  if (num_indices == 0) {
    // set capcity of comm_info array
    kernel_entry->cap = 20;
    op_dat_mpi_comm_info dat_comm =
        (op_dat_mpi_comm_info)xmalloc(sizeof(op_dat_mpi_comm_info_core));
    kernel_entry->comm_info = (op_dat_mpi_comm_info *)xmalloc(
        sizeof(op_dat_mpi_comm_info *) * (kernel_entry->cap));
    strncpy((char *)dat_comm->name, dat->name, 20);
    dat_comm->size = dat->size;
    dat_comm->index = dat->index;
    dat_comm->count = 0;
    dat_comm->bytes = 0;

    // add first values
    dat_comm->count += 1;
    dat_comm->bytes += tot_halo_size;

    kernel_entry->comm_info[num_indices] = dat_comm;
    kernel_entry->num_indices++;
  } else {
    int index = search_op_mpi_kernel(dat, kernel_entry, num_indices);
    if (index < 0) {
      // increase capacity of comm_info array
      if (num_indices >= kernel_entry->cap) {
        kernel_entry->cap = kernel_entry->cap * 2;
        kernel_entry->comm_info = (op_dat_mpi_comm_info *)xrealloc(
            kernel_entry->comm_info,
            sizeof(op_dat_mpi_comm_info *) * (kernel_entry->cap));
      }

      op_dat_mpi_comm_info dat_comm =
          (op_dat_mpi_comm_info)xmalloc(sizeof(op_dat_mpi_comm_info_core));

      strncpy((char *)dat_comm->name, dat->name, 20);
      dat_comm->size = dat->size;
      dat_comm->index = dat->index;
      dat_comm->count = 0;
      dat_comm->bytes = 0;

      // add first values
      dat_comm->count += 1;
      dat_comm->bytes += tot_halo_size;

      kernel_entry->comm_info[num_indices] = dat_comm;
      kernel_entry->num_indices++;
    } else {
      kernel_entry->comm_info[index]->count += 1;
      kernel_entry->comm_info[index]->bytes += tot_halo_size;
    }
  }
}
#endif

#ifdef COMM_PERF
void op_mpi_perf_comms(void *k_i, int nargs, op_arg *args) {

  for (int n = 0; n < nargs; n++) {
    if (args[n].argtype == OP_ARG_DAT && args[n].sent == 2) {
      op_mpi_perf_comm(k_i, (&args[n])->dat);
    }
  }
}
#endif

/*******************************************************************************
 * Routine to exit an op2 mpi application -
 *******************************************************************************/

void op_mpi_exit() {
  op_mpi_unified_exit();

  // cleanup performance data - need to do this in some op_mpi_exit() routine
  op_mpi_kernel *kernel_entry;
  for (auto it = op_mpi_kernel_map.begin(); it != op_mpi_kernel_map.end();) {
    kernel_entry = it->second;
#ifdef COMM_PERF
    for (int i = 0; i < kernel_entry->num_indices; i++)
      op_free(kernel_entry->comm_info[i]); 
#endif
    it = op_mpi_kernel_map.erase(it);
    op_free(kernel_entry);
  }

  // free memory allocated to halos
  op_halo_destroy();
  // free memory used for holding partition information
  op_partition_destroy();
  OP_map_halos = std::vector<MapHalo>();
  op_free(OP_map_partial_exchange);
}

int getSetSizeFromOpArg(op_arg *arg) {
  if (arg->dat->set->size + OP_set_halos[arg->dat->set->index].import_exec.size() +
      OP_set_halos[arg->dat->set->index].import_nonexec.size() >
      std::numeric_limits<int>::max()) {
    throw std::overflow_error("Set size is too large to be represented as an int");
  }
  return arg->opt ? (int)(arg->dat->set->size +
                     OP_set_halos[arg->dat->set->index].import_exec.size() +
                     OP_set_halos[arg->dat->set->index].import_nonexec.size())
                  : 0;
}

int getHybridGPU() { return OP_hybrid_gpu; }

void op_mpi_set_dirtybit(int nargs, op_arg *args) {

  for (int n = 0; n < nargs; n++) {
    if (args[n].argtype == OP_ARG_DAT) {
      set_dirtybit(&args[n], 1);
    }
  }
}

void op_mpi_set_dirtybit_cuda(int nargs, op_arg *args) {

  for (int n = 0; n < nargs; n++) {
    if (args[n].argtype == OP_ARG_DAT) {
      set_dirtybit(&args[n], 2);
    }
  }
}

/*******************************************************************************
 * The per-dat and grouped exchange entry points
 *
 * translator-v1 code, and anything else written against the older API, calls
 * these. The per-dat and grouped exchanges that implemented them are gone; they
 * now run the unified exchange, which writes the same halo bytes, and keep the
 * two things the old drivers did around it: argument checks under OP_diags, and
 * MPI time charged to the current kernel.
 *******************************************************************************/

static int exchange_via_unified(op_set set, int nargs, op_arg *args, int device) {
  if (OP_diags > 0) {
    int dummy;
    for (int n = 0; n < nargs; n++)
      op_arg_check(set, n, args[n], &dummy, "halo_exchange mpi");
  }
  double cpu_start, wall_start, cpu_end, wall_end;
  op_timers_core(&cpu_start, &wall_start);
  const int size = op_mpi_halo_exchanges_unified(set, nargs, args, device);
  op_timers_core(&cpu_end, &wall_end);
  if (OP_kern_max > 0)
    OP_kernels[OP_kern_curr].mpi_time += wall_end - wall_start;
  return size;
}

static void wait_via_unified(int nargs, op_arg *args) {
  double cpu_start, wall_start, cpu_end, wall_end;
  op_timers_core(&cpu_start, &wall_start);
  op_mpi_wait_all_unified(nargs, args);
  op_timers_core(&cpu_end, &wall_end);
  if (OP_kern_max > 0)
    OP_kernels[OP_kern_curr].mpi_time += wall_end - wall_start;
}

int op_mpi_halo_exchanges(op_set set, int nargs, op_arg *args) {
  return exchange_via_unified(set, nargs, args, 1);
}

int op_mpi_halo_exchanges_cuda(op_set set, int nargs, op_arg *args) {
  return exchange_via_unified(set, nargs, args, 2);
}

int op_mpi_halo_exchanges_grouped(op_set set, int nargs, op_arg *args, int device) {
  return exchange_via_unified(set, nargs, args, device);
}

void op_mpi_wait_all(int nargs, op_arg *args) { wait_via_unified(nargs, args); }

void op_mpi_wait_all_cuda(int nargs, op_arg *args) { wait_via_unified(nargs, args); }

void op_mpi_wait_all_grouped(int nargs, op_arg *args, int device) {
  (void)device;
  wait_via_unified(nargs, args);
}

void op_mpi_test_all(int nargs, op_arg *args) { op_mpi_test_all_unified(nargs, args); }

void op_mpi_test_all_grouped(int nargs, op_arg *args) { op_mpi_test_all_unified(nargs, args); }

void op_mpi_reset_halos(int nargs, op_arg *args) {
  for (int n = 0; n < nargs; n++) {
    op_reset_halo(&args[n]);
  }
}

void op_mpi_barrier() { MPI_Barrier(OP_MPI_WORLD); }

int op_is_root() {
  int my_rank;
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  return my_rank == MPI_ROOT;
}

/*******************************************************************************
 * Get the global size of a set
 *******************************************************************************/

idx_g_t op_get_size(op_set set) {
  int my_rank, comm_size;

  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_WORLD, &comm_size);

  idx_g_t *sizes = (idx_g_t *)xmalloc(sizeof(idx_g_t) * comm_size);
  idx_g_t size = set->size;
  MPI_Allgather(&size, 1, get_mpi_type(&size), 
                sizes, 1, get_mpi_type(sizes), OP_MPI_WORLD);

  idx_g_t g_size = 0;
  for (int i = 0; i < comm_size; i++)
    g_size = g_size + sizes[i];

  op_free(sizes);
  return g_size;
}

idx_g_t op_get_global_set_offset(op_set set) {
  int my_rank, comm_size;

  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_WORLD, &comm_size);

  idx_g_t *sizes = (idx_g_t *)xmalloc(sizeof(idx_g_t) * comm_size);
  idx_g_t size = set->size;
  MPI_Allgather(&size, 1, get_mpi_type(&size), 
                sizes, 1, get_mpi_type(sizes), OP_MPI_WORLD);


  idx_g_t g_offset = 0;
  for (int i = 0; i < my_rank; i++)
    g_offset = g_offset + sizes[i];

  op_free(sizes);
  return g_offset;
}

#ifdef __cplusplus
}
#endif
