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
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>

#include <op_mpi_core.h>
#include <op_mpi_halo.h>

using op::mpi::exchange_rows;
using op::mpi::fail;
using op::mpi::HaloList;
using op::mpi::MapHalo;
using op::mpi::PartRange;
using op::mpi::migrate_rows;
using op::mpi::SetHalo;

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

/* Time spent in each loop, by loop name */
struct op_mpi_kernel {
  double time = 0.0; // total time spent in this kernel (compute + comm - overlap)
  int count = 0;     // number of times this kernel is called
};
static std::unordered_map<std::string, op_mpi_kernel> op_mpi_kernel_map;

//
// global variables to hold partition information on an MPI rank
//

int OP_part_index = 0;
part *OP_part_list;

//
// Save original partition ranges
//

std::vector<PartRange> orig_part_range;

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

/* A map from global indices, kept as map_gbl until halo creation. The caller
   registers the array the pointer API names the map by. */
static op_map decl_map_global(op_set from, op_set to, int dim, const idx_g_t *imap_g, char const *name) {

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
  return map;
}

op_map op_decl_map(op_set from, op_set to, int dim, int *imap,
                   char const *name) {

  idx_g_t *imap_g = (idx_g_t *)malloc((idx_g_t)from->size * dim * sizeof(idx_g_t));
  for (idx_g_t i = 0; i < (idx_g_t)from->size * dim; i++) {
    imap_g[i] = (idx_g_t)imap[i];
  }

  op_map map = decl_map_global(from, to, dim, imap_g, name);
  free(imap_g);
  if (imap != NULL)
    op_register_map_ptr(imap, map);
  return map;
}

op_map op_decl_map_long(op_set from, op_set to, int dim, idx_g_t *imap_g,
                   char const *name) {
  op_map map = decl_map_global(from, to, dim, imap_g, name);
  if (imap_g != NULL)
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
 * Halo list constructors (op_mpi_halo.h)
 *
 * C++ linkage, unlike the rest of this file: they take and return C++ types.
 *******************************************************************************/

extern "C++" {
namespace op::mpi {

HaloList HaloList::from_groups(op_set set, std::vector<int> ranks,
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

HaloList HaloList::from_pairs(op_set set, const int *pairs, int n_ints) {
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
  return from_groups(set, std::move(ranks), std::move(sizes), std::move(list));
}

/* A halo list from what an exchange delivered: one entry per sending rank,
   ranks ascending, each holding what that rank sent. Takes the exchange's
   buffers over rather than copying them. */
static HaloList from_received(op_set set, Received<int> &&got) {
  return HaloList::from_groups(set, std::move(got.ranks), std::move(got.counts), std::move(got.data));
}

HaloList transpose(const HaloList &list, MPI_Comm comm) {
  /* One message per neighbour, each viewing its block of the list in place. */
  std::vector<msg::BlockView<int>> messages;
  messages.reserve(list.ranks_size());
  for (int i = 0; i < list.ranks_size(); i++)
    messages.emplace_back(list.ranks[i], list.list.get() + list.disps[i],
                          (std::size_t)list.sizes[i]);

  return from_received(list.set, sparse::exchange(comm, messages));
}

std::vector<PartRange> part_ranges(MPI_Comm comm) {
  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  std::vector<idx_g_t> mine(OP_set_index), all((std::size_t)comm_size * OP_set_index);
  for (int s = 0; s < OP_set_index; s++)
    mine[s] = OP_set_list[s]->size;
  MPI_Allgather(mine.data(), OP_set_index, get_mpi_type(mine.data()), all.data(), OP_set_index,
                get_mpi_type(all.data()), comm);
  std::vector<PartRange> ranges(OP_set_index);
  for (int s = 0; s < OP_set_index; s++) {
    ranges[s].set = OP_set_list[s];
    ranges[s].start.resize(comm_size + 1);
    for (int r = 0; r < comm_size; r++)
      ranges[s].start[r + 1] = ranges[s].start[r] + all[(std::size_t)r * OP_set_index + s];
  }
  return ranges;
}

void fail(const char *format, ...) {
  va_list args;
  va_start(args, format);
  std::vfprintf(stderr, format, args);
  va_end(args);
  std::fflush(stderr);
  MPI_Abort(OP_MPI_WORLD, 2);
  std::abort(); // MPI_Abort does not return
}

void exchange_rows(MPI_Comm comm, const char *rows, std::size_t row_bytes, const HaloList &exp,
                   const HaloList &imp, char *into, std::size_t gap) {
  /* On the sparse exchange's private communicator, so nothing the caller sends can
     be matched instead, and on a tag its protocol never uses: it probes tags 0 and
     1 from any source and receives tag 2 from named ones. At most one message
     between two ranks per call, received from a named source, so MPI's order
     keeps consecutive calls apart. */
  comm = detail::comm_state(comm).comm;
  constexpr int tag = 3;
  int my_rank;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Datatype row;
  MPI_Type_contiguous((int)row_bytes, MPI_BYTE, &row);
  MPI_Type_commit(&row);

  std::vector<MPI_Request> requests(imp.ranks_size() + exp.ranks_size());
  for (int i = 0; i < imp.ranks_size(); i++)
    MPI_Irecv(into + (imp.disps[i] + (imp.ranks[i] > my_rank ? gap : 0)) * row_bytes, imp.sizes[i], row, imp.ranks[i],
              tag, comm, &requests[i]);

  std::vector<char> packed(row_bytes * exp.size());
  for (int i = 0; i < exp.ranks_size(); i++) {
    for (idx_l_t j = exp.disps[i]; j < exp.disps[i] + exp.sizes[i]; j++)
      memcpy(packed.data() + j * row_bytes, rows + (std::size_t)exp.list[j] * row_bytes, row_bytes);
    MPI_Isend(packed.data() + exp.disps[i] * row_bytes, exp.sizes[i], row, exp.ranks[i], tag, comm,
              &requests[imp.ranks_size() + i]);
  }

  MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  MPI_Type_free(&row);
}

char *migrate_rows(MPI_Comm comm, const char *rows, std::size_t row_bytes, int n_rows, const int *elem_part,
                   int my_rank, const HaloList &exp, const HaloList &imp) {
  const std::size_t kept = std::count(elem_part, elem_part + n_rows, my_rank);
  const auto above = std::upper_bound(imp.ranks.begin(), imp.ranks.end(), my_rank) - imp.ranks.begin();
  const std::size_t below = above < imp.ranks_size() ? imp.disps[above] : imp.size();
  char *out = (char *)xmalloc(row_bytes * (kept + imp.size()));
  char *at = out + below * row_bytes;
  for (int i = 0; i < n_rows; i++)
    if (elem_part[i] == my_rank) {
      memcpy(at, rows + (std::size_t)i * row_bytes, row_bytes);
      at += row_bytes;
    }
  exchange_rows(comm, rows, row_bytes, exp, imp, out, kept);
  return out;
}

}  // namespace op::mpi
}  // extern "C++"



/*******************************************************************************
 * Check whether a map reaches every element of its to-set: each rank tells the
 * owner of every to-set element its entries reach, and the owners count what
 * nobody reached. Expects map entries and set elements in the same global
 * numbering, as before partitioning. Collective over OP_MPI_WORLD.
 *******************************************************************************/

int is_onto_map(op_map map) {
  int my_rank;
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  const PartRange range = op::mpi::part_ranges(OP_MPI_WORLD)[map->to->index];

  std::vector<idx_g_t> reached(map->map_gbl, map->map_gbl + static_cast<std::size_t>(map->from->size) * map->dim);
  std::sort(reached.begin(), reached.end());
  reached.erase(std::unique(reached.begin(), reached.end()), reached.end());
  auto told = op::mpi::sparse::exchange_by(OP_MPI_WORLD, reached, [&](idx_g_t g) {
    int local;
    return range.owner(g, &local);
  });

  std::vector<char> hit(map->to->size, 0);
  for (idx_g_t g : told)
    hit[g - range.start[my_rank]] = 1;
  long long missed = std::count(hit.begin(), hit.end(), 0), missed_anywhere = 0;
  MPI_Allreduce(&missed, &missed_anywhere, 1, MPI_LONG_LONG, MPI_SUM, OP_MPI_WORLD);
  return missed_anywhere == 0;
}

/*******************************************************************************
 * Main MPI halo creation routine
 *******************************************************************************/

/* Where rank sits in a list's ranks, or -1. */
static int rank_slot(const HaloList &h, int rank) {
  auto it = std::lower_bound(h.ranks.begin(), h.ranks.end(), rank);
  return it != h.ranks.end() && *it == rank ? (int)(it - h.ranks.begin()) : -1;
}

/* Where a halo list names element `index` of rank `rank`, as a position in its
   list, or -1. Needs each rank's block sorted, as halo creation builds them. */
static int position_of(const HaloList &h, int rank, int index) {
  const int r = rank_slot(h, rank);
  if (r < 0)
    return -1;
  const idx_l_t *begin = h.list.get() + h.disps[r], *end = begin + h.sizes[r];
  const idx_l_t *at = std::lower_bound(begin, end, index);
  return at != end && *at == index ? (int)(at - h.list.get()) : -1;
}

/* Print, on the root, the average, minimum and maximum over the ranks of a
   per-rank count. */
static void print_spread(const char *label, idx_g_t n, int width) {
  int my_rank, comm_size;
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_WORLD, &comm_size);
  idx_g_t sum = 0, most[2] = {n, -n}, max[2] = {0, 0};
  MPI_Reduce(&n, &sum, 1, get_mpi_type(&n), MPI_SUM, MPI_ROOT, OP_MPI_WORLD);
  MPI_Reduce(most, max, 2, get_mpi_type(&n), MPI_MAX, MPI_ROOT, OP_MPI_WORLD); // -max(-n) is min(n)
  if (my_rank == MPI_ROOT)
    printf("%-19s %*lld %*lld %*lld\n", label, width, (long long)(sum / comm_size), width, (long long)-max[1], width,
           (long long)max[0]);
}

void op_halo_create() {
  // declare timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  double time;
  double max_time;
  op_timers(&cpu_t1, &wall_t1); // timer start for list create

  int my_rank, comm_size;
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_WORLD, &comm_size);

  /* Compute global partition range information for each set*/
  const std::vector<PartRange> part_range = op::mpi::part_ranges(OP_MPI_WORLD);

  // save this partition range information if it is not already saved during
  // a call to some partitioning routine
  if (orig_part_range.empty())
    orig_part_range = part_range;

  OP_set_halos = std::vector<SetHalo>(OP_set_index);

  /*----- STEP 1 - Construct export lists for execute set elements: each element
    whose mapping table entries reach another rank goes to that rank -----*/

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    std::vector<int> pairs;
    for (int m = 0; m < OP_map_index; m++) { // for each mapping table from this set
      op_map map = OP_map_list[m];
      if (map->from != set) continue;
      for (int e = 0; e < set->size; e++)
        for (int j = 0; j < map->dim; j++) {
          int local_index;
          const int part = part_range[map->to->index].owner(map->map_gbl[(size_t)e * map->dim + j], &local_index);
          if (part != my_rank)
            pairs.insert(pairs.end(), {part, e});
        }
    }
    OP_set_halos[set->index].export_exec = HaloList::from_pairs(set, pairs.data(), (int)pairs.size());
  }

  /*---- STEP 2 - construct import lists for mappings and execute sets------*/

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];

    OP_set_halos[set->index].import_exec =
        op::mpi::transpose(OP_set_halos[set->index].export_exec, OP_MPI_WORLD);
  }

  /*--STEP 3 -Exchange mapping table entries using the import/export lists--*/

  for (int m = 0; m < OP_map_index; m++) { // for each maping table
    op_map map = OP_map_list[m];
    const SetHalo &halo = OP_set_halos[map->from->index];

    // append the rows of the from-set's exec halo to the mapping table
    map->map_gbl = (idx_g_t *)xrealloc(
        map->map_gbl, (map->dim * (size_t)(map->from->size + halo.import_exec.size())) * sizeof(idx_g_t));
    exchange_rows(OP_MPI_WORLD, (const char *)map->map_gbl, sizeof(idx_g_t) * map->dim, halo.export_exec,
                  halo.import_exec, (char *)(map->map_gbl + (size_t)map->dim * map->from->size));
  }

  /*-- STEP 4 - Create import lists for non-execute set elements: those the
  mapping table entries, the exec halo's included, reach on another rank but the
  exec halo does not hold --*/

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    const HaloList &exec = OP_set_halos[set->index].import_exec;
    std::vector<int> pairs;
    for (int m = 0; m < OP_map_index; m++) { // for each mapping table to this set
      op_map map = OP_map_list[m];
      if (map->to != set) continue;
      const size_t n = (size_t)(map->from->size + OP_set_halos[map->from->index].import_exec.size()) * map->dim;
      for (size_t k = 0; k < n; k++) {
        int local_index;
        const int part = part_range[set->index].owner(map->map_gbl[k], &local_index);
        if (part != my_rank && position_of(exec, part, local_index) < 0)
          pairs.insert(pairs.end(), {part, local_index});
      }
    }
    OP_set_halos[set->index].import_nonexec = HaloList::from_pairs(set, pairs.data(), (int)pairs.size());
  }

  /*----------- STEP 5 - construct non-execute set export lists -------------*/

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];

    /* The nonexec import list was built unilaterally from the received map
       entries, so its owners do not know about it; sending it back is what
       gives them their export list. */
    OP_set_halos[set->index].export_nonexec =
        op::mpi::transpose(OP_set_halos[set->index].import_nonexec, OP_MPI_WORLD);
  }

  /*-STEP 6 and 7 - Exchange execute, then non-execute, set elements' data
   * using the import/export lists--*/

  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    const SetHalo &halo = OP_set_halos[dat->set->index];
    const std::size_t exec_end = dat->set->size + halo.import_exec.size();

    // append the exec halo, then the non-exec halo, to the data array
    dat->data = (char *)xrealloc(dat->data, (exec_end + halo.import_nonexec.size()) * dat->size);
    exchange_rows(OP_MPI_WORLD, dat->data, dat->size, halo.export_exec, halo.import_exec,
                  dat->data + (std::size_t)dat->set->size * dat->size);
    exchange_rows(OP_MPI_WORLD, dat->data, dat->size, halo.export_nonexec, halo.import_nonexec,
                  dat->data + exec_end * dat->size);
  }

  /*-STEP 8 - Renumber mapping tables into local indices: owned elements first,
   * then the exec halo, then the nonexec halo -*/

  for (int m = 0; m < OP_map_index; m++) { // for each mapping table
    op_map map = OP_map_list[m];
    op_set set = map->to;
    const SetHalo &halo = OP_set_halos[set->index];
    const size_t n = (size_t)(map->from->size + OP_set_halos[map->from->index].import_exec.size()) * map->dim;
    map->map = (int *)xmalloc(n * sizeof(int));
    for (size_t k = 0; k < n; k++) {
      int local_index, at;
      const int part = part_range[set->index].owner(map->map_gbl[k], &local_index);
      if (part == my_rank)
        map->map[k] = local_index;
      else if ((at = position_of(halo.import_exec, part, local_index)) >= 0)
        map->map[k] = set->size + at;
      else if ((at = position_of(halo.import_nonexec, part, local_index)) >= 0)
        map->map[k] = set->size + halo.import_exec.size() + at;
      else
        fail("Error: element %d of set %s, held by rank %d, is not in rank %d's halo (map %s)\n", local_index,
             set->name, part, my_rank, map->name);
    }
    free(map->map_gbl);
    map->map_gbl = NULL;
  }

  // set dirty bits of all data arrays to 0
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    dat->dirtybit = 0;
  }

  /*-STEP 10 - Separate core elements: each set's elements that no exec export
   * list names go first, then the exported ones, both in their current order.
   * Their original global indices (g_index) move with them like their data, so
   * they are saved first: by partitioning, or here, where without it the
   * elements are those declared -*/

  if (OP_part_index != OP_set_index) {
    OP_part_list = (part *)xmalloc(OP_set_index * sizeof(part));
    for (int s = 0; s < OP_set_index; s++) { // for each set
      op_set set = OP_set_list[s];
      idx_g_t *g_index = (idx_g_t *)xmalloc(sizeof(idx_g_t) * set->size);
      int *partition = (int *)xmalloc(sizeof(int) * set->size);
      for (int i = 0; i < set->size; i++) {
        g_index[i] = part_range[set->index].global(my_rank, i);
        partition[i] = my_rank;
      }
      decl_partition(set, g_index, partition);
    }
  }
  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    const HaloList &exec = OP_set_halos[set->index].export_exec;
    std::vector<char> exported(set->size, 0);
    for (int i = 0; i < exec.size(); i++)
      exported[exec.list[i]] = 1;
    std::vector<int> moved_to(set->size);
    int next = 0;
    for (int e = 0; e < set->size; e++)
      if (!exported[e])
        moved_to[e] = next++;
    set->core_size = next;
    for (int e = 0; e < set->size; e++)
      if (exported[e])
        moved_to[e] = next++;
    if (set->core_size != set->size)
      op::mpi::move_owned(set, moved_to);
  }

  /* Step 10 moved owned elements and rewrote the export lists; the import lists
     other ranks hold still name the old positions. */
  op_halo_refresh_imports();

  /*-STEP 11 - exec and nonexec sizes -*/
  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    set->exec_size = OP_set_halos[set->index].import_exec.size();
    set->nonexec_size = OP_set_halos[set->index].import_nonexec.size();
  }

  /*-STEP 12 ---------- Clean up and Compute rough halo size
   * numbers------------*/

  op_timers(&cpu_t2, &wall_t2); // timer stop for list create
  // compute import/export lists creation time
  time = wall_t2 - wall_t1;
  MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);

  // avg/min/max set sizes and exec sizes accross the MPI universe
  const bool root = my_rank == MPI_ROOT;
  for (int s = 0; s < OP_set_index; s++) {
    op_set set = OP_set_list[s];
    const SetHalo &halo = OP_set_halos[set->index];
    if (root)
      printf("Num of %8s (avg | min | max)\n", set->name);
    print_spread("total elems", set->size, 10);
    print_spread("core elems", set->core_size, 10);
    print_spread("exec halo elems", halo.import_exec.size(), 10);
    print_spread("non-exec halo elems", halo.import_nonexec.size(), 10);
    if (root)
      printf("-----------------------------------------------------\n");
  }
  if (root)
    printf("\n\n");

  // avg/min/max number of MPI neighbors per process accross the MPI universe
  for (int s = 0; s < OP_set_index; s++) {
    op_set set = OP_set_list[s];
    const SetHalo &halo = OP_set_halos[set->index];
    if (root)
      printf("MPI neighbors for exchanging %8s (avg | min | max)\n", set->name);
    print_spread("exec halo elems", halo.import_exec.ranks_size(), 4);
    if (root)
      printf("MPI neighbors for exchanging %8s (avg | min | max)\n", set->name);
    print_spread("non-exec halo elems", halo.import_nonexec.ranks_size(), 4);
    if (root)
      printf("-----------------------------------------------------\n");
  }

  // average worst case halo size in Bytes
  idx_g_t tot_halo_size = 0;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    const SetHalo &halo = OP_set_halos[item->dat->set->index];
    tot_halo_size += (halo.import_exec.size() + halo.import_nonexec.size()) * (size_t)item->dat->size;
  }
  idx_g_t avg_halo_size;
  MPI_Reduce(&tot_halo_size, &avg_halo_size, 1, get_mpi_type(&tot_halo_size), MPI_SUM, MPI_ROOT,
             OP_MPI_WORLD);

  // print performance results
  if (root) {
    printf("Max total halo creation time = %lf\n", max_time);
    printf("Average (worst case) Halo size = %lld Bytes\n", (long long)(avg_halo_size / comm_size));
  }
}

/*******************************************************************************
 * Move a set's owned elements to new positions on this rank
 *
 * Element i goes to moved_to[i], a permutation of 0..set->size-1. The rows of
 * the set's dats and of every map from it move with their elements, in place;
 * every map entry naming an owned element is renumbered (a map from the set to
 * itself gets both); and so are the set's export lists, the partial-exchange
 * export lists of the maps onto it, and its g_index. Halo rows and entries
 * naming halo elements stay as they are. Halo creation separates each set's
 * core this way, and op_renumber applies its orderings this way. C++ linkage,
 * as it takes a std::span.
 *******************************************************************************/

extern "C++" {
void op::mpi::move_owned(op_set set, std::span<const int> moved_to) {
  const int n = set->size;
  std::vector<char> moved;
  auto move_rows = [&](char *rows, std::size_t row_bytes) {
    moved.resize((std::size_t)n * row_bytes);
    for (int i = 0; i < n; i++)
      std::copy_n(rows + (std::size_t)i * row_bytes, row_bytes, moved.data() + (std::size_t)moved_to[i] * row_bytes);
    std::copy(moved.begin(), moved.end(), rows);
  };

  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries)
    if (item->dat->set == set && item->dat->data != NULL)
      move_rows(item->dat->data, item->dat->size);
  for (int m = 0; m < OP_map_index; m++) {
    op_map map = OP_map_list[m];
    if (map->from == set)
      move_rows((char *)map->map, map->dim * sizeof(int));
    if (map->to == set) {
      const std::size_t entries =
          (std::size_t)(map->from->size + OP_set_halos[map->from->index].import_exec.size()) * map->dim;
      for (std::size_t k = 0; k < entries; k++)
        if (map->map[k] < n)
          map->map[k] = moved_to[map->map[k]];
    }
  }

  SetHalo &halo = OP_set_halos[set->index];
  std::vector<HaloList *> exports = {&halo.export_exec, &halo.export_nonexec};
  for (int m = 0; m < (int)OP_map_halos.size(); m++)
    if (OP_map_list[m]->to == set)
      exports.push_back(&OP_map_halos[m].export_nonexec);
  for (HaloList *list : exports)
    for (idx_l_t k = 0; k < list->size(); k++)
      list->list[k] = moved_to[list->list[k]];

  move_rows((char *)OP_part_list[set->index]->g_index, sizeof(idx_g_t));
}
}  // extern "C++"

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
      HaloList current = op::mpi::transpose(*exp, OP_MPI_WORLD);
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
        HaloList::from_groups(to, std::move(ranks), std::move(sizes), std::move(indices));

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
    OP_map_halos[m].export_nonexec = op::mpi::from_received(to, std::move(wanted));
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
  int my_rank;
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);

  // Send every element back to the rank that declared it, then sort each rank's
  // elements into declaration order.
  part p = OP_part_list[dat->set->index];
  std::vector<int> home(dat->set->size);
  std::vector<int> pairs;
  for (int i = 0; i < dat->set->size; i++) {
    int local_index;
    home[i] = orig_part_range[dat->set->index].owner(p->g_index[i], &local_index);
    if (home[i] != my_rank)
      pairs.insert(pairs.end(), {home[i], i});
  }
  const HaloList exp = HaloList::from_pairs(dat->set, pairs.data(), (int)pairs.size());
  const HaloList imp = op::mpi::transpose(exp, OP_MPI_WORLD);

  char *moved = migrate_rows(OP_MPI_WORLD, dat->data, dat->size, dat->set->size, home.data(), my_rank, exp, imp);
  idx_g_t *g_index = (idx_g_t *)migrate_rows(OP_MPI_WORLD, (const char *)p->g_index, sizeof(idx_g_t),
                                             dat->set->size, home.data(), my_rank, exp, imp);
  // Every element here now is one this rank declared, so each goes to its index
  // in this rank's block: no sort.
  const PartRange &orig = orig_part_range[dat->set->index];
  const idx_g_t first = orig.start[my_rank];
  const int count = (int)(orig.start[my_rank + 1] - first);
  assert(count == (int)std::count(home.begin(), home.end(), my_rank) + imp.size());
  char *data = (char *)xmalloc((size_t)count * dat->size);
  for (int k = 0; k < count; k++)
    memcpy(data + (size_t)(g_index[k] - first) * dat->size, moved + (size_t)k * dat->size, dat->size);
  op_free(moved);
  op_free(g_index);

  op_dat temp_dat = (op_dat)xmalloc(sizeof(op_dat_core));

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
  int my_rank;
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);

  char *src = (char *)ptr;

  // No partitioning information: data is already in declaration order
  if (orig_part_range.empty() || OP_part_list == NULL) {
    if (local_size != (size_t)dat->set->size) {
      fail("Error: op_mpi_put_data local_size %zu does not match set size %d "
           "for dat %s\n",
           local_size, dat->set->size, dat->name);
    }
    if (local_size > 0)
      memcpy(dat->data, src, local_size * (size_t)dat->size);
    dat->dirtybit = 1;
    dat->dirty_hd = 1;
    return;
  }

  const PartRange &orig = orig_part_range[dat->set->index];
  const size_t orig_size = orig.start[my_rank + 1] - orig.start[my_rank];

  if (local_size != orig_size) {
    fail("Error: op_mpi_put_data local_size %zu does not match original "
         "partition size %zu for dat %s on rank %d\n",
         local_size, orig_size, dat->name, my_rank);
  }

  // For each element held here, the rank that declared it and its index there:
  // those declared here are copied now, the rest are asked of their ranks.
  part p = OP_part_list[dat->set->index];
  std::vector<int> orig_local(dat->set->size), pairs;
  for (int i = 0; i < dat->set->size; i++) {
    const int orig_rank =
        orig.owner(p->g_index[i], &orig_local[i]);
    if (orig_rank == my_rank)
      memcpy(&dat->data[(size_t)dat->size * i], &src[(size_t)dat->size * orig_local[i]], dat->size);
    else
      pairs.insert(pairs.end(), {orig_rank, i});
  }

  /* The import list: elements to fill, by declaring rank. Each of those ranks is
     asked for the elements' indices there, in the import list's own order, so the
     list's ranks and sizes describe the request. */
  const HaloList imp = HaloList::from_pairs(dat->set, pairs.data(), (int)pairs.size());
  std::vector<int> want(imp.size());
  for (int i = 0; i < imp.size(); i++)
    want[i] = orig_local[imp.list[i]];
  std::vector<op::mpi::msg::BlockView<int>> messages;
  messages.reserve(imp.ranks_size());
  for (int i = 0; i < imp.ranks_size(); i++)
    messages.emplace_back(imp.ranks[i], want.data() + imp.disps[i], (std::size_t)imp.sizes[i]);

  // The export list: rows of the user's array each rank asked for.
  const HaloList exp = op::mpi::from_received(dat->set, op::mpi::sparse::exchange(OP_MPI_WORLD, messages));
  for (int i = 0; i < exp.size(); i++)
    if (exp.list[i] < 0 || (size_t)exp.list[i] >= orig_size) {
      fail("Error: op_mpi_put_data original local index %d out of range "
           "(orig_size %zu) for dat %s on rank %d\n",
           exp.list[i], orig_size, dat->name, my_rank);
    }

  // Send them, and scatter what arrives into place.
  std::vector<char> received((size_t)dat->size * imp.size());
  exchange_rows(OP_MPI_WORLD, src, dat->size, exp, imp, received.data());
  for (int i = 0; i < imp.size(); i++)
    memcpy(&dat->data[(size_t)dat->size * imp.list[i]], &received[(size_t)dat->size * i], dat->size);

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

    for (const auto &[name, k] : op_mpi_kernel_map)
      if (k.count > 0)
        printf("%-10s  %6d       %10.4f      %10.4f    \n", name.c_str(), k.count, k.time, k.time / k.count);
    printf("___________________________________________________\n");

    if (my_rank == MPI_ROOT) {
      printf("___________________________________________________\n");
      printf("\nKernel        Count   Max time(sec)   Avg time(sec)  \n");
    }

    for (auto &[name, k] : op_mpi_kernel_map) {
      MPI_Reduce(&k.count, &count, 1, MPI_INT, MPI_MAX, MPI_ROOT, OP_MPI_IO_WORLD);
      MPI_Reduce(&k.time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, MPI_ROOT, OP_MPI_IO_WORLD);
      MPI_Reduce(&k.time, &tot_time, 1, MPI_DOUBLE, MPI_MAX, MPI_ROOT, OP_MPI_IO_WORLD);

      if (my_rank == MPI_ROOT && count > 0) {
        printf("%-10s  %6d       %10.4f      %10.4f    \n", name.c_str(), count, tot_time, avg_time / comm_size);
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
  op_mpi_kernel &kernel = op_mpi_kernel_map[name];
  kernel.count += 1;
  kernel.time += time;
  return &kernel;
}

/*******************************************************************************
 * Routine to exit an op2 mpi application -
 *******************************************************************************/

void op_mpi_exit() {
  op_mpi_halo_exchanges_exit();

  op_mpi_kernel_map.clear();

  // free memory allocated to halos
  op_halo_destroy();
  // free memory used for holding partition information
  op_partition_destroy();
  OP_map_halos = std::vector<MapHalo>();
  op_free(OP_map_partial_exchange);
}

int getSetSizeFromOpArg(op_arg *arg) {
  if (!arg->opt)
    return 0;
  const op_set set = arg->dat->set;
  return set->size + set->exec_size + set->nonexec_size;
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

idx_g_t op_get_size(op_set set) { return op::mpi::sum_over_ranks(set->size, OP_MPI_WORLD); }

idx_g_t op_get_global_set_offset(op_set set) { return op::mpi::sum_below_rank(set->size, OP_MPI_WORLD); }

#ifdef __cplusplus
}
#endif
