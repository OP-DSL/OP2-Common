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
 * op_mpi_part_core.c
 *
 * Implements the OP2 Distributed memory (MPI) Partitioning wrapper routines,
 * data migration and support utility functions
 *
 * written by: Gihan R. Mudalige, (Started 07-04-2011)
 */

// mpi header
#include <mpi.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <tuple>
#include <span>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <numeric>
#include <unistd.h>



#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_mpi_comm.h>
#include <op_util.h>
#include <limits.h>

// ptscotch header
#ifdef HAVE_PTSCOTCH
#include <ptscotch.h>
#endif

// parmetis header
#ifdef HAVE_PARMETIS
#include <parmetis.h>
#if !defined(PARMETIS_VER_4)
typedef int idx_t;
#endif
#endif

// kaHIP header
#ifdef HAVE_KAHIP
#include <parhip_interface.h>
#endif

#ifndef HAVE_PARMETIS
typedef float real_t;
#endif

#include <op_lib_mpi.h>
#include <op_mpi_halo.h>

using op::mpi::exchange_rows;
using op::mpi::fail;
using op::mpi::HaloList;
using op::mpi::migrate_rows;
using op::mpi::PartRange;

// double min/max
#include <float.h>

extern int *OP_map_partial_exchange; // flag for each map ..
// used for checking if partial halo exchanges
// are to be performed

#ifdef HAVE_PARMETIS
static void partition_graph_parmetis(op_map primary_map);
static void partition_geomkway(op_map primary_map, op_dat coords);
#endif
#ifdef HAVE_KAHIP
static void partition_graph_kahip(op_map primary_map);
#endif
#ifdef HAVE_PTSCOTCH
static void partition_graph_ptscotch(op_map primary_map);
#endif


/* Deliberately not `using op::mpi::sparse::exchange`: ADL would find std::exchange too,
   since the arguments are std:: types, and it can win on overload resolution. */

struct Adjacency {
  idx_g_t idx1;
  idx_g_t idx2;
};

struct Fpart {
  idx_g_t idx;
  int target_part;
};

/* A vote for the partition of a "to" set element, by its global index: the
   partition of a "from" element that maps to it. */
struct Vote {
  idx_g_t to;
  int part;
};

/* Where renumber_maps looks an element up by its original global index. A set's
   original indices are 0..n-1, so they are dealt out in equal blocks, one per
   rank: rank r keeps the entries for its block, wherever the elements have
   migrated to. Any rank can work out where to ask, with no table and nothing
   sized by the number of ranks. */
struct Directory {
  idx_g_t q, r;  // the first r blocks hold q + 1 indices, the rest q

  Directory(idx_g_t n, int ranks) : q{n / ranks}, r{n % ranks} {}
  int rank_of(idx_g_t g) const { return g < r * (q + 1) ? g / (q + 1) : r + (g - r * (q + 1)) / q; }
  idx_g_t begin(int rank) const { return rank * q + std::min<idx_g_t>(rank, r); }
};

/* An element's original global index and its index in the current numbering. */
struct Renumbered {
  idx_g_t original;
  idx_g_t current;
};




/*******************************************************************************
 * Initialise partitioning data structures with the current (block)
*  partitioning information
 *******************************************************************************/
static std::vector<PartRange> initialise(int my_rank);

//
// MPI Communicator for partitioning
//

static MPI_Comm OP_PART_WORLD;

/*******************************************************************************
 * Utility function to find the number of times a value appears in an array
 *******************************************************************************/

static int frequencyof(int value, int *array, int size) {
  int frequency = 0;
  for (int i = 0; i < size; i++) {
    if (array[i] == value)
      frequency++;
  }
  return frequency;
}

/*******************************************************************************
 * Utility function to find the mode of a set of numbers in an array
 *******************************************************************************/

static int find_mode(int *array, int size) {
  int count = 0, mode = array[0], current;
  for (int i = 0; i < size; i++) {
    if (i > 0 && array[i] == array[i - 1])
      continue;
    current = frequencyof(array[i], array, size);
    if (count < current) {
      count = current;
      mode = array[i];
    }
  }
  return mode;
}

/*******************************************************************************
 * Utility function to see if a target op_set is held in an op_set array
 *******************************************************************************/

static int compare_all_sets(op_set target_set, op_set other_sets[], int size) {
  for (int i = 0; i < size; i++) {
    if (compare_sets(target_set, other_sets[i]) == 1)
      return i;
  }
  return -1;
}

/*******************************************************************************
 * Routine to force adjacent elements to the same partition
 *******************************************************************************/
static void partition_force(op_set primary_set, op_map map, const std::vector<PartRange> &part_range) {
  if (map->to->index != primary_set->index)
    fail("Error in partition_force: map %s goes to set %s, not to the primary set %s\n", map->name, map->to->name,
         primary_set->name);

  part primary_set_part = OP_part_list[primary_set->index];

  /* These come out in element order with a data-derived destination, so there is
     nothing stable to point at: each message owns its element, and exchange
     groups them by target rank. */
  std::vector<op::mpi::msg::Item<Adjacency>> adjacencies_out;
  for (int i = 0; i < map->from->size; ++i) {
    int local_index;
    int target_part = part_range[primary_set->index].owner(map->map_gbl[i * map->dim], &local_index);

    for (int j = 1; j < map->dim; j++)
      adjacencies_out.emplace_back(
          target_part, Adjacency{map->map_gbl[i * map->dim], map->map_gbl[i * map->dim + j]});
  }

  auto adjacencies = op::mpi::sparse::exchange(OP_MPI_WORLD, adjacencies_out,
                                       op::mpi::Coalesce::yes);

  int global_num_changed;
  do {
    std::vector<op::mpi::msg::Item<Fpart>> fparts_out;
    for (auto& adjacency : adjacencies) {
      int local_idx2;
      int target_part = part_range[primary_set->index].owner(adjacency.idx2, &local_idx2);

      int local_idx1;
      part_range[primary_set->index].owner(adjacency.idx1, &local_idx1);

      fparts_out.emplace_back(target_part,
                              Fpart{adjacency.idx2, primary_set_part->elem_part[local_idx1]});
    }

    auto fparts = op::mpi::sparse::exchange(OP_MPI_WORLD, fparts_out, op::mpi::Coalesce::yes);

    int num_changed = 0;
    for (auto& fpart : fparts) {
      int local_idx;
      part_range[primary_set->index].owner(fpart.idx, &local_idx);
      if (primary_set_part->elem_part[local_idx] == fpart.target_part) continue;

      primary_set_part->elem_part[local_idx] = fpart.target_part;
      ++num_changed;
    }

    MPI_Allreduce(&num_changed, &global_num_changed, 1, get_mpi_type(&num_changed), MPI_SUM, OP_MPI_WORLD);
    op_printf("global_num_changed = %d\n", global_num_changed);
  } while (global_num_changed > 0);
}

/*******************************************************************************
 * Routine to use a partitioned map->to set to partition the map->from set
 *******************************************************************************/

static int partition_from_set(op_map map, int my_rank, const std::vector<PartRange> &part_range) {
  part p_set = OP_part_list[map->to->index];

  // go through the map and build an import list of the non-local "to" elements
  std::vector<int> pairs;
  for (int i = 0; i < map->from->size; i++) {
    for (int j = 0; j < map->dim; j++) {
      int local_index;
      int part = part_range[map->to->index].owner(map->map_gbl[i * map->dim + j], &local_index);
      if (part != my_rank) {
        pairs.push_back(part);
        pairs.push_back(local_index);
      }
    }
  }
  HaloList pi_list = HaloList::from_pairs(map->to, pairs.data(), (int)pairs.size());
  HaloList pe_list = op::mpi::transpose(pi_list, OP_PART_WORLD);

  // fetch the partition of every imported "to" element from its owner
  std::vector<int> imp_part(pi_list.size());
  exchange_rows(OP_PART_WORLD, (const char *)p_set->elem_part, sizeof(int), pe_list, pi_list, (char *)imp_part.data());

  // allocate memory to hold the partition details for the set thats going to be
  // partitioned
  int *partition = (int *)xmalloc(sizeof(int) * map->from->size);

  // go through the mapping table and the imported partition information and
  // partition the "from" set
  std::vector<int> found_parts(map->dim);
  for (int i = 0; i < map->from->size; i++) {
    for (int j = 0; j < map->dim; j++) {
      int local_index;
      int part = part_range[map->to->index].owner(map->map_gbl[i * map->dim + j], &local_index);

      if (part == my_rank)
        found_parts[j] = p_set->elem_part[local_index];
      else // get partition information from imported data
      {
        int r = binary_search(pi_list.ranks.data(), part, 0, pi_list.ranks_size() - 1);
        if (r >= 0) {
          int elem = binary_search(&pi_list.list[pi_list.disps[r]],
                                   local_index, 0, pi_list.sizes[r] - 1);
          if (elem >= 0)
            found_parts[j] = imp_part[pi_list.disps[r] + elem];
          else {
            fail("Element %d not found in partition import list\n",
                 local_index);
          }
        } else {
          fail("Rank %d not found in partition import list\n", part);
        }
      }
    }
    partition[i] = find_mode(found_parts.data(), map->dim);
  }

  OP_part_list[map->from->index]->elem_part = partition;
  OP_part_list[map->from->index]->is_partitioned = 1;
  return 1;
}

/*******************************************************************************
 * Routine to use the partitioned map->from set to partition the map->to set
 *******************************************************************************/

static int partition_to_set(op_map map, int my_rank, const std::vector<PartRange> &part_range) {
  const int *from_part = OP_part_list[map->from->index]->elem_part;
  const PartRange &range = part_range[map->to->index];
  auto owner = [&](const Vote &v) {
    int local_index;
    return range.owner(v.to, &local_index);
  };

  // One vote per mapping table entry, sent to the owner of the element it names.
  // Sorted, each owner's votes are one run, so one message.
  std::vector<Vote> votes((std::size_t)map->from->size * map->dim);
  for (std::size_t k = 0; k < votes.size(); k++)
    votes[k] = {map->map_gbl[k], from_part[k / map->dim]};
  auto by_element = [](const Vote &a, const Vote &b) { return a.to != b.to ? a.to < b.to : a.part < b.part; };
  std::sort(votes.begin(), votes.end(), by_element);
  auto received = op::mpi::sparse::exchange_by(OP_PART_WORLD, votes, owner);
  const std::span<Vote> got{received.data.get(), received.size()};
  std::sort(got.begin(), got.end(), by_element);

  // Each element joins the partition most of its votes name, the lowest on a tie.
  int *partition = (int *)xmalloc(sizeof(int) * map->to->size);
  std::fill(partition, partition + map->to->size, -1);
  const idx_g_t first = range.start[my_rank];
  for (auto v = got.begin(); v != got.end();) {
    const auto element_end = std::find_if(v, got.end(), [&](const Vote &w) { return w.to != v->to; });
    int best = -1;
    std::ptrdiff_t most = 0;
    for (auto p = v; p != element_end;) {
      const auto part_end = std::find_if(p, element_end, [&](const Vote &w) { return w.part != p->part; });
      if (part_end - p > most) {
        most = part_end - p;
        best = p->part;
      }
      p = part_end;
    }
    partition[v->to - first] = best;
    v = element_end;
  }

  // An element nothing voted for means the map is not onto; -1 if so on any rank.
  int ok = std::find(partition, partition + map->to->size, -1) == partition + map->to->size ? 1 : -1;
  if (ok < 0 && OP_diags > 2)
    printf("on rank %d: Map %s is not an on-to mapping from set %s to set %s\n", my_rank, map->name,
           map->from->name, map->to->name);
  int result = 1;
  MPI_Allreduce(&ok, &result, 1, MPI_INT, MPI_MIN, OP_PART_WORLD);

  if (result == 1) {
    OP_part_list[map->to->index]->elem_part = partition;
    OP_part_list[map->to->index]->is_partitioned = 1;
  } else {
    op_free(partition);
  }

  return result;
}

/*******************************************************************************
 * Routine to partition all secondary sets using primary set partition
 *******************************************************************************/

static void partition_all(op_set primary_set, int my_rank) {
  // Compute global partition range information for each set
  const std::vector<PartRange> part_range = op::mpi::part_ranges(OP_PART_WORLD);

  bool force_part_done = false;
  for (int i = 0; i < OP_map_index; i++) {
    if (OP_map_list[i]->force_part && force_part_done) {
      op_printf("Warning: force_part set on multiple maps\n");
    }

    if (OP_map_list[i]->force_part) {
      partition_force(primary_set, OP_map_list[i], part_range);
      force_part_done = true;
    }
  }

  int sets_partitioned = 1;
  int maps_used = 0;

  std::vector<op_set> all_partitioned_sets(OP_set_index);
  std::vector<int> all_used_maps(OP_map_index);
  for (int i = 0; i < OP_map_index; i++) {
    all_used_maps[i] = -1;
  }

  // begin with the partitioned primary set
  all_partitioned_sets[0] = OP_set_list[primary_set->index];

  int error = 0;
  while (sets_partitioned < OP_set_index && error == 0) {
    std::vector<int> cost(OP_map_index);
    for (int i = 0; i < OP_map_index; i++)
      cost[i] = 99;

    // compute a "cost" associated with using each mapping table
    for (int m = 0; m < OP_map_index; m++) {
      op_map map = OP_map_list[m];

      if (linear_search(all_used_maps.data(), map->index, 0, maps_used - 1) <
          0) // if not used before
      {
        part to_set = OP_part_list[map->to->index];
        part from_set = OP_part_list[map->from->index];

        // partitioning a set using a mapping from a partitioned set costs
        // more than partitioning a set using a mapping to a partitioned set
        // i.e. preferance is given to the latter over the former
        if (from_set->is_partitioned == 1 &&
            compare_all_sets(map->from, all_partitioned_sets.data(),
                             sets_partitioned) >= 0)
          cost[map->index] = 2;
        else if (to_set->is_partitioned == 1 &&
                 compare_all_sets(map->to, all_partitioned_sets.data(),
                                  sets_partitioned) >= 0)
          cost[map->index] = (map->dim == 1 ? 0 : 1);
      }
    }

    while (1) {
      int selected = min(cost.data(), OP_map_index);

      if (selected >= 0) {
        op_map map = OP_map_list[selected];

        // partition using this map
        part to_set = OP_part_list[map->to->index];
        part from_set = OP_part_list[map->from->index];

        if (to_set->is_partitioned == 1 && from_set->is_partitioned == 0) {
          if (partition_from_set(map, my_rank, part_range) > 0) {
            all_partitioned_sets[sets_partitioned++] = map->from;
            all_used_maps[maps_used++] = map->index;
            break;
          } else // partitioning unsuccessful with this map- find another map
            cost[selected] = 99;
        } else if (from_set->is_partitioned == 1 &&
                   to_set->is_partitioned == 0) {
          if (partition_to_set(map, my_rank, part_range) > 0) {
            all_partitioned_sets[sets_partitioned++] = map->to;
            all_used_maps[maps_used++] = map->index;
            break;
          } else // partitioning unsuccessful with this map - find another map
            cost[selected] = 99;
        } else {
          cost[selected] = 99;
        }
      } else // partitioning error;
      {
        printf("On rank %d: Partitioning error\n", my_rank);
        error = 1;
        break;
      }
    }
  }

  if (my_rank == MPI_ROOT) {
    int die = 0;
    printf("Sets partitioned = %d\n", sets_partitioned);
    if (sets_partitioned != OP_set_index) {
      for (int s = 0; s < OP_set_index; s++) { // for each set
        op_set set = OP_set_list[s];
        part P = OP_part_list[set->index];
        if (P->is_partitioned != 1) {
          printf("Unable to find mapping between primary set and %s \n",
                 P->set->name);
          if (P->set->size != 0)
            die = 1;
        }
      }
      if (die)
        fail("Partitioning aborted !\n");
    }
  }
}

/*******************************************************************************
 * Renumber every map's entries from original global indices into the current
 * numbering, where rank r's elements of a set are part_range[s].start[r] onwards in
 * g_index order.
 *
 * Request/reply through a Directory, one set at a time: every rank registers
 * the elements it holds with their directory, asks the directories for the
 * elements its maps reach but it does not hold, and each directory answers from
 * its block. Memory and traffic follow what a rank holds and references, never
 * the number of ranks.
 *******************************************************************************/

static void renumber_maps(int my_rank, int comm_size) {
  const std::vector<PartRange> part_range = op::mpi::part_ranges(OP_PART_WORLD);

  for (int s = 0; s < OP_set_index; s++) {
    op_set set = OP_set_list[s];

    // Only a set that some map reaches needs its directory. Every rank declares
    // the same maps, so every rank skips the same sets and the exchanges pair up.
    bool reached = false;
    for (int m = 0; m < OP_map_index && !reached; m++)
      reached = OP_map_list[m]->to == set;
    if (!reached)
      continue;

    const idx_g_t n = part_range[s].size();
    const idx_g_t first = part_range[s].start[my_rank];
    const Directory dir(n, comm_size);
    const idx_g_t block = dir.begin(my_rank);

    // The original indices of the elements held here, ascending since migration.
    const std::span<const idx_g_t> held{OP_part_list[s]->g_index, static_cast<std::size_t>(set->size)};
    auto held_at = [&](idx_g_t g) -> idx_g_t {
      auto it = std::lower_bound(held.begin(), held.end(), g);
      return it != held.end() && *it == g ? it - held.begin() : -1;
    };

    // Register every element held here, with its current index.
    std::vector<Renumbered> mine(held.size());
    for (std::size_t i = 0; i < held.size(); i++)
      mine[i] = {held[i], first + static_cast<idx_g_t>(i)};
    auto registered = op::mpi::sparse::exchange_by(OP_PART_WORLD, mine,
                                                   [&](const Renumbered &e) { return dir.rank_of(e.original); });
    std::vector<idx_g_t> table(dir.begin(my_rank + 1) - block, -1);
    for (const Renumbered &e : registered)
      table[e.original - block] = e.current;

    // Ask for the elements of this set that a map reaches and this rank does not hold.
    std::vector<idx_g_t> wanted;
    for (int m = 0; m < OP_map_index; m++) {
      op_map map = OP_map_list[m];
      if (map->to != set) continue;
      for (std::size_t k = 0; k < static_cast<std::size_t>(map->from->size) * map->dim; k++) {
        const idx_g_t g = map->map_gbl[k];
        if (g < 0 || g >= n) {
          fail("renumber_maps: map %s has entry %lld, outside set %s of %lld elements\n", map->name,
               (long long)g, set->name, (long long)n);
        }
        if (held_at(g) < 0) wanted.push_back(g);
      }
    }
    std::sort(wanted.begin(), wanted.end());
    wanted.erase(std::unique(wanted.begin(), wanted.end()), wanted.end());
    auto asked = op::mpi::sparse::exchange_by(OP_PART_WORLD, wanted, [&](idx_g_t g) { return dir.rank_of(g); });

    // Answer each rank in the order it asked.
    std::vector<idx_g_t> answers(asked.size());
    for (std::size_t k = 0; k < answers.size(); k++) {
      answers[k] = table[asked.data[k] - block];
      if (answers[k] < 0) {
        fail("renumber_maps: element %lld of set %s is held by no rank\n", (long long)asked.data[k], set->name);
      }
    }
    std::vector<op::mpi::msg::BlockView<idx_g_t>> back;
    back.reserve(asked.num_neighbours());
    for (int i = 0; i < asked.num_neighbours(); i++)
      back.emplace_back(asked.ranks[i], answers.data() + asked.disps[i], asked.counts[i]);
    // Grouped by directory rank, ascending, each in the order asked: since the
    // directory rank is monotone in the index, that is the order of wanted.
    auto current = op::mpi::sparse::exchange(OP_PART_WORLD, back);
    assert(current.size() == wanted.size());

    // Rewrite every map onto this set.
    for (int m = 0; m < OP_map_index; m++) {
      op_map map = OP_map_list[m];
      if (map->to != set) continue;
      for (std::size_t k = 0; k < static_cast<std::size_t>(map->from->size) * map->dim; k++) {
        const idx_g_t g = map->map_gbl[k];
        const idx_g_t i = held_at(g);
        map->map_gbl[k] = i >= 0 ? first + i
                                 : current.data[std::lower_bound(wanted.begin(), wanted.end(), g) - wanted.begin()];
      }
    }
  }
}


/* Move every set's elements to the ranks partitioning gave them - each dat on
   the set, each mapping table from it, and g_index. Every rank starts from its
   block of the original global numbering, in order, so migrate_rows' rank order
   leaves every set in original global index order with no sort. Sets are
   independent, so each is done in one pass. */
static void migrate_all(int my_rank) {
  for (int s = 0; s < OP_set_index; s++) {
    op_set set = OP_set_list[s];
    part p = OP_part_list[s];

    // The elements leaving, by destination, and arriving, by source.
    std::vector<int> pairs;
    for (int i = 0; i < set->size; i++)
      if (p->elem_part[i] != my_rank)
        pairs.insert(pairs.end(), {p->elem_part[i], i});
    const HaloList exp = HaloList::from_pairs(set, pairs.data(), (int)pairs.size());
    const HaloList imp = op::mpi::transpose(exp, OP_PART_WORLD);

    op_dat_entry *item;
    TAILQ_FOREACH(item, &OP_dat_list, entries) {
      op_dat dat = item->dat;
      if (compare_sets(dat->set, set) != 1) continue;
      char *moved = migrate_rows(OP_PART_WORLD, dat->data, dat->size, set->size, p->elem_part, my_rank, exp, imp);
      op_free(dat->data);
      dat->data = moved;
    }
    for (int m = 0; m < OP_map_index; m++) {
      op_map map = OP_map_list[m];
      if (compare_sets(map->from, set) != 1) continue;
      char *moved = migrate_rows(OP_PART_WORLD, (const char *)map->map_gbl, sizeof(idx_g_t) * map->dim, set->size,
                                 p->elem_part, my_rank, exp, imp);
      op_free(map->map_gbl);
      map->map_gbl = (idx_g_t *)moved;
    }
    char *moved = migrate_rows(OP_PART_WORLD, (const char *)p->g_index, sizeof(idx_g_t), set->size, p->elem_part,
                               my_rank, exp, imp);
    op_free(p->g_index);
    p->g_index = (idx_g_t *)moved;

    // Every element here is now this rank's.
    const int size = (int)std::count(p->elem_part, p->elem_part + set->size, my_rank) + imp.size();
    op_free(p->elem_part);
    p->elem_part = (int *)xmalloc(sizeof(int) * size);
    std::fill(p->elem_part, p->elem_part + size, my_rank);
    set->size = size;

    // In original global index order already: see migrate_rows.
    assert(std::is_sorted(p->g_index, p->g_index + size));
  }
}

/* A partitioner's output as OP2 keeps it - int, xmalloc'd - checked to name a
   rank for every element of the set. */
template <class T> static int *checked_partition(const T *part, op_set set, int my_rank, int comm_size) {
  int *partition = (int *)xmalloc(sizeof(int) * set->size);
  for (int i = 0; i < set->size; i++) {
    if ((long long)part[i] < 0 || (long long)part[i] >= comm_size) { // T may be unsigned
      fail("Partitioning problem: on rank %d, set %s element %d not assigned a partition\n", my_rank, set->name, i);
    }
    partition[i] = (int)part[i];
  }
  return partition;
}

/* A set's geometry at the block layout, set->size * dim doubles in local order,
   from op_set_coords{,_derived}; empty if none was registered. Derived geometry
   averages coordinates any rank may hold, so the ones held elsewhere are asked of
   their owners, each rank talking only to the ranks it needs. */
static std::vector<double> block_coords(op_set set, const std::vector<PartRange> &part_range, int my_rank, int *dim) {
  const op_dat coords = set->coords;
  if (coords == NULL)
    return {};
  const int d = coords->dim;
  const double *xyz = reinterpret_cast<const double *>(coords->data);
  *dim = d;
  const op_map map = set->coords_map;
  if (map == NULL)
    return std::vector<double>(xyz, xyz + (std::size_t)set->size * d);

  const PartRange &range = part_range[coords->set->index];
  auto owner = [&](idx_g_t g) {
    int local;
    return range.owner(g, &local);
  };
  const std::span<const idx_g_t> entries{map->map_gbl, (std::size_t)set->size * map->dim};
  std::vector<idx_g_t> wanted;
  for (idx_g_t g : entries)
    if (owner(g) != my_rank)
      wanted.push_back(g);
  std::sort(wanted.begin(), wanted.end());
  wanted.erase(std::unique(wanted.begin(), wanted.end()), wanted.end());
  auto asked = op::mpi::sparse::exchange_by(OP_PART_WORLD, wanted, owner);

  // Answer each rank in the order it asked. The owner is monotone in the index,
  // so the answers arrive grouped by owner in the order of wanted.
  std::vector<double> answers(asked.size() * d);
  for (std::size_t k = 0; k < asked.size(); k++)
    std::copy_n(xyz + (asked.data[k] - range.start[my_rank]) * d, d, answers.data() + k * d);
  std::vector<op::mpi::msg::BlockView<double>> back;
  back.reserve(asked.num_neighbours());
  for (int i = 0; i < asked.num_neighbours(); i++)
    back.emplace_back(asked.ranks[i], answers.data() + (std::size_t)asked.disps[i] * d,
                      (std::size_t)asked.counts[i] * d);
  auto got = op::mpi::sparse::exchange(OP_PART_WORLD, back);

  std::vector<double> out((std::size_t)set->size * d, 0.0);
  for (int e = 0; e < set->size; e++) {
    for (int j = 0; j < map->dim; j++) {
      const idx_g_t g = entries[(std::size_t)e * map->dim + j];
      int local;
      const double *x = range.owner(g, &local) == my_rank
                            ? xyz + (std::size_t)local * d
                            : got.data.get() + (std::lower_bound(wanted.begin(), wanted.end(), g) - wanted.begin()) * d;
      for (int c = 0; c < d; c++)
        out[(std::size_t)e * d + c] += x[c];
    }
    for (int c = 0; c < d; c++)
      out[(std::size_t)e * d + c] /= map->dim;
  }
  return out;
}

/* The coordinates a geometric partitioner works from, set->size * dim doubles at
   the block layout: the dat op_partition was given, else the geometry registered
   for the set. Collective over OP_PART_WORLD. */
static std::vector<double> partition_coords(op_set set, op_dat dat, const std::vector<PartRange> &part_range,
                                            int my_rank, int *dim) {
  if (dat == nullptr)
    return block_coords(set, part_range, my_rank, dim);
  if (dat->set != set)
    fail("Coordinates %s are on set %s, not on %s, the set being partitioned\n", dat->name, dat->set->name, set->name);
  const std::size_t width = dat->size / dat->dim;
  if (width != sizeof(double) && width != sizeof(float))
    fail("Coordinates %s hold neither doubles nor floats\n", dat->name);
  *dim = dat->dim;
  std::vector<double> xyz((std::size_t)set->size * dat->dim);
  for (std::size_t i = 0; i < xyz.size(); i++) {
    if (width == sizeof(double)) {
      memcpy(&xyz[i], dat->data + i * width, sizeof(double));
    } else {
      float v;
      memcpy(&v, dat->data + i * width, sizeof v);
      xyz[i] = v;
    }
  }
  return xyz;
}

/* Partition with one partitioner: primary(my_rank, comm_size, part_range) gives
   each element of primary_set a rank, from the block layout, as an xmalloc'd
   array; every other set follows it through the maps, then every element moves
   to its rank and the maps are renumbered. Collective; primary runs on a fresh
   OP_PART_WORLD. */
template <class Primary> static void partition_with(const char *name, op_set primary_set, Primary primary) {
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers(&cpu_t1, &wall_t1);

  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_PART_WORLD);
  MPI_Comm_rank(OP_PART_WORLD, &my_rank);
  MPI_Comm_size(OP_PART_WORLD, &comm_size);

  const std::vector<PartRange> part_range = initialise(my_rank);
  OP_part_list[primary_set->index]->elem_part = primary(my_rank, comm_size, part_range);
  OP_part_list[primary_set->index]->is_partitioned = 1;
  partition_all(primary_set, my_rank);
  migrate_all(my_rank);
  renumber_maps(my_rank, comm_size);

  op_timers(&cpu_t2, &wall_t2);
  double time = wall_t2 - wall_t1, max_time;
  MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_ROOT, OP_PART_WORLD);
  MPI_Comm_free(&OP_PART_WORLD);
  if (my_rank == MPI_ROOT)
    printf("Max total %s partitioning time = %lf\n", name, max_time);
}

/*******************************************************************************
 * Partition with a partition vector from the application: partvec holds the
 * rank of each element of the primary set
 *******************************************************************************/

static void partition_external(op_set primary_set, op_dat partvec) {
  partition_with("external", primary_set, [&](int, int, const std::vector<PartRange> &) {
    int *partition = (int *)xmalloc(sizeof(int) * primary_set->size);
    memcpy(partition, partvec->data, sizeof(int) * primary_set->size);
    return partition;
  });
}

/*******************************************************************************
 * This routine partitions a given set randomly
 *******************************************************************************/

static void partition_random(op_set primary_set) {
  partition_with("random", primary_set, [&](int, int comm_size, const std::vector<PartRange> &) {
    int *partition = (int *)xmalloc(sizeof(int) * primary_set->size);
    for (int i = 0; i < primary_set->size; i++)
      partition[i] = (int)((double)rand() / ((double)RAND_MAX + 1) * comm_size);
    return partition;
  });
}

/*******************************************************************************
 * Routine to revert back to the original partitioning
 *******************************************************************************/

void op_partition_destroy() {
  if (OP_part_list == NULL) // nothing was partitioned, nor were halos created
    return;
  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    op_free(OP_part_list[set->index]->g_index);
    op_free(OP_part_list[set->index]->elem_part);
    op_free(OP_part_list[set->index]);
  }
  op_free(OP_part_list);
  OP_part_list = NULL;
  orig_part_range.clear();
}

#ifdef HAVE_PARMETIS

/* Coordinates as ParMETIS takes them, as real_t. Its coordinate binning
   (IRBinCoordinates) closes the last bin at max * (1 + 2 eps), which is not above
   the maximum when that is zero or negative: the largest coordinate then fits no
   bin, and the search for one runs past ParMETIS' work arrays - heap corruption or
   a hang. A 2D mesh given as 3D with z = 0 is enough. So a dimension whose maximum
   is not positive is shifted to make it 1; the others are passed as they are.
   Collective over OP_PART_WORLD. */
static std::vector<real_t> parmetis_coords(const std::vector<double> &xyz, int dim) {
  std::vector<real_t> out(xyz.begin(), xyz.end());
  std::vector<double> local_max(dim, -std::numeric_limits<double>::infinity()), global_max(dim);
  for (std::size_t i = 0; i < out.size(); i++)
    local_max[i % dim] = std::max(local_max[i % dim], (double)out[i]);
  MPI_Allreduce(local_max.data(), global_max.data(), dim, MPI_DOUBLE, MPI_MAX, OP_PART_WORLD);
  for (std::size_t i = 0; i < out.size(); i++)
    if (global_max[i % dim] <= 0)
      out[i] += (real_t)(1 - global_max[i % dim]);
  return out;
}

/*******************************************************************************
 * Wrapper routine to use ParMETIS_V3_PartGeom() which partitions a set
 * Using its XYZ Geometry Data
 *******************************************************************************/

static void partition_geom(op_set set, op_dat coords) {
  partition_with("geometric", set, [&](int my_rank, int comm_size, const std::vector<PartRange> &part_range) {
    const std::vector<idx_g_t> &start = part_range[set->index].start;
    std::vector<idx_t> vtxdist(start.begin(), start.end()), partition(set->size);
    int dim = 0;
    const std::vector<double> xyz = partition_coords(set, coords, part_range, my_rank, &dim);
    std::vector<real_t> pm_xyz = parmetis_coords(xyz, dim);
    idx_t ndims = dim;
    ParMETIS_V3_PartGeom(vtxdist.data(), &ndims, pm_xyz.data(), partition.data(), &OP_PART_WORLD);
    return checked_partition(partition.data(), set, my_rank, comm_size);
  });
}

#endif

/*******************************************************************************
 * Use OPlus style recursive bisection in the inertial directions
 *******************************************************************************/

/* The primary set's ranks by recursive bisection along the inertial axes, for
   partition_with. */
static int *inertial_partition(op_set set, op_dat coords, int my_rank, int comm_size,
                               const std::vector<PartRange> &part_range) {
  // three doubles per element, z = 0 for 2D coordinates
  int dim = 0;
  const std::vector<double> xyz = partition_coords(set, coords, part_range, my_rank, &dim);
  double *x = (double *)xcalloc((std::size_t)set->size * 3 + 1, sizeof(double));
  for (int i = 0; i < set->size; i++)
    std::copy_n(&xyz[(std::size_t)i * dim], dim, &x[(std::size_t)i * 3]);

  MPI_Comm mpi_comm = OP_PART_WORLD; // halved at each level

  /* - STEP 1 figure out partitioning - */
  const PartRange &range = part_range[set->index];
  idx_g_t global_size = range.size();          // losg
  idx_g_t block_lower = range.start[my_rank];  // losg1
  int block_size = set->size;                  // losgd

  idx_g_t *global_indices = (idx_g_t *)xmalloc((block_size > 0 ? block_size : 1) * sizeof(idx_g_t));
  for (int i = 0; i < block_size; i++)
    global_indices[i] = block_lower + i;
  int nlevel = 0;
  while ((1 << nlevel) < comm_size)
    nlevel++;

  int current_part_size = block_size;       // losl
  idx_g_t current_group_size = global_size; // lopl

  MPI_Request s_request, s_request2;
  MPI_Status s_status, s_status2;
  MPI_Status r_status;

  double *dist;
  double mtx[9];
  double mtx_global[9];
  double q[3], p[3], mue;
  for (int level = 0; level < nlevel; level++) {
    // op_inert begin
    if (comm_size != 1) {
      dist = (double *)xmalloc((current_part_size==0?1:current_part_size) * sizeof(double));
      double x_local = 0.0;
      double y_local = 0.0;
      double z_local = 0.0;
      for (int i = 0; i < current_part_size; i++) {
        x_local += x[3 * i];
        y_local += x[3 * i + 1];
        z_local += x[3 * i + 2];
      }
      double x_global = 0.0;
      double y_global = 0.0;
      double z_global = 0.0;
      MPI_Allreduce(&x_local, &x_global, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
      MPI_Allreduce(&y_local, &y_global, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
      MPI_Allreduce(&z_local, &z_global, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
      x_global /= (double)current_group_size;
      y_global /= (double)current_group_size;
      z_global /= (double)current_group_size;
      // printf("Centre of gravity: %5.14f %5.14f %5.14f\n", x_global, y_global, z_global);
      for (int i = 0; i < 9; i++)
        mtx[i] = 0.0;
      for (int i = 0; i < 9; i++)
        mtx_global[i] = 0.0;
      for (int i = 0; i < current_part_size; i++) {
        mtx[0] += (x[3 * i] - x_global) * (x[3 * i] - x_global); //sxx (1,1)
        mtx[4] += (x[3 * i + 1] - y_global) * (x[3 * i + 1] - y_global); //syy (2,2)
        mtx[8] += (x[3 * i + 2] - z_global) * (x[3 * i + 2] - z_global); //szz (3,3)
        mtx[1] += (x[3 * i] - x_global) * (x[3 * i + 1] - y_global); //sxy (2,1)
        mtx[2] += (x[3 * i] - x_global) * (x[3 * i + 2] - z_global); //sxz (1,3)
        mtx[5] += (x[3 * i + 1] - y_global) * (x[3 * i + 2] - z_global); //syz (3,2)
      }
      mtx[3] = mtx[1]; //sxy (1,2)
      mtx[6] = mtx[2]; //sxz (1,3)
      mtx[7] = mtx[5]; //syz (2,3)
      MPI_Allreduce(mtx, mtx_global, 9, MPI_DOUBLE, MPI_SUM, mpi_comm);

      q[0] = 0.0;
      q[1] = 0.0;
      q[2] = 0.0;
      if (mtx_global[0] >= mtx_global[4] && mtx_global[0] >= mtx_global[8])
        q[0] = 1.0;
      if (mtx_global[4] >= mtx_global[0] && mtx_global[4] >= mtx_global[8])
        q[1] = 1.0;
      if (mtx_global[8] >= mtx_global[0] && mtx_global[8] >= mtx_global[4])
        q[2] = 1.0;

      //op_power

      //normalize q
      double qmod = 0.0;
      for (int i = 0; i < 3; i++) qmod += q[i]*q[i];
      qmod = 1.0/sqrt(qmod);
      for (int i = 0; i < 3; i++) q[i] = qmod*q[i];

      int iter = 0;
      double err = 1.0;
      while (iter++ < 20000 && err > 1e-9) {
        // p = A*x
        for (int i = 0; i < 3; i++) {
          p[i] = 0.0;
          for (int j = 0; j < 3; j++) {
            p[i] += mtx_global[i * 3 + j] * q[j];
          }
        }
        // mue = qT*p
        mue = 0.0;
        for (int i = 0; i < 3; i++)
          mue += q[i] * p[i];
        double sum = 0.0;
        for (int i = 0; i < 3; i++)
          sum += p[i] * p[i];
        sum = 1.0 / sqrt(sum);
        for (int i = 0; i < 3; i++)
          p[i] *= sum;
        err = 0.0;
        for (int i = 0; i < 3; i++)
          err += (q[i] - p[i]) * (q[i] - p[i]);
        err = sqrt(err);
        for (int i = 0; i < 3; i++)
          q[i] = p[i];
      }
      // printf("Converged %d %5.14f (%5.14f %5.14f %5.14f)\n",iter,mue,q[0],q[1],q[2]);
      // op_power end
      for (int i = 0; i < current_part_size; i++)
        dist[i] = x[3 * i] * p[0] + x[3 * i + 1] * p[1] + x[3 * i + 2] * p[2];
      // op_inert end

      // op_sort begin
      double distmin = DBL_MAX;
      double distmax = -1.0*DBL_MAX;
      double distavg = 0.0;
      for (int i = 0; i < current_part_size; i++) {
        distmin = distmin < dist[i] ? distmin : dist[i];
        distmax = distmax > dist[i] ? distmax : dist[i];
        distavg += dist[i];
      }
      double distmin_g = distmin;
      double distmax_g = distmax;
      double distavg_g = 0.0;
      MPI_Allreduce(&distmin, &distmin_g, 1, MPI_DOUBLE, MPI_MIN, mpi_comm);
      MPI_Allreduce(&distmax, &distmax_g, 1, MPI_DOUBLE, MPI_MAX, mpi_comm);
      MPI_Allreduce(&distavg, &distavg_g, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
      distavg_g /= (double)current_group_size;

      double dbnd = distmax_g - distmin_g;
      double dlower = distmin_g;
      double dupper = distmax_g;
      double dsplit = distavg_g;
      idx_g_t nsplit = current_group_size * (comm_size / 2) / comm_size;
      idx_g_t nlower_g = 0;
      while (1) {
        idx_g_t nlower = 0;
        nlower_g = 0;
        for (int i = 0; i < current_part_size; i++)
          nlower += (dist[i] <= dsplit ? 1 : 0);
        MPI_Allreduce(&nlower, &nlower_g, 1, get_mpi_type(&nlower), MPI_SUM, mpi_comm);
        if (nlower_g == nsplit)
          break;
        else if (nlower_g < nsplit) {
          dlower = dsplit;
          dsplit = (dsplit + dupper) / 2.0;
        } else if (nlower_g > nsplit) {
          dupper = dsplit;
          dsplit = (dsplit + dlower) / 2.0;
        }

        if (dupper - dlower < 1e-8 * dbnd)
          break;
      }
      idx_g_t current_group_lower = nlower_g;
      idx_g_t current_group_upper = current_group_size - nlower_g;

      double *x_keep =
          (double *)xmalloc(3 * (current_part_size>0?current_part_size:1) * sizeof(double));
      idx_g_t *idx_gbl_keep = (idx_g_t *)xmalloc(current_part_size * sizeof(idx_g_t));
      double *x_send =
          (double *)xmalloc(3 * (current_part_size>0?current_part_size:1) * sizeof(double));
      idx_g_t *idx_gbl_send = (idx_g_t *)xmalloc((current_part_size > 1 ? current_part_size : 1) * sizeof(idx_g_t));
      int keep_ctr = 0;
      int send_ctr = 0;
      if (my_rank <= comm_size / 2 - 1) {
        for (int i = 0; i < current_part_size; i++) {
          if (dist[i] <= dsplit) {
            idx_gbl_keep[keep_ctr] = global_indices[i];
            for (int j = 0; j < 3; j++)
              x_keep[3 * keep_ctr + j] = x[3 * i + j];
            keep_ctr++;
          } else {
            idx_gbl_send[send_ctr] = global_indices[i];
            for (int j = 0; j < 3; j++)
              x_send[3 * send_ctr + j] = x[3 * i + j];
            send_ctr++;
          }
        }
      } else {
        for (int i = 0; i < current_part_size; i++) {
          if (dist[i] > dsplit) {
            idx_gbl_keep[keep_ctr] = global_indices[i];
            for (int j = 0; j < 3; j++)
              x_keep[3 * keep_ctr + j] = x[3 * i + j];
            keep_ctr++;
          } else {
            idx_gbl_send[send_ctr] = global_indices[i];
            for (int j = 0; j < 3; j++)
              x_send[3 * send_ctr + j] = x[3 * i + j];
            send_ctr++;
          }
        }
      }

      int target_part = comm_size - my_rank - 1;
      if (my_rank == target_part)
        target_part--;
      // Send send buffer size
      MPI_Isend(&send_ctr, 1, get_mpi_type(&send_ctr), target_part, 0, mpi_comm, &s_request);

      // Receive send buffer sizes
      int size_0 = 0;
      int size_1 = 0;
      if (2 * my_rank != (comm_size - 1))
        MPI_Recv(&size_0, 1, get_mpi_type(&size_0), target_part, 0, mpi_comm, &r_status);
      if (2 * (my_rank + 1) == comm_size - 1)
        MPI_Recv(&size_1, 1, get_mpi_type(&size_1), target_part - 1, 0, mpi_comm, &r_status);

      // Allocate for next iteration
      op_free(x);
      op_free(global_indices);
      x = (double *)xrealloc(
          x_keep, (keep_ctr + size_0 + size_1 + 1) * 3 *
                      sizeof(double)); // Implicitly assign x = x_keep
      global_indices = (idx_g_t *)xrealloc(
          idx_gbl_keep,
          (keep_ctr + size_0 + size_1 + 1) *
              sizeof(idx_g_t)); // Implicitly assign global_indices = idx_gbl_keep
      current_part_size = keep_ctr + size_0 + size_1;

      MPI_Wait(&s_request, &s_status);

      MPI_Isend(x_send, 3 * send_ctr, MPI_DOUBLE, target_part, 1, mpi_comm,
                &s_request);
      MPI_Isend(idx_gbl_send, send_ctr, get_mpi_type(idx_gbl_send), target_part, 2, mpi_comm,
                &s_request2);

      // Pack kept - receive 0 - receive 1
      if (2 * my_rank != (comm_size - 1)) {
        MPI_Recv(&x[keep_ctr * 3], size_0 * 3, MPI_DOUBLE, target_part, 1,
                 mpi_comm, &r_status);
        MPI_Recv(&global_indices[keep_ctr], size_0, get_mpi_type(global_indices), target_part, 2,
                 mpi_comm, &r_status);
      }
      if (2 * (my_rank + 1) == comm_size - 1) {
        MPI_Recv(&x[(keep_ctr + size_0) * 3], size_1 * 3, MPI_DOUBLE,
                 target_part - 1, 1, mpi_comm, &r_status);
        MPI_Recv(&global_indices[keep_ctr + size_0], size_1, get_mpi_type(global_indices),
                 target_part - 1, 2, mpi_comm, &r_status);
      }

      // Divide the group in two, the lower half of the ranks and the upper, in order
      const bool lower = my_rank <= comm_size / 2 - 1;
      current_group_size = lower ? current_group_lower : current_group_upper;
      MPI_Comm half;
      MPI_Comm_split(mpi_comm, lower ? 0 : 1, my_rank, &half);
      MPI_Wait(&s_request, &s_status);
      MPI_Wait(&s_request2, &s_status2);
      if (mpi_comm != OP_PART_WORLD)
        MPI_Comm_free(&mpi_comm);
      mpi_comm = half;
      MPI_Comm_rank(mpi_comm, &my_rank);
      MPI_Comm_size(mpi_comm, &comm_size);
      op_free(dist);
      op_free(idx_gbl_send);
      op_free(x_send);
    }
  }
  op_free(x);
  if (mpi_comm != OP_PART_WORLD)
    MPI_Comm_free(&mpi_comm);
  // back to the whole communicator
  MPI_Comm_rank(OP_PART_WORLD, &my_rank);
  MPI_Comm_size(OP_PART_WORLD, &comm_size);

  // Tell each element's declaring rank which rank it ended on. Sorted, the
  // elements for one rank are one run, so one message.
  std::sort(global_indices, global_indices + current_part_size);
  auto ended_on = op::mpi::sparse::exchange_by(
      OP_PART_WORLD, std::span<const idx_g_t>(global_indices, current_part_size), [&](idx_g_t g) {
        int local_index;
        return range.owner(g, &local_index);
      });
  op_free(global_indices);
  if (ended_on.size() != (std::size_t)block_size) {
    fail("Error at rank %d: original(%d) vs. collected(%zu) size mismatch! Aborting...\n", my_rank, block_size,
         ended_on.size());
  }
  int *partition = (int *)xmalloc(sizeof(int) * set->size);
  for (int i = 0; i < ended_on.num_neighbours(); i++)
    for (idx_g_t g : ended_on.from_neighbour(i))
      partition[g - block_lower] = ended_on.ranks[i];
  return partition;
}

static void partition_inertial(op_set set, op_dat coords) {
  partition_with("inertial", set, [&](int my_rank, int comm_size, const std::vector<PartRange> &part_range) {
    return inertial_partition(set, coords, my_rank, comm_size, part_range);
  });
}

/* What a partitioner takes op_partition's dat for. */
enum class DatUse : char {
  none,
  partition, // the partition itself: the rank of each element of the set
  coords,    // coordinates; without a dat, the geometry registered for the set
};

/* The partitioners op_partition can run, by library and routine, and the inputs
   each needs. A library that was not built has no entries. */
struct Partitioner {
  const char *lib, *routine; // routine null: the library has one, and ignores the name
  bool set, map;             // whether it needs op_partition's set and map
  DatUse dat;                // what it takes op_partition's dat for
  int min_dim, max_dim;      // the dimensions that dat may have
  bool partial_halos;        // whether partial halo exchanges may follow it
  void (*run)(op_set, op_map, op_dat);
};

static const Partitioner partitioners[] = {
#ifdef HAVE_KAHIP
    {"KAHIP", "KWAY", false, true, DatUse::none, 0, 0, true,
     [](op_set, op_map m, op_dat) { partition_graph_kahip(m); }},
#endif
#ifdef HAVE_PTSCOTCH
    {"PTSCOTCH", "KWAY", false, true, DatUse::none, 0, 0, true,
     [](op_set, op_map m, op_dat) { partition_graph_ptscotch(m); }},
#endif
#ifdef HAVE_PARMETIS
    {"PARMETIS", "KWAY", false, true, DatUse::none, 0, 0, true,
     [](op_set, op_map m, op_dat) { partition_graph_parmetis(m); }},
    {"PARMETIS", "GEOMKWAY", false, true, DatUse::coords, 1, 3, true,
     [](op_set, op_map m, op_dat d) { partition_geomkway(m, d); }},
    {"PARMETIS", "GEOM", false, false, DatUse::coords, 1, 3, true,
     [](op_set s, op_map, op_dat d) { partition_geom(d != nullptr ? d->set : s, d); }},
#endif
    {"RANDOM", nullptr, true, false, DatUse::none, 0, 0, true,
     [](op_set s, op_map, op_dat) { partition_random(s); }},
    // no partial halos after an external partition, which may leave orphaned elements
    {"EXTERNAL", nullptr, true, false, DatUse::partition, 1, 1, false,
     [](op_set s, op_map, op_dat d) { partition_external(s, d); }},
    {"INERTIAL", nullptr, false, false, DatUse::coords, 2, 3, true,
     [](op_set s, op_map, op_dat d) { partition_inertial(d != nullptr ? d->set : s, d); }},
};

/* The partitioner op_partition's arguments name, if it is built and given what it
   needs; else null, having said why. */
static const Partitioner *select_partitioner(const char *lib, const char *routine, op_set set, op_map map,
                                             op_dat dat) {
  bool lib_built = false;
  for (const Partitioner &p : partitioners) {
    if (strcmp(p.lib, lib) != 0)
      continue;
    lib_built = true;
    if (p.routine != nullptr && strcmp(p.routine, routine) != 0)
      continue;
    op_printf("Selected Partitioning Library : %s\n", lib);
    if (p.routine != nullptr)
      op_printf("Selected Partitioning Routine : %s\n", routine);
    // the set a geometric partitioner partitions, and whose coordinates it needs
    const op_set geo = p.dat != DatUse::coords ? nullptr
                       : p.map                 ? (map != nullptr ? map->to : nullptr)
                       : dat != nullptr        ? dat->set
                                               : set;
    const char *missing = p.set && set == nullptr                                         ? "set"
                          : p.map && map == nullptr                                       ? "map"
                          : p.dat == DatUse::partition && dat == nullptr                  ? "dat"
                          : p.dat != DatUse::none && dat != nullptr && dat->data == nullptr ? "dat"
                          : p.dat == DatUse::coords && geo == nullptr                     ? "set or a dat"
                                                                                          : nullptr;
    if (missing != nullptr) {
      op_printf("Partitioning %s needs a %s, given NULL - UNSUPPORTED Partitioner Specification\n", lib, missing);
      return nullptr;
    }
    if (p.dat == DatUse::coords && dat == nullptr && geo->coords == nullptr) {
      op_printf("Partitioning %s needs coordinates: a dat, or geometry for set %s registered with op_set_coords\n",
                lib, geo->name);
      return nullptr;
    }
    const int dim = dat != nullptr ? dat->dim : p.dat == DatUse::coords ? geo->coords->dim : 0;
    if (p.dat != DatUse::none && (dim < p.min_dim || dim > p.max_dim)) {
      if (p.min_dim == p.max_dim)
        op_printf("Partitioning %s needs a dat of dimension %d, given %d\n", lib, p.min_dim, dim);
      else
        op_printf("Partitioning %s needs coordinates of dimension %d to %d, given %d\n", lib, p.min_dim, p.max_dim,
                  dim);
      return nullptr;
    }
    return &p;
  }
  if (lib_built)
    op_printf("Partitioning Routine : %s UNSUPPORTED\n", routine);
  else if (strcmp(lib, "KAHIP") == 0 || strcmp(lib, "PTSCOTCH") == 0 || strcmp(lib, "PARMETIS") == 0)
    op_printf("OP2 Library Not built with Partitioning Library : %s\n", lib);
  else
    op_printf("Partitioning Library : %s UNSUPPORTED\n", lib);
  return nullptr;
}

/*******************************************************************************
* Toplevel partitioning selection function - also triggers halo creation
*******************************************************************************/
void partition(const char *lib_name, const char *lib_routine, op_set prime_set,
               op_map prime_map, op_dat data) {
  const Partitioner *p = select_partitioner(lib_name ? lib_name : "NULL", lib_routine ? lib_routine : "NULL",
                                            prime_set, prime_map, data);
  if (p != nullptr)
    p->run(prime_set, prime_map, data);
  else
    op_printf("Reverting to trivial block partitioning\n");

  // trigger halo creation routines
  op_halo_create();

  /* Partial halos only after a partitioner that allows them: block, or an external
     partition, may leave orphaned set elements, which make partial halos fail at
     run time. */
  if (p != nullptr && p->partial_halos)
    op_halo_permap_create();
  else
    OP_map_partial_exchange = (int *)xcalloc(OP_map_index, sizeof(int));

#ifdef DEBUG // sanity check: primary map elements whose every entry is in the halo
  if (prime_map != NULL) {
    int ctr = 0;
    for (int i = 0; i < prime_map->from->size; i++) {
      bool orphan = true;
      for (int j = 0; j < prime_map->dim; j++)
        orphan = orphan && prime_map->map[i * prime_map->dim + j] >= prime_map->to->size;
      ctr += orphan;
    }
    printf("Orphan edges: %d\n", ctr);
  }
#endif
  OP_is_partitioned = 1;
}

extern int **OP_map_ptr_list;
/* Called from Fortran by its C name; declared in no header. */
extern "C" void op_partition_ptr(const char *lib_name, const char *lib_routine, op_set prime_set, int *prime_map,
                                 double *coords) {
  // the dat declared from coords, if any; a NULL coords would match any dat declared without data
  op_dat item_dat = NULL;
  if (coords != NULL) {
    op_dat_entry *item;
    TAILQ_FOREACH(item, &OP_dat_list, entries)
      if (item->orig_ptr == coords) {
        item_dat = item->dat;
        break;
      }
    if (item_dat == NULL)
      fail("Error in op_partition: no op_dat was declared from the coordinates at %p\n", (void *)coords);
  }

  op_map item_map = op_search_map_ptr(prime_map);

  if (item_map == NULL)
    fail("Error in op_partition: no op_map was declared from the map at %p\n", (void *)prime_map);

  op_partition(lib_name, lib_routine, prime_set, item_map, item_dat);
}


/*******************************************************************************
 * Initialise partitioning data structures with the current (block)
*  partitioning information
 *******************************************************************************/
static std::vector<PartRange> initialise(int my_rank) {
  // Compute global partition range information for each set, and keep it for
  // reversing the partitioning
  std::vector<PartRange> part_range = op::mpi::part_ranges(OP_PART_WORLD);
  orig_part_range = part_range;

  OP_part_list = (part *)xmalloc(OP_set_index * sizeof(part));
  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    idx_g_t *g_index = (idx_g_t *)xmalloc(sizeof(idx_g_t) * set->size);
    for (int i = 0; i < set->size; i++)
      g_index[i] = part_range[set->index].global(my_rank, i);
    decl_partition(set, g_index, NULL);
  }

  return part_range;
}

/*******************************************************************************
 * Construct the adjacency list of the to-set of the primary_map: each local
 * to-element's neighbours are the to-elements of every from-element mapping to
 * it, itself included, by global index, without repeats, in first-seen order.
 * A from-element whose row reaches another rank's to-element is sent there.
 *******************************************************************************/
static std::vector<std::vector<idx_g_t>> construct_adj_list(op_map primary_map, int my_rank,
                                                            const std::vector<PartRange> &part_range) {
  const int dim = primary_map->dim;
  const PartRange &range = part_range[primary_map->to->index];

  // each from-element goes to every other rank its row reaches (from_pairs drops repeats)
  std::vector<int> pairs;
  for (int e = 0; e < primary_map->from->size; e++) {
    for (int j = 0; j < dim; j++) {
      int local_index;
      int part = range.owner(primary_map->map_gbl[(std::size_t)e * dim + j], &local_index);
      if (part != my_rank) {
        pairs.push_back(part);
        pairs.push_back(e);
      }
    }
  }
  HaloList exp_list = HaloList::from_pairs(primary_map->from, pairs.data(), (int)pairs.size());
  HaloList imp_list = op::mpi::transpose(exp_list, OP_PART_WORLD);

  std::vector<idx_g_t> foreign_maps((std::size_t)dim * imp_list.size());
  exchange_rows(OP_PART_WORLD, (const char *)primary_map->map_gbl, sizeof(idx_g_t) * dim, exp_list, imp_list,
                (char *)foreign_maps.data());

  std::vector<std::vector<idx_g_t>> adj(primary_map->to->size);
  auto add_rows = [&](const idx_g_t *rows, int n_rows) {
    for (int i = 0; i < n_rows; i++) {
      const idx_g_t *row = rows + (std::size_t)i * dim;
      for (int j = 0; j < dim; j++) {
        int local_index;
        if (range.owner(row[j], &local_index) != my_rank)
          continue;
        std::vector<idx_g_t> &neighbours = adj[local_index];
        for (int k = 0; k < dim; k++)
          if (std::find(neighbours.begin(), neighbours.end(), row[k]) == neighbours.end())
            neighbours.push_back(row[k]);
      }
    }
  };
  add_rows(primary_map->map_gbl, primary_map->from->size);
  add_rows(foreign_maps.data(), imp_list.size());
  return adj;
}

#ifdef DEBUG
static inline void check_global_index_int32_range(idx_g_t g_index,
                                                  const op_map primary_map,
                                                  int my_rank) {
  if (g_index < (idx_g_t)INT32_MIN || g_index > (idx_g_t)INT32_MAX) {
    fail("Error: global index out of 32-bit integer range for map %s on rank %d (index=%lld)\n",
         primary_map->name, my_rank, (long long)g_index);
  }
}
#endif

/*******************************************************************************
 * Setup variables for k-way partitioning
 *******************************************************************************/
template <class T>
static std::tuple<T *, T *, T *, T *, T, T, real_t *, real_t *>
setup_part_data(op_map primary_map, int my_rank, int comm_size, std::vector<std::vector<idx_g_t>> adj,
                const std::vector<PartRange> &part_range) {
  T comm_size_pm = comm_size;

  T *vtxdist = (T *)xmalloc(sizeof(T) * (comm_size + 1));
  std::copy(part_range[primary_map->to->index].start.begin(), part_range[primary_map->to->index].start.end(), vtxdist);

  // neighbours sorted, self excluded
  std::size_t n_edges = 0;
  for (const std::vector<idx_g_t> &neighbours : adj)
    n_edges += neighbours.size();
  T *xadj = (T *)xmalloc(sizeof(T) * (primary_map->to->size + 1));
  T *adjncy = (T *)xmalloc(sizeof(T) * n_edges);
  T count = 0;
  for (int i = 0; i < primary_map->to->size; i++) {
    idx_g_t g_index = part_range[primary_map->to->index].global(my_rank, i);
#ifdef DEBUG
    if constexpr (sizeof(T) == sizeof(int)) {
      check_global_index_int32_range(g_index, primary_map, my_rank);
    }
#endif
    if (adj[i].size() < 2) {
      fail("The from set: %s of primary map: %s is not an on to set of to-set: %s\n"
           "Need to select a different primary map\n",
           primary_map->from->name, primary_map->name, primary_map->to->name);
    }

    std::sort(adj[i].begin(), adj[i].end());
    xadj[i] = count;
    for (idx_g_t neighbour : adj[i])
      if (neighbour != g_index)
        adjncy[count++] = (T)neighbour;
  }
  xadj[primary_map->to->size] = count;

  T *partition_pm = (T *)xmalloc(sizeof(T) * primary_map->to->size);
  for (int i = 0; i < primary_map->to->size; i++) {
    partition_pm[i] = -99;
  }

  int *hybrid_flags = (int *)xmalloc(comm_size * sizeof(int));
  MPI_Allgather(&OP_hybrid_gpu, 1, MPI_INT, hybrid_flags, 1, MPI_INT,
                OP_PART_WORLD);
  double total = 0;
  for (int i = 0; i < comm_size; i++)
    total += hybrid_flags[i] == 1 ? OP_hybrid_balance : 1.0;

  T ncon = 1;
  real_t *tpwgts = (real_t *)xmalloc(comm_size * sizeof(real_t) * ncon);
  for (int i = 0; i < comm_size * (int)ncon; i++)
    tpwgts[i] = hybrid_flags[i] == 1 ? OP_hybrid_balance / total : 1.0 / total;

  real_t *ubvec = (real_t *)xmalloc(sizeof(real_t) * ncon);
  *ubvec = 1.05;

  op_free(hybrid_flags);
  return std::make_tuple(vtxdist, xadj, adjncy, partition_pm, comm_size_pm,
                         ncon, tpwgts, ubvec);
}

/*******************************************************************************
 * Generic wrapper for kway-partition functions
 *******************************************************************************/

#ifdef HAVE_PARMETIS
static void perform_kway_partition(idx_t *vtxdist, idx_t *xadj, idx_t *adjncy,
                            idx_t *wgtflag, idx_t *numflag, idx_t *ncon,
                            idx_t *nparts, real_t *tpwgts, real_t *ubvec,
                            idx_t *options, idx_t *edgecut, idx_t *part,
                            MPI_Comm *comm) {
  ParMETIS_V3_PartKway(vtxdist, xadj, adjncy, NULL, NULL, wgtflag, numflag,
                       ncon, nparts, tpwgts, ubvec, options, edgecut, part,
                       comm);
}
#endif

#ifdef HAVE_KAHIP
static void perform_kway_partition(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy,
                            idxtype *, idxtype *, idxtype *, idxtype *nparts,
                            real_t *, real_t *, idxtype *, idxtype *edgecut,
                            idxtype *part, MPI_Comm *comm) {
  double imb = 0.03;
  ParHIPPartitionKWay(vtxdist, xadj, adjncy, NULL, NULL, (int *)nparts, &imb,
                      false, 1, ULTRAFASTMESH, (int *)edgecut, part, comm);
}
#endif


#ifdef HAVE_PTSCOTCH
// Helper function to set up PTScotch data structures
static std::tuple<SCOTCH_Dgraph*, SCOTCH_Num*, SCOTCH_Num*, SCOTCH_Num*>
setup_ptscotch_data(op_map primary_map, int my_rank, std::vector<std::vector<idx_g_t>> adj,
                    const std::vector<PartRange> &part_range) {

    SCOTCH_Dgraph *grafptr = SCOTCH_dgraphAlloc();
    SCOTCH_dgraphInit(grafptr, OP_PART_WORLD);

    SCOTCH_Num baseval = 0;

    // vertex local number - number of vertexes on local mpi rank
    SCOTCH_Num vertlocnbr = primary_map->to->size;

    // vertex local max - put same value as vertlocnbr
    SCOTCH_Num vertlocmax = vertlocnbr;

    SCOTCH_Num *vendloctab = NULL; // not needed
    SCOTCH_Num *veloloctab = NULL; // not needed
    SCOTCH_Num *vlblocltab = NULL; // not needed

    // local vertex adjacency index array, of size (vertlocnbr+1), and the
    // global indices of each vertex's neighbours, self excluded, in the
    // order construct_adj_list found them
    std::size_t n_edges = 0;
    for (const std::vector<idx_g_t> &neighbours : adj)
        n_edges += neighbours.size();
    SCOTCH_Num *vertloctab =
        (SCOTCH_Num *)xmalloc(sizeof(SCOTCH_Num) * (vertlocnbr + 1));
    SCOTCH_Num *edgeloctab = (SCOTCH_Num *)xmalloc(sizeof(SCOTCH_Num) * std::max<std::size_t>(n_edges, 1));
    SCOTCH_Num count = 0;
    for (int i = 0; i < primary_map->to->size; i++) {
        idx_g_t g_index = part_range[primary_map->to->index].global(my_rank, i);
        vertloctab[i] = count;
        for (idx_g_t neighbour : adj[i])
            if (neighbour != g_index)
                edgeloctab[count++] = (SCOTCH_Num)neighbour;

        if (vertloctab[i] == count && primary_map->from != primary_map->to) {
            // Only warn if it's not a self-map and has no non-self neighbors
             printf("Warning: Set element %d on rank %d has no non-self neighbours in map %s for PTScotch\n", (int)g_index, my_rank, primary_map->name);
        }
    }
    vertloctab[primary_map->to->size] = count;

    // local number of arcs (number of edges excluding self-loops)
    SCOTCH_Num edgelocnbr = count;
    SCOTCH_Num edgelocsiz = edgelocnbr;

    SCOTCH_Num *edgegsttab = NULL; // not needed
    SCOTCH_Num *edloloctab = NULL; // not needed


    // build a PT-Scotch graph
    SCOTCH_dgraphBuild(grafptr, baseval, vertlocnbr, vertlocmax, vertloctab,
                        vendloctab, veloloctab, vlblocltab, edgelocnbr, edgelocsiz,
                        edgeloctab, edgegsttab, edloloctab);

    int test = SCOTCH_dgraphCheck(grafptr);
    if (test == 1) {
        fail("PT-Scotch Graph Inconsistent - Aborting\n");
    }

    SCOTCH_Num *partloctab =
        (SCOTCH_Num *)xmalloc(sizeof(SCOTCH_Num) * primary_map->to->size);
    for (SCOTCH_Num i = 0; i < primary_map->to->size; i++) {
        partloctab[i] = -99;
    }

    return std::make_tuple(grafptr, partloctab, vertloctab, edgeloctab);
}

// Helper function to call PTScotch partitioner
static void perform_ptscotch_partition(SCOTCH_Dgraph *grafptr, int comm_size, SCOTCH_Num *partloctab, SCOTCH_Num *vertloctab, SCOTCH_Num *edgeloctab) {
    SCOTCH_Strat straptr;
    SCOTCH_stratInit(&straptr);
    // Optional: Set specific PTScotch strategy here if needed
    // SCOTCH_stratDgraphMapBuild(&straptr, SCOTCH_STRATDEFAULT, comm_size, comm_size, 1.05);

    SCOTCH_dgraphPart(grafptr, comm_size, &straptr, partloctab);
    op_free(vertloctab);
    op_free(edgeloctab);
    SCOTCH_stratExit(&straptr);
    SCOTCH_dgraphExit(grafptr); // Free graph structure inside SCOTCH
    op_free(grafptr); // Free the pointer allocated by SCOTCH_dgraphAlloc
}

#endif // HAVE_PTSCOTCH


/*******************************************************************************
 * Generalized Graph Partitioner (handles ParMETIS, KaHIP, PTScotch)
 *******************************************************************************/
/* The primary set's ranks from a graph partitioner, for partition_with: the
   to-set of primary_map, with an edge between two of its elements when a
   from-element maps to both. T is the partitioner's index type. */
template <class T>
static int *graph_partition(op_map primary_map, const char *partitioner_name, bool geometric, op_dat coords,
                            int my_rank, int comm_size, const std::vector<PartRange> &part_range) {
#ifdef HAVE_PARMETIS
  // Coordinates make the ParMETIS k-way partitioning geometric: PartGeomKway.
  std::vector<real_t> xyz;
  int dim = 0;
  if (geometric) {
    const std::vector<double> c = partition_coords(primary_map->to, coords, part_range, my_rank, &dim);
    xyz = parmetis_coords(c, dim);
  }
#endif

  /*--STEP 1 - Construct adjacency list */
  std::vector<std::vector<idx_g_t>> adj = construct_adj_list(primary_map, my_rank, part_range);

  /*-- STEP 1.5 - Call Partitioner-Specific Setup & Partition */
  if (strcmp(partitioner_name, "PARMETIS") == 0 || strcmp(partitioner_name, "KAHIP") == 0) {
#if defined(HAVE_PARMETIS) || defined(HAVE_KAHIP)
      bool use_kahip = (strcmp(partitioner_name, "KAHIP") == 0);
      T *vtxdist, *xadj, *adjncy, *partition_pm;
      T comm_size_pm, ncon;
      real_t *tpwgts, *ubvec;
      // moved in, so it is freed before the partitioner runs
      std::tie(vtxdist, xadj, adjncy, partition_pm, comm_size_pm, ncon, tpwgts,
              ubvec) = setup_part_data<T>(primary_map, my_rank, comm_size, std::move(adj), part_range);

      T edge_cut = 0;
      T numflag = 0;
      T wgtflag = 0;
      T options[3] = {1, 3, 15};

      if (my_rank == MPI_ROOT) {
          printf("-----------------------------------------------------------\n");
          if (use_kahip) printf("ParHIPPartitionKWay Output\n");
          else if (geometric) printf("ParMETIS_V3_PartGeomKway Output\n");
          else printf("ParMETIS_V3_PartKway Output\n");
          printf("-----------------------------------------------------------\n");
      }

#ifdef HAVE_PARMETIS
      if constexpr (std::is_same_v<T, idx_t>) {
        if (geometric) {
          T ndims = dim;
          ParMETIS_V3_PartGeomKway(vtxdist, xadj, adjncy, NULL, NULL, &wgtflag, &numflag, &ndims, xyz.data(), &ncon,
                                   &comm_size_pm, tpwgts, ubvec, options, &edge_cut, partition_pm, &OP_PART_WORLD);
        }
      }
      if (!use_kahip && !geometric) {
        perform_kway_partition(vtxdist, xadj, adjncy, &wgtflag, &numflag, &ncon,
                              &comm_size_pm, tpwgts, ubvec, options, &edge_cut,
                              partition_pm, &OP_PART_WORLD);
      }
#endif
#ifdef HAVE_KAHIP
      if (use_kahip) {
          perform_kway_partition(vtxdist, xadj, adjncy, &wgtflag, &numflag, &ncon,
                                &comm_size_pm, tpwgts, ubvec, options, &edge_cut,
                                partition_pm, &OP_PART_WORLD);
      }
#endif

      if (my_rank == MPI_ROOT) {
          printf("-----------------------------------------------------------\n");
      }

      op_free(vtxdist); op_free(xadj); op_free(adjncy); op_free(ubvec); op_free(tpwgts);

      int *partition = checked_partition(partition_pm, primary_map->to, my_rank, comm_size);
      op_free(partition_pm);
      return partition;
#else
      // Error: Library not available
      fail("ERROR: %s requested but not compiled.\n", partitioner_name);
#endif
  } else if (strcmp(partitioner_name, "PTSCOTCH") == 0) {
#ifdef HAVE_PTSCOTCH
      SCOTCH_Dgraph *grafptr;
      SCOTCH_Num *partloctab;
      SCOTCH_Num *vertloctab;
      SCOTCH_Num *edgeloctab;
      // moved in, so it is freed before the partitioner runs
      std::tie(grafptr, partloctab, vertloctab, edgeloctab) =
          setup_ptscotch_data(primary_map, my_rank, std::move(adj), part_range);

      if (my_rank == MPI_ROOT) {
          printf("-----------------------------------------------------------\n");
          printf("PT-Scotch Output\n");
          printf("-----------------------------------------------------------\n");
      }
      perform_ptscotch_partition(grafptr, comm_size, partloctab, vertloctab, edgeloctab);
      if (my_rank == MPI_ROOT) {
          printf("-----------------------------------------------------------\n");
      }

      int *partition = checked_partition(partloctab, primary_map->to, my_rank, comm_size);
      op_free(partloctab);
      return partition;
#else
      // Error: Library not available
      fail("ERROR: PTScotch requested but not compiled.\n");
#endif
  } else {
       // Error: Unknown partitioner
      fail("ERROR: Unknown partitioner '%s'\n", partitioner_name);
  }
  return nullptr;
}

template <class T>
static void partition_graph_generic(op_map primary_map, const char *partitioner_name, bool geometric = false,
                                    op_dat coords = nullptr) {
#ifdef DEBUG
  // check if the  primary_map is an on to map from the from-set to the to-set
  if (is_onto_map(primary_map) != 1) {
    fail("Map %s is an not an onto map from set %s to set %s \n", primary_map->name, primary_map->from->name,
         primary_map->to->name);
  }
#endif
  partition_with(partitioner_name, primary_map->to, [&](int my_rank, int comm_size, const std::vector<PartRange> &part_range) {
    return graph_partition<T>(primary_map, partitioner_name, geometric, coords, my_rank, comm_size, part_range);
  });
}

// Specializations/Wrappers to call the generic function
#ifdef HAVE_PARMETIS
static void partition_graph_parmetis(op_map primary_map) {
    partition_graph_generic<idx_t>(primary_map, "PARMETIS");
}

static void partition_geomkway(op_map primary_map, op_dat coords) {
    partition_graph_generic<idx_t>(primary_map, "PARMETIS", true, coords);
}
#endif
#ifdef HAVE_KAHIP
static void partition_graph_kahip(op_map primary_map) {
    partition_graph_generic<idxtype>(primary_map, "KAHIP");
}
#endif
#ifdef HAVE_PTSCOTCH
static void partition_graph_ptscotch(op_map primary_map) {
    // Pass dummy template type idx_t, it's ignored by the PTScotch path
    partition_graph_generic<idx_t>(primary_map, "PTSCOTCH");
}
#endif

