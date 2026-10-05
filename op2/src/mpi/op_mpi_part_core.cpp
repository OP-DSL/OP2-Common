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
using op::mpi::HaloList;
using op::mpi::migrate_rows;

// double min/max
#include <float.h>

extern int *OP_map_partial_exchange; // flag for each map ..
// used for checking if partial halo exchanges
// are to be performed

template <class T>
void op_partition_kway_generic(op_map primary_map, bool use_kahip);

void op_partition_graph_parmetis(op_map primary_map);
void op_partition_graph_kahip(op_map primary_map);
#ifdef HAVE_PTSCOTCH
void op_partition_graph_ptscotch(op_map primary_map);
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


#ifdef __cplusplus
extern "C" {
#endif


/*******************************************************************************
 * Initialise partitioning data structures with the current (block)
*  partitioning information
 *******************************************************************************/
idx_g_t **initialise(int my_rank, int comm_size);

//
// MPI Communicator for partitioning
//

MPI_Comm OP_PART_WORLD;

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
static void partition_force(op_set primary_set, op_map map, int comm_size,
                            idx_g_t **part_range) {
  if (map->to->index != primary_set->index) {
    printf("Error in partition_force: map target set (%d) does not match primary set (%d)\n",
        map->to->index, primary_set->index);
    exit(-1);
  }

  part primary_set_part = OP_part_list[primary_set->index];

  /* These come out in element order with a data-derived destination, so there is
     nothing stable to point at: each message owns its element, and exchange
     groups them by target rank. */
  std::vector<op::mpi::msg::Item<Adjacency>> adjacencies_out;
  for (int i = 0; i < map->from->size; ++i) {
    int local_index;
    int target_part = get_partition(map->map_gbl[i * map->dim], part_range[primary_set->index],
                                    &local_index, comm_size, primary_set);

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
      int target_part = get_partition(adjacency.idx2, part_range[primary_set->index],
                                      &local_idx2, comm_size, primary_set);

      int local_idx1;
      get_partition(adjacency.idx1, part_range[primary_set->index], &local_idx1, comm_size, primary_set);

      fparts_out.emplace_back(target_part,
                              Fpart{adjacency.idx2, primary_set_part->elem_part[local_idx1]});
    }

    auto fparts = op::mpi::sparse::exchange(OP_MPI_WORLD, fparts_out, op::mpi::Coalesce::yes);

    int num_changed = 0;
    for (auto& fpart : fparts) {
      int local_idx;
      get_partition(fpart.idx, part_range[primary_set->index], &local_idx, comm_size, primary_set);
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

static int partition_from_set(op_map map, int my_rank, int comm_size,
                              idx_g_t **part_range) {
  part p_set = OP_part_list[map->to->index];

  // go through the map and build an import list of the non-local "to" elements
  std::vector<int> pairs;
  for (int i = 0; i < map->from->size; i++) {
    for (int j = 0; j < map->dim; j++) {
      int local_index;
      int part = get_partition(map->map_gbl[i * map->dim + j], part_range[map->to->index], &local_index,
                               comm_size, map->to);
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
      int part = get_partition(map->map_gbl[i * map->dim + j], part_range[map->to->index], &local_index,
                               comm_size, map->to);

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
            printf("Element %d not found in partition import list\n",
                   local_index);
            MPI_Abort(OP_PART_WORLD, 2);
          }
        } else {
          printf("Rank %d not found in partition import list\n", part);
          MPI_Abort(OP_PART_WORLD, 2);
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

static int partition_to_set(op_map map, int my_rank, int comm_size,
                            idx_g_t **part_range) {
  const int *from_part = OP_part_list[map->from->index]->elem_part;
  idx_g_t *range = part_range[map->to->index];
  auto owner = [&](const Vote &v) {
    int local_index;
    return get_partition(v.to, range, &local_index, comm_size, map->to);
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
  const idx_g_t first = range[2 * my_rank];
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

static void partition_all(op_set primary_set, int my_rank, int comm_size) {
  // Compute global partition range information for each set
  idx_g_t **part_range = (idx_g_t **)xmalloc(OP_set_index * sizeof(idx_g_t *));
  get_part_range(part_range, my_rank, comm_size, OP_PART_WORLD);

  bool force_part_done = false;
  for (int i = 0; i < OP_map_index; i++) {
    if (OP_map_list[i]->force_part && force_part_done) {
      op_printf("Warning: force_part set on multiple maps\n");
    }

    if (OP_map_list[i]->force_part) {
      partition_force(primary_set, OP_map_list[i], comm_size, part_range);
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
          if (partition_from_set(map, my_rank, comm_size, part_range) > 0) {
            all_partitioned_sets[sets_partitioned++] = map->from;
            all_used_maps[maps_used++] = map->index;
            break;
          } else // partitioning unsuccessful with this map- find another map
            cost[selected] = 99;
        } else if (from_set->is_partitioned == 1 &&
                   to_set->is_partitioned == 0) {
          if (partition_to_set(map, my_rank, comm_size, part_range) > 0) {
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
      if (die) {
        printf("Partitioning aborted !\n");
        MPI_Abort(OP_PART_WORLD, 1);
      }
    }
  }

  for (int i = 0; i < OP_set_index; i++)
    op_free(part_range[i]);
  op_free(part_range);
}

/*******************************************************************************
 * Renumber every map's entries from original global indices into the current
 * numbering, where rank r's elements of a set are part_range[2r] onwards in
 * g_index order.
 *
 * Request/reply through a Directory, one set at a time: every rank registers
 * the elements it holds with their directory, asks the directories for the
 * elements its maps reach but it does not hold, and each directory answers from
 * its block. Memory and traffic follow what a rank holds and references, never
 * the number of ranks.
 *******************************************************************************/

static void renumber_maps(int my_rank, int comm_size) {
  idx_g_t **part_range = (idx_g_t **)xmalloc(OP_set_index * sizeof(idx_g_t *));
  get_part_range(part_range, my_rank, comm_size, OP_PART_WORLD);

  for (int s = 0; s < OP_set_index; s++) {
    op_set set = OP_set_list[s];

    // Only a set that some map reaches needs its directory. Every rank declares
    // the same maps, so every rank skips the same sets and the exchanges pair up.
    bool reached = false;
    for (int m = 0; m < OP_map_index && !reached; m++)
      reached = OP_map_list[m]->to == set;
    if (!reached)
      continue;

    const idx_g_t n = part_range[s][2 * comm_size - 1] + 1;
    const idx_g_t first = part_range[s][2 * my_rank];
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
          printf("renumber_maps: map %s has entry %lld, outside set %s of %lld elements\n", map->name,
                 (long long)g, set->name, (long long)n);
          MPI_Abort(OP_PART_WORLD, 2);
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
        printf("renumber_maps: element %lld of set %s is held by no rank\n", (long long)asked.data[k], set->name);
        MPI_Abort(OP_PART_WORLD, 2);
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

  for (int i = 0; i < OP_set_index; i++)
    op_free(part_range[i]);
  op_free(part_range);
}


/* Move every set's elements to the ranks partitioning gave them - each dat on
   the set, each mapping table from it, and g_index - then sort them all by
   original global index. Sets are independent, so each is done in one pass. */
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

    // Sort the set's arrays by original global index.
    if (size == 0) continue;
    idx_g_t *permutation = (idx_g_t *)xmalloc(sizeof(idx_g_t) * size);
    std::copy(p->g_index, p->g_index + size, permutation);
    op_sort_get_permutation(permutation, size);
    TAILQ_FOREACH(item, &OP_dat_list, entries)
      if (compare_sets(item->dat->set, set) == 1)
        op_reorder_data(permutation, item->dat->data, size, item->dat->size);
    for (int m = 0; m < OP_map_index; m++)
      if (compare_sets(OP_map_list[m]->from, set) == 1)
        op_reorder_data(permutation, (char *)OP_map_list[m]->map_gbl, size, OP_map_list[m]->dim * sizeof(idx_g_t));
    op_reorder_data(permutation, (char *)p->g_index, size, sizeof(idx_g_t));
    op_free(permutation);
  }
}

/*****************************************************************************************************************************************
 * This routine partitions based on information contained in an op_dat called
 *partvecXXXX (number of total partitions, padded with 0s)
 *****************************************************************************************************************************************/

void op_partition_external(op_set primary_set, op_dat partvec) {
  // declare timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  double time;
  double max_time;

  op_timers(&cpu_t1, &wall_t1); // timer start for partitioning

  // create new communicator for partitioning
  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_PART_WORLD);
  MPI_Comm_rank(OP_PART_WORLD, &my_rank);
  MPI_Comm_size(OP_PART_WORLD, &comm_size);

  /*--STEP 0 - initialise partitioning data stauctures with the current (block)
    partitioning information */

  idx_g_t **part_range = initialise(my_rank, comm_size);

  int *partition = (int *)xmalloc(sizeof(int) * primary_set->size);
  memcpy(partition, partvec->data, sizeof(int) * primary_set->size);

  // initialise primary set as partitioned
  OP_part_list[primary_set->index]->elem_part = partition;
  OP_part_list[primary_set->index]->is_partitioned = 1;

  // free part range
  for (int i = 0; i < OP_set_index; i++)
    op_free(part_range[i]);
  op_free(part_range);

  /*-STEP 2 - Partition all other sets,migrate data and renumber mapping
   * tables-*/

  // partition all other sets
  partition_all(primary_set, my_rank, comm_size);

  // migrate data, sort elements
  migrate_all(my_rank);

  // renumber mapping tables
  renumber_maps(my_rank, comm_size);

  op_timers(&cpu_t2, &wall_t2); // timer stop for partitioning

  // print time for partitioning
  time = wall_t2 - wall_t1;
  MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_ROOT, OP_PART_WORLD);
  MPI_Comm_free(&OP_PART_WORLD);
  if (my_rank == MPI_ROOT)
    printf("Max total random partitioning time = %lf\n", max_time);
}

/*******************************************************************************
 * This routine partitions a given set randomly
 *******************************************************************************/

void op_partition_random(op_set primary_set) {
  // declare timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  double time;
  double max_time;

  op_timers(&cpu_t1, &wall_t1); // timer start for partitioning

  // create new communicator for partitioning
  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_PART_WORLD);
  MPI_Comm_rank(OP_PART_WORLD, &my_rank);
  MPI_Comm_size(OP_PART_WORLD, &comm_size);

  /*--STEP 0 - initialise partitioning data stauctures with the current (block)
    partitioning information */

  idx_g_t **part_range = initialise(my_rank, comm_size);

  /*-----STEP 1 - Partition Primary set using a random number generator
   * --------*/

  int *partition = (int *)xmalloc(sizeof(int) * primary_set->size);
  // printf("RAND_MAX = %d",RAND_MAX);
  for (int i = 0; i < primary_set->size; i++) {
    // not sure if this is the best way to generate the required random number
    partition[i] = // rand()%comm_size;
        (int)((double)rand() / ((double)RAND_MAX + 1) * comm_size);
  }

  // initialise primary set as partitioned
  OP_part_list[primary_set->index]->elem_part = partition;
  OP_part_list[primary_set->index]->is_partitioned = 1;

  // free part range
  for (int i = 0; i < OP_set_index; i++)
    op_free(part_range[i]);
  op_free(part_range);

  /*-STEP 2 - Partition all other sets,migrate data and renumber mapping
   * tables-*/

  // partition all other sets
  partition_all(primary_set, my_rank, comm_size);

  // migrate data, sort elements
  migrate_all(my_rank);

  // renumber mapping tables
  renumber_maps(my_rank, comm_size);

  op_timers(&cpu_t2, &wall_t2); // timer stop for partitioning

  // printf time for partitioning
  time = wall_t2 - wall_t1;
  MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_ROOT, OP_PART_WORLD);
  MPI_Comm_free(&OP_PART_WORLD);
  if (my_rank == MPI_ROOT)
    printf("Max total random partitioning time = %lf\n", max_time);
}

/*******************************************************************************
 * Routine to revert back to the original partitioning
 *******************************************************************************/

void op_partition_destroy() {
  // destroy OP_part_list[]
  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    op_free(OP_part_list[set->index]->g_index);
    op_free(OP_part_list[set->index]->elem_part);
    op_free(OP_part_list[set->index]);
  }
  op_free(OP_part_list);
  for (int i = 0; i < OP_set_index; i++)
    op_free(orig_part_range[i]);
  op_free(orig_part_range);
}

#ifdef HAVE_PARMETIS

extern "C++" {

/* A coordinates dat as ParMETIS takes it: real_t, one row of 1 to 3 values per
   element, from a dat of doubles or floats. */
static std::vector<real_t> parmetis_coordinates(op_dat coords) {
  if (coords->dim < 1 || coords->dim > 3) {
    printf("Dimensions of Coordinate array not one of 3D,2D or 1D\n");
    printf("Not supported by ParMetis - Indicate correct coordinates array\n");
    MPI_Abort(OP_PART_WORLD, 1);
  }
  const std::size_t width = coords->size / coords->dim;
  if (width != sizeof(double) && width != sizeof(float)) {
    printf("Coordinates %s hold neither doubles nor floats\n", coords->name);
    MPI_Abort(OP_PART_WORLD, 1);
  }
  std::vector<real_t> xyz((std::size_t)coords->set->size * coords->dim);
  for (std::size_t i = 0; i < xyz.size(); i++) {
    const char *at = coords->data + i * width;
    if (width == sizeof(double)) {
      double v;
      memcpy(&v, at, sizeof v);
      xyz[i] = (real_t)v;
    } else {
      float v;
      memcpy(&v, at, sizeof v);
      xyz[i] = (real_t)v;
    }
  }
  return xyz;
}

}  // extern "C++"

/*******************************************************************************
 * Wrapper routine to use ParMETIS_V3_PartGeom() which partitions a set
 * Using its XYZ Geometry Data
 *******************************************************************************/

void op_partition_geom(op_dat coords) {
  // declare timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  double time;
  double max_time;

  op_timers(&cpu_t1, &wall_t1); // timer start for partitioning

  // create new communicator for partitioning
  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_PART_WORLD);
  MPI_Comm_rank(OP_PART_WORLD, &my_rank);
  MPI_Comm_size(OP_PART_WORLD, &comm_size);

  /*--STEP 0 - initialise partitioning data stauctures with the current (block)
    partitioning information */

  idx_g_t **part_range = initialise(my_rank, comm_size);

  /*--- STEP 1 - Partition primary set using its coordinates (1D,2D or 3D)
   * -----*/

  // Setup data structures for ParMetis PartGeom
  idx_t *vtxdist = (idx_t *)xmalloc(sizeof(idx_t) * (comm_size + 1));
  idx_t *partition = (idx_t *)xmalloc(sizeof(idx_t) * coords->set->size);

  idx_t ndims = coords->dim;
  std::vector<real_t> xyz = parmetis_coordinates(coords);

  for (int i = 0; i < comm_size; i++) {
    vtxdist[i] = part_range[coords->set->index][2 * i];
  }
  vtxdist[comm_size] =
      part_range[coords->set->index][2 * (comm_size - 1) + 1] + 1;

  // use xyz coordinates to feed into ParMETIS_V3_PartGeom
  ParMETIS_V3_PartGeom(vtxdist, &ndims, xyz.data(), partition, &OP_PART_WORLD);
  op_free(vtxdist);

  // free part range
  for (int i = 0; i < OP_set_index; i++)
    op_free(part_range[i]);
  op_free(part_range);

  // sanity check to see if all elements were partitioned
  for (idx_g_t i = 0; i < coords->set->size; i++) {
    if (partition[i] < 0) {
      printf("Partitioning problem: on rank %d, set %s element %lld not assigned "
             "a partition\n",
             my_rank, coords->name, i);
      MPI_Abort(OP_PART_WORLD, 2);
    }
  }

  // initialise primary set as partitioned: copied, not cast, since ParMETIS'
  // idx_t can be 64 bits wide
  int *elem_part = (int *)xmalloc(sizeof(int) * coords->set->size);
  for (int i = 0; i < coords->set->size; i++)
    elem_part[i] = (int)partition[i];
  op_free(partition);
  OP_part_list[coords->set->index]->elem_part = elem_part;
  OP_part_list[coords->set->index]->is_partitioned = 1;

  /*-STEP 2 - Partition all other sets,migrate data and renumber mapping
   * tables-*/

  // partition all other sets
  partition_all(coords->set, my_rank, comm_size);

  // migrate data, sort elements
  migrate_all(my_rank);

  // renumber mapping tables
  renumber_maps(my_rank, comm_size);

  op_timers(&cpu_t2, &wall_t2); // timer stop for partitioning

  // printf time for partitioning
  time = wall_t2 - wall_t1;
  MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_ROOT, OP_PART_WORLD);
  MPI_Comm_free(&OP_PART_WORLD);
  if (my_rank == MPI_ROOT)
    printf("Max total geometric partitioning time = %lf\n", max_time);
}

#endif

/*******************************************************************************
 * Use OPlus style recursive bisection in the inertial directions
 *******************************************************************************/

void op_partition_inertial(op_dat x_dat) {
  // declare timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  double time;
  double max_time;
  double *x = (double *)xmalloc(x_dat->set->size * x_dat->dim * sizeof(double));
  memcpy(x, x_dat->data, x_dat->set->size * x_dat->dim * sizeof(double));

  op_timers(&cpu_t1, &wall_t1); // timer start for partitioning

  // create new communicator for partitioning
  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_PART_WORLD);
  MPI_Comm_rank(OP_PART_WORLD, &my_rank);
  MPI_Comm_size(OP_PART_WORLD, &comm_size);

  MPI_Comm mpi_comm = OP_PART_WORLD;
  MPI_Group current_group;
  MPI_Comm_group(mpi_comm, &current_group);
  /*--STEP 0 - initialise partitioning data stauctures with the current (block)
    partitioning information */

  // Compute global partition range information for each set
  idx_g_t **part_range = initialise(my_rank, comm_size);

  /* - STEP 1 figure out partitioning - */
  int global_size =
      part_range[x_dat->set->index][2 * (comm_size - 1) + 1] + 1;   // losg
  int block_lower = part_range[x_dat->set->index][2 * my_rank];     // losg1
  int block_upper = part_range[x_dat->set->index][2 * my_rank + 1]; // losg2
  int block_size = block_upper - block_lower + 1;                   // losgd

  int *global_indices = (int *)xmalloc((block_size>0?block_size:1) * sizeof(int));
  for (int i = 0; i < block_size; i++)
    global_indices[i] = block_lower + i;
  int nlevel = 0;
  while ((1 << nlevel) < comm_size)
    nlevel++;

  int current_part_size = block_size;   // losl
  int current_group_size = global_size; // lopl

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
      long nsplit = ((long)current_group_size * (long)(comm_size / 2)) / (long)comm_size;
      int nlower_g = 0;
      while (1) {
        int nlower = 0;
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
      int current_group_lower = nlower_g;
      int current_group_upper = current_group_size - nlower_g;

      double *x_keep =
          (double *)xmalloc(3 * (current_part_size>0?current_part_size:1) * sizeof(double));
      int *idx_gbl_keep = (int *)xmalloc(current_part_size * sizeof(int));
      double *x_send =
          (double *)xmalloc(3 * (current_part_size>0?current_part_size:1) * sizeof(double));
      int *idx_gbl_send = (int *)xmalloc((current_part_size>1?current_part_size:1) * sizeof(int));
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
      global_indices = (int *)xrealloc(
          idx_gbl_keep,
          (keep_ctr + size_0 + size_1 + 1) *
              sizeof(int)); // Implicitly assign global_indices = idx_gbl_keep
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

      // Divide group in two
      int *processes_lower = (int *)xmalloc(comm_size / 2 * sizeof(int));
      int *processes_upper =
          (int *)xmalloc((comm_size - comm_size / 2) * sizeof(int));
      for (int i = 0; i < comm_size / 2; i++)
        processes_lower[i] = i;
      for (int i = 0; i < comm_size - comm_size / 2; i++)
        processes_upper[i] = comm_size / 2 + i;

      MPI_Group lower_group, upper_group;
      MPI_Group_incl(current_group, comm_size / 2, processes_lower,
                     &lower_group);
      MPI_Group_incl(current_group, comm_size - comm_size / 2, processes_upper,
                     &upper_group);
      MPI_Comm lower_comm, upper_comm;
      MPI_Comm_create(mpi_comm, lower_group, &lower_comm);
      MPI_Comm_create(mpi_comm, upper_group, &upper_comm);

      // Join one of the groups
      if (my_rank <= comm_size / 2 - 1) {
        current_group_size = current_group_lower;
        current_group = lower_group;
        mpi_comm = lower_comm;
      } else {
        current_group_size = current_group_upper;
        current_group = upper_group;
        mpi_comm = upper_comm;
      }
      MPI_Comm_rank(mpi_comm, &my_rank);
      MPI_Comm_size(mpi_comm, &comm_size);
      // free stuff
      op_free(dist);
      op_free(processes_lower);
      op_free(processes_upper);
      MPI_Wait(&s_request, &s_status);
      MPI_Wait(&s_request2, &s_status2);
      op_free(idx_gbl_send);
      op_free(x_send);
    }
  }
  op_free(x);
  op_sort(global_indices, current_part_size);
  MPI_Comm_dup(OP_MPI_WORLD, &OP_PART_WORLD);
  MPI_Comm_rank(OP_PART_WORLD, &my_rank);
  MPI_Comm_size(OP_PART_WORLD, &comm_size);
  // start binning (global indices -> processes)
  int *sizes = (int *)xcalloc(comm_size, sizeof(int));
  int target = 0;
  for (int i = 0; i < current_part_size; i++) {
    while (
        !(orig_part_range[x_dat->set->index][2 * target] <= global_indices[i] &&
          orig_part_range[x_dat->set->index][2 * target + 1] >=
              global_indices[i]))
      target++;
    sizes[target]++;
  }
  int *sizes_recv = (int *)xcalloc(comm_size, sizeof(int));
  MPI_Alltoall(sizes, 1, get_mpi_type(sizes), sizes_recv, 1, get_mpi_type(sizes_recv), OP_PART_WORLD);

  // Sanity check
  int total_size = 0;
  for (int i = 0; i < comm_size; i++)
    total_size += sizes_recv[i];
  if (total_size != block_size) {
    printf("Error at rank %d: original(%d) vs. collected(%d) size mismatch! "
           "Aborting...\n",
           my_rank, block_size, total_size);
    MPI_Abort(OP_PART_WORLD, 2);
  }

  // How many partitions we are sending to/receiving from
  int send_count = 0;
  int recv_count = 0;
  for (int i = 0; i < comm_size; i++) {
    if (sizes[i])
      send_count++;
    if (sizes_recv[i])
      recv_count++;
  }

  // Send
  MPI_Request *send_requests =
      (MPI_Request *)xmalloc(send_count * sizeof(MPI_Request));
  MPI_Request *recv_requests =
      (MPI_Request *)xmalloc(recv_count * sizeof(MPI_Request));
  MPI_Status *send_statuses =
      (MPI_Status *)xmalloc(send_count * sizeof(MPI_Status));
  MPI_Status *recv_statuses =
      (MPI_Status *)xmalloc(recv_count * sizeof(MPI_Status));
  int *global_indices_recv = (int *)xmalloc(total_size * sizeof(int));
  send_count = 0;
  recv_count = 0;
  int send_offset = 0;
  int recv_offset = 0;
  for (int i = 0; i < comm_size; i++) {
    if (sizes[i]) {
      MPI_Isend(&global_indices[send_offset], sizes[i], get_mpi_type(global_indices), i, 0,
                OP_PART_WORLD, &send_requests[send_count]);
      send_offset += sizes[i];
      send_count++;
    }
    if (sizes_recv[i]) {
      MPI_Irecv(&global_indices_recv[recv_offset], sizes_recv[i], get_mpi_type(global_indices_recv), i, 0,
                OP_PART_WORLD, &recv_requests[recv_count]);
      recv_offset += sizes_recv[i];
      recv_count++;
    }
  }
  MPI_Waitall(recv_count, recv_requests, recv_statuses);
  int *partition = (int *)xmalloc(sizeof(int) * x_dat->set->size);
  recv_count = 0;
  recv_offset = 0;
  for (int i = 0; i < comm_size; i++) {
    if (sizes_recv[i]) {
      for (int j = recv_offset; j < recv_offset + sizes_recv[i]; j++)
        partition[global_indices_recv[j] - block_lower] = i;
      recv_offset += sizes_recv[i];
      recv_count++;
    }
  }
  MPI_Waitall(send_count, send_requests, send_statuses);
  op_free(global_indices_recv);
  op_free(global_indices);
  // Debugging: print partvecs to file
  /*FILE *file;
  char fname[64];
  sprintf(fname,"partvec_op2_%04d",my_rank);
  file = fopen(fname,"w");
  for (int i = 0; i < x_dat->set->size; i++)
    fprintf(file,"%06d\n",partition[i]+1);
  fclose(file);*/

  // initialise primary set as partitioned
  OP_part_list[x_dat->set->index]->elem_part = (int *)partition;
  OP_part_list[x_dat->set->index]->is_partitioned = 1;

  // free part range
  for (int i = 0; i < OP_set_index; i++)
    op_free(part_range[i]);
  op_free(part_range);

  /*-STEP 2 - Partition all other sets,migrate data and renumber mapping
   * tables-*/

  // partition all other sets
  partition_all(x_dat->set, my_rank, comm_size);

  // migrate data, sort elements
  migrate_all(my_rank);

  // renumber mapping tables
  renumber_maps(my_rank, comm_size);

  op_timers(&cpu_t2, &wall_t2); // timer stop for partitioning
  // printf time for partitioning
  time = wall_t2 - wall_t1;
  MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_ROOT, OP_PART_WORLD);
  MPI_Comm_free(&OP_PART_WORLD);
  if (my_rank == MPI_ROOT)
    printf("Max total inertial partitioning time = %lf\n", max_time);
}

/*******************************************************************************
* Toplevel partitioning selection function - also triggers halo creation
*******************************************************************************/
void partition(const char *lib_name, const char *lib_routine, op_set prime_set,
               op_map prime_map, op_dat data) {
#if !defined(HAVE_PTSCOTCH) && !defined(HAVE_PARMETIS)
  /* Suppress warning */
  (void)lib_routine;
  (void)prime_map;
#endif

  int partial_halo_flag =
      1; // flag to indicate that partial halos should be created
  // default is 1, but if a proper partitioning is not done
  // i.e. with ParMetis or PTScotch then we will have orphen
  // set elements which will result in a runtime error
  // when using partial halos

  /*initial error checks for NULL variables*/
  if (lib_name == NULL)
    lib_name = "NULL";
  if (lib_routine == NULL)
    lib_routine = "NULL";

  if (strcmp(lib_name, "KAHIP") == 0) {
#ifdef HAVE_KAHIP
    op_printf("Selected Partitioning Routine : %s\n", lib_routine);
    if (strcmp(lib_routine, "KWAY") == 0) {
      op_printf("Selected Partitioning Routine : %s\n", lib_routine);
      if (prime_map != NULL)
        op_partition_graph_kahip(prime_map); // use kahip k-way partitioning
      else {
        op_printf("Partitioning prime_map : NULL UNSUPPORTED\n");
        op_printf("Reverting to trivial block partitioning\n");
        partial_halo_flag = 0;
      }
    } else {
      op_printf("Partitioning Routine : %s UNSUPPORTED\n", lib_routine);
      op_printf("Reverting to trivial block partitioning\n");
      partial_halo_flag = 0;
    }
#else
    /*  Suppress warning */
    (void)data;
    op_printf("OP2 Library Not built with Partitioning Library : %s\n",
              lib_name);
    op_printf("Ignoring input routine : %s\n", lib_routine);
    if (prime_set != NULL)
      op_printf("Ignoring input set : %s\n", prime_set->name);
    if (prime_map != NULL)
      op_printf("Ignoring input mapping : %s\n", prime_map->name);
    if (data != NULL)
      op_printf("Ignoring input coordinates : %s\n", data->name);
    op_printf("Reverting to trivial block partitioning\n");
    partial_halo_flag = 0;
#endif
  } else if (strcmp(lib_name, "PTSCOTCH") == 0) {
#ifdef HAVE_PTSCOTCH
    op_printf("Selected Partitioning Library : %s\n", lib_name);
    if (strcmp(lib_routine, "KWAY") == 0) {
      op_printf("Selected Partitioning Routine : %s\n", lib_routine);
      if (prime_map != NULL)
        op_partition_graph_ptscotch(prime_map); // use ptscotch kaway partitioning
      else {
        op_printf("Partitioning prime_map : NULL UNSUPPORTED\n");
        op_printf("Reverting to trivial block partitioning\n");
        partial_halo_flag = 0;
      }
    } else {
      op_printf("Partitioning Routine : %s UNSUPPORTED\n", lib_routine);
      op_printf("Reverting to trivial block partitioning\n");
      partial_halo_flag = 0;
    }
#else
    op_printf("OP2 Library Not built with Partitioning Library : %s\n",
              lib_name);
    op_printf("Ignoring input routine : %s\n", lib_routine);
    if (prime_set != NULL)
      op_printf("Ignoring input mapping : %s\n", prime_set->name);
    if (prime_map != NULL)
      op_printf("Ignoring input mapping : %s\n", prime_map->name);
    if (data != NULL)
      op_printf("Ignoring input data : %s\n", data->name);
    op_printf("Reverting to trivial block partitioning\n");
    partial_halo_flag = 0;
#endif
  } else if (strcmp(lib_name, "PARMETIS") == 0) {
#ifdef HAVE_PARMETIS
    op_printf("Selected Partitioning Library : %s\n", lib_name);
    if (strcmp(lib_routine, "KWAY") == 0) {
      op_printf("Selected Partitioning Routine : %s\n", lib_routine);
      if (prime_map != NULL)
        op_partition_graph_parmetis(prime_map); // use parmetis kaway partitioning
      else {
        op_printf("Partitioning prime_map : NULL - UNSUPPORTED Partitioner "
                  "Specification\n");
        op_printf("Reverting to trivial block partitioning\n");
        partial_halo_flag = 0;
      }
    } else if (strcmp(lib_routine, "GEOMKWAY") == 0) {
      op_printf("Selected Partitioning Routine : %s\n", lib_routine);
      if (prime_map != NULL && data != NULL)
        op_partition_geomkway(data,
                              prime_map); // use parmetis kawaygeom partitioning
      else {
        op_printf("Partitioning prime_map or coordinates : NULL - UNSUPPORTED "
                  "Partitioner Specification\n");
        op_printf("Reverting to trivial block partitioning\n");
        partial_halo_flag = 0;
      }
    } else if (strcmp(lib_routine, "GEOM") == 0) {
      op_printf("Selected Partitioning Routine : %s\n", lib_routine);
      if (data != NULL)
        op_partition_geom(data); // use parmetis geometric partitioning
      else {
        op_printf("Partitioning coordinates: NULL - UNSUPPORTED Partitioner "
                  "Specification\n");
        op_printf("Reverting to trivial block partitioning\n");
        partial_halo_flag = 0;
      }
    } else {
      op_printf("Partitioning Routine : %s UNSUPPORTED\n", lib_routine);
      op_printf("Reverting to trivial block partitioning\n");
      partial_halo_flag = 0;
    }
#else
    /*  Suppress warning */
    (void)data;
    op_printf("OP2 Library Not built with Partitioning Library : %s\n",
              lib_name);
    op_printf("Ignoring input routine : %s\n", lib_routine);
    if (prime_set != NULL)
      op_printf("Ignoring input set : %s\n", prime_set->name);
    if (prime_map != NULL)
      op_printf("Ignoring input mapping : %s\n", prime_map->name);
    if (data != NULL)
      op_printf("Ignoring input coordinates : %s\n", data->name);
    op_printf("Reverting to trivial block partitioning\n");
    partial_halo_flag = 0;
#endif
  } else if (strcmp(lib_name, "RANDOM") == 0) {
    op_printf("Selected Partitioning Routine : %s\n", lib_name);
    if (prime_set != NULL)
      op_partition_random(
          prime_set); // use a random partitioning - used for debugging
    else {
      op_printf("Partitioning prime_set : NULL - UNSUPPORTED Partitioner "
                "Specification\n");
      op_printf("Reverting to trivial block partitioning\n");
      partial_halo_flag = 0;
    }
  } else if (strcmp(lib_name, "EXTERNAL") == 0) {
    op_printf("Selected Partitioning Routine : %s\n", lib_name);
    if (prime_set != NULL) {
      if (data->data != NULL) {
        if (data->dim == 1) {
          op_partition_external(
              prime_set,
              data); // use an external partitioning read in from hdf5 file
          partial_halo_flag = 0;
        } else {
          op_printf("External Partition vector should be an integer array with "
                    "dimension 1\n");
          op_printf("Reverting to trivial block partitioning\n");
          partial_halo_flag = 0;
        }
      } else {
        op_printf("External Partition vector : NULL - UNSUPPORTED Partitioner "
                  "Specification\n");
        partial_halo_flag = 0;
      }
    } else {
      op_printf("Partitioning prime_set : NULL - UNSUPPORTED Partitioner "
                "Specification\n");
      op_printf("Reverting to trivial block partitioning\n");
      partial_halo_flag = 0;
    }
  } else if (strcmp(lib_name, "INERTIAL") == 0) {
    op_printf("Selected Partitioning Routine : %s\n", lib_name);
    if (data->data != NULL) {
      if (data->dim == 3)
        op_partition_inertial(data); // use Oplus style Inertial partitioning
      else {
        op_printf("Onlt supports 3D Inertial Bisection Partitioning - Need 3D "
                  "coordinates - dim should be 3\n");
        op_printf("Reverting to trivial block partitioning\n");
        partial_halo_flag = 0;
      }
    } else {
      op_printf("Partitioning based on dataset : NULL - UNSUPPORTED "
                "Partitioner Specification\n");
      op_printf("Reverting to trivial block partitioning\n");
      partial_halo_flag = 0;
    }
  } else {
    op_printf("Partitioning Library : %s UNSUPPORTED\n", lib_name);
    op_printf("Ignoring input routine : %s\n", lib_routine);
    if (prime_set != NULL)
      op_printf("Ignoring input set : %s\n", prime_set->name);
    if (prime_map != NULL)
      op_printf("Ignoring input mapping : %s\n", prime_map->name);
    if (data != NULL)
      op_printf("Ignoring input coordinates : %s\n", data->name);
    op_printf("Reverting to trivial block partitioning\n");
    partial_halo_flag = 0;
  }

  // trigger halo creation routines
  op_halo_create();

  if (partial_halo_flag == 1) // only do partial halo
    op_halo_permap_create();  // creation if a valid partitioning is done
  else {
    OP_map_partial_exchange = (int *)xmalloc(OP_map_index * sizeof(int));
    for (int i = 0; i < OP_map_index; i++)
      OP_map_partial_exchange[i] = 0;
  }

#ifdef DEBUG // sanity check to identify if the partitioning results in ophan
             // elements
  int ctr = 0;
  for (int i = 0; i < prime_map->from->size; i++) {
    if (prime_map->map[2 * i] >= prime_map->to->size &&
        prime_map->map[2 * i + 1] >= prime_map->to->size)
      ctr++;
  }
  printf("Orphan edges: %d\n", ctr);
#endif
  OP_is_partitioned = 1;
}

extern int **OP_map_ptr_list;
void op_partition_ptr(const char *lib_name, const char *lib_routine,
                      op_set prime_set, int *prime_map, double *coords) {
  op_dat_entry *item;
  op_dat_entry *tmp_item;
  op_dat item_dat = NULL;
  for (item = TAILQ_FIRST(&OP_dat_list); item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    // printf("Available op_dat %s with pointer %p\n", item->dat->name,
    // item->dat->data);
    if (item->orig_ptr == coords) {
      // printf("%s(%p), ", item->dat->name, item->dat->data);
      item_dat = item->dat;
      break;
    }
  }
  // printf("\n");
  if (item_dat == NULL) {
    printf("ERROR in op_partition: op_dat not found for dat with %p pointer\n",
           (void*)coords);
  }

  op_map item_map = op_search_map_ptr(prime_map);

  if (item_map == NULL) {
    printf("ERROR in op_partition: op_map not found for %p pointer\n", (void*)prime_map);
    exit(-1);
  }

  op_partition(lib_name, lib_routine, prime_set, item_map, item_dat);
}

#ifdef __cplusplus
}
#endif

/*******************************************************************************
 * Initialise partitioning data structures with the current (block)
*  partitioning information
 *******************************************************************************/
idx_g_t **initialise(int my_rank, int comm_size) {
  // Compute global partition range information for each set
  idx_g_t **part_range = (idx_g_t **)xmalloc(OP_set_index * sizeof(idx_g_t *));
  get_part_range(part_range, my_rank, comm_size, OP_PART_WORLD);

  // save the original part_range for future partition reversing
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

  // allocate memory for list
  OP_part_list = (part *)xmalloc(OP_set_index * sizeof(part));

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    // printf("set %s size = %d\n", set.name, set.size);
    idx_g_t *g_index = (idx_g_t *)xmalloc(sizeof(idx_g_t) * set->size);
    for (int i = 0; i < set->size; i++)
      g_index[i] =
          get_global_index(i, my_rank, part_range[set->index], comm_size);
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
std::vector<std::vector<idx_g_t>> construct_adj_list(op_map primary_map, int my_rank, int comm_size,
                                                     idx_g_t **part_range) {
  const int dim = primary_map->dim;
  idx_g_t *range = part_range[primary_map->to->index];

  // each from-element goes to every other rank its row reaches (from_pairs drops repeats)
  std::vector<int> pairs;
  for (int e = 0; e < primary_map->from->size; e++) {
    for (int j = 0; j < dim; j++) {
      int local_index;
      int part = get_partition(primary_map->map_gbl[(std::size_t)e * dim + j], range, &local_index, comm_size,
                               primary_map->to);
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
        if (get_partition(row[j], range, &local_index, comm_size, primary_map->to) != my_rank)
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
    op_printf("Error: global index out of 32-bit integer range for map %s on rank %d (index=%lld)\n",
              primary_map->name, my_rank, (long long)g_index);
    MPI_Abort(OP_PART_WORLD, 2);
  }
}
#endif

/*******************************************************************************
 * Setup variables for k-way partitioning
 *******************************************************************************/
template <class T>
std::tuple<T *, T *, T *, T *, T, T, real_t *, real_t *>
setup_part_data(op_map primary_map, int my_rank, int comm_size, std::vector<std::vector<idx_g_t>> adj,
                idx_g_t **part_range) {
  T comm_size_pm = comm_size;

  T *vtxdist = (T *)xmalloc(sizeof(T) * (comm_size + 1));
  for (int i = 0; i < comm_size; i++) {
    vtxdist[i] = part_range[primary_map->to->index][2 * i];
  }
  vtxdist[comm_size] =
      part_range[primary_map->to->index][2 * (comm_size - 1) + 1] + 1;

  // neighbours sorted, self excluded
  std::size_t n_edges = 0;
  for (const std::vector<idx_g_t> &neighbours : adj)
    n_edges += neighbours.size();
  T *xadj = (T *)xmalloc(sizeof(T) * (primary_map->to->size + 1));
  T *adjncy = (T *)xmalloc(sizeof(T) * n_edges);
  T count = 0;
  for (int i = 0; i < primary_map->to->size; i++) {
    idx_g_t g_index = get_global_index(
        i, my_rank, part_range[primary_map->to->index], comm_size);
#ifdef DEBUG
    if constexpr (sizeof(T) == sizeof(int)) {
      check_global_index_int32_range(g_index, primary_map, my_rank);
    }
#endif
    if (adj[i].size() < 2) {
      printf("The from set: %s of primary map: %s is not an on to set of "
             "to-set: %s\n",
             primary_map->from->name, primary_map->name, primary_map->to->name);
      printf("Need to select a different primary map\n");
      MPI_Abort(OP_PART_WORLD, 2);
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
  for (int i = 0; i < comm_size * ncon; i++)
    tpwgts[i] = hybrid_flags[i] == 1 ? OP_hybrid_balance / total : 1.0 / total;

  real_t *ubvec = (real_t *)xmalloc(sizeof(real_t) * ncon);
  *ubvec = 1.05;

  op_free(hybrid_flags);
  return std::make_tuple(vtxdist, xadj, adjncy, partition_pm, comm_size_pm,
                         ncon, tpwgts, ubvec);
}

/*******************************************************************************
 * Check partitioning was performed as expected
 *******************************************************************************/
template <class T>
void check_partition(op_map primary_map, T *partition_pm, int my_rank,
                     int comm_size) {
  int *partition = (int *)xmalloc(sizeof(int) * primary_map->to->size);
  for (int i = 0; i < primary_map->to->size; i++) {
    // sanity check to see if all elements were partitioned
    if (partition_pm[i] < 0 || partition_pm[i] >= comm_size) {
      printf("Partitioning problem: on rank %d, set %s element %d not assigned "
             "a partition\n",
             my_rank, primary_map->to->name, i);
      MPI_Abort(OP_PART_WORLD, 2);
    }
    partition[i] = partition_pm[i];
  }
  free(partition_pm);

  // initialise primary set as partitioned
  OP_part_list[primary_map->to->index]->elem_part = partition;
  OP_part_list[primary_map->to->index]->is_partitioned = 1;
}

/*******************************************************************************
 * Generic wrapper for kway-partition functions
 *******************************************************************************/

#ifdef HAVE_PARMETIS
void perform_kway_partition(idx_t *vtxdist, idx_t *xadj, idx_t *adjncy,
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
void perform_kway_partition(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy,
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
std::tuple<SCOTCH_Dgraph*, SCOTCH_Num*, SCOTCH_Num*, SCOTCH_Num*>
setup_ptscotch_data(op_map primary_map, int my_rank, int comm_size, std::vector<std::vector<idx_g_t>> adj,
                idx_g_t **part_range) {

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
        idx_g_t g_index = get_global_index(
            i, my_rank, part_range[primary_map->to->index], comm_size);
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
        printf("PT-Scotch Graph Inconsistent - Aborting\n");
        MPI_Abort(OP_PART_WORLD, 2);
    }

    SCOTCH_Num *partloctab =
        (SCOTCH_Num *)xmalloc(sizeof(SCOTCH_Num) * primary_map->to->size);
    for (SCOTCH_Num i = 0; i < primary_map->to->size; i++) {
        partloctab[i] = -99;
    }

    return std::make_tuple(grafptr, partloctab, vertloctab, edgeloctab);
}

// Helper function to call PTScotch partitioner
void perform_ptscotch_partition(SCOTCH_Dgraph *grafptr, int comm_size, SCOTCH_Num *partloctab, SCOTCH_Num *vertloctab, SCOTCH_Num *edgeloctab) {
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
template <class T> // Keep template for ParMETIS/KaHIP type compatibility
void op_partition_graph_generic(op_map primary_map, const char* partitioner_name,
                                op_dat coords = nullptr) {
  // declare timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  double time;
  double max_time;

  op_timers(&cpu_t1, &wall_t1); // timer start for partitioning

  // create new communicator for partitioning
  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_PART_WORLD);
  MPI_Comm_rank(OP_PART_WORLD, &my_rank);
  MPI_Comm_size(OP_PART_WORLD, &comm_size);

#ifdef DEBUG
    // check if the  primary_map is an on to map from the from-set to the to-set
  if (is_onto_map(primary_map) != 1) {
    printf("Map %s is an not an onto map from set %s to set %s \n",
           primary_map->name, primary_map->from->name, primary_map->to->name);
    MPI_Abort(OP_PART_WORLD, 2);
  }
#endif

#ifdef HAVE_PARMETIS
  // Coordinates make the ParMETIS k-way partitioning geometric: PartGeomKway.
  std::vector<real_t> xyz;
  if (coords != nullptr) {
    if (compare_sets(coords->set, primary_map->to) == 0) {
      printf("primary map's to set %s mismatches the op_dat's set %s: on rank %d\n", primary_map->to->name,
             coords->set->name, my_rank);
      MPI_Abort(OP_PART_WORLD, 2);
    }
    xyz = parmetis_coordinates(coords);
  }
#endif

  /*--STEP 0 - initialise partitioning data structures */
  idx_g_t **part_range = initialise(my_rank, comm_size);

  /*--STEP 1 - Construct adjacency list */
  std::vector<std::vector<idx_g_t>> adj = construct_adj_list(primary_map, my_rank, comm_size, part_range);

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

      // clean up part_range before calling Partitioner
      for (int i = 0; i < OP_set_index; i++) op_free(part_range[i]);
      op_free(part_range);

      if (my_rank == MPI_ROOT) {
          printf("-----------------------------------------------------------\n");
          if (use_kahip) printf("ParHIPPartitionKWay Output\n");
          else if (coords != nullptr) printf("ParMETIS_V3_PartGeomKway Output\n");
          else printf("ParMETIS_V3_PartKway Output\n");
          printf("-----------------------------------------------------------\n");
      }

#ifdef HAVE_PARMETIS
      if constexpr (std::is_same_v<T, idx_t>) {
        if (coords != nullptr) {
          T ndims = coords->dim;
          ParMETIS_V3_PartGeomKway(vtxdist, xadj, adjncy, NULL, NULL, &wgtflag, &numflag, &ndims, xyz.data(), &ncon,
                                   &comm_size_pm, tpwgts, ubvec, options, &edge_cut, partition_pm, &OP_PART_WORLD);
        }
      }
      if (!use_kahip && coords == nullptr) {
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

      check_partition<T>(primary_map, partition_pm, my_rank, comm_size);
#else
      // Error: Library not available
      if (my_rank == MPI_ROOT) printf("ERROR: %s requested but not compiled.\n", partitioner_name);
      MPI_Abort(OP_PART_WORLD, 1);
#endif
  } else if (strcmp(partitioner_name, "PTSCOTCH") == 0) {
#ifdef HAVE_PTSCOTCH
      SCOTCH_Dgraph *grafptr;
      SCOTCH_Num *partloctab;
      SCOTCH_Num *vertloctab;
      SCOTCH_Num *edgeloctab;
      // moved in, so it is freed before the partitioner runs
      std::tie(grafptr, partloctab, vertloctab, edgeloctab) =
          setup_ptscotch_data(primary_map, my_rank, comm_size, std::move(adj), part_range);

      // clean up part_range before calling Partitioner
      for (int i = 0; i < OP_set_index; i++) op_free(part_range[i]);
      op_free(part_range);

      if (my_rank == MPI_ROOT) {
          printf("-----------------------------------------------------------\n");
          printf("PT-Scotch Output\n");
          printf("-----------------------------------------------------------\n");
      }
      perform_ptscotch_partition(grafptr, comm_size, partloctab, vertloctab, edgeloctab);
      if (my_rank == MPI_ROOT) {
          printf("-----------------------------------------------------------\n");
      }

      check_partition(primary_map, partloctab, my_rank, comm_size);
#else
      // Error: Library not available
       if (my_rank == MPI_ROOT) printf("ERROR: PTScotch requested but not compiled.\n");
      MPI_Abort(OP_PART_WORLD, 1);
#endif
  } else {
       // Error: Unknown partitioner
       if (my_rank == MPI_ROOT) printf("ERROR: Unknown partitioner '%s'\n", partitioner_name);
       MPI_Abort(OP_PART_WORLD, 1);
  }


  /*-STEP 2 - Partition all other sets,migrate data and renumber mapping tables-*/
  partition_all(primary_map->to, my_rank, comm_size);
  migrate_all(my_rank);
  renumber_maps(my_rank, comm_size);
  /* Final timing and cleanup */
  op_timers(&cpu_t2, &wall_t2);
  time = wall_t2 - wall_t1;
  MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_ROOT, OP_PART_WORLD);
  MPI_Comm_free(&OP_PART_WORLD);
  if (my_rank == MPI_ROOT)
    printf("Max total %s partitioning time = %lf\n", partitioner_name, max_time);
}


// Specializations/Wrappers to call the generic function
#ifdef HAVE_PARMETIS
void op_partition_graph_parmetis(op_map primary_map) {
    op_partition_graph_generic<idx_t>(primary_map, "PARMETIS");
}

void op_partition_geomkway(op_dat coords, op_map primary_map) {
    op_partition_graph_generic<idx_t>(primary_map, "PARMETIS", coords);
}
#endif
#ifdef HAVE_KAHIP
void op_partition_graph_kahip(op_map primary_map) {
    op_partition_graph_generic<idxtype>(primary_map, "KAHIP");
}
#endif
#ifdef HAVE_PTSCOTCH
void op_partition_graph_ptscotch(op_map primary_map) {
    // Pass dummy template type idx_t, it's ignored by the PTScotch path
    op_partition_graph_generic<idx_t>(primary_map, "PTSCOTCH");
}
#endif

