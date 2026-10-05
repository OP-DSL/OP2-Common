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

#ifndef __OP_MPI_HALO_H
#define __OP_MPI_HALO_H

/*
 * The MPI library's halo lists and the C++ helpers that build and use them.
 *
 * Internal to the library: op_lib_mpi.h, which applications include, does not
 * include this header, so none of it reaches application code. The types and
 * helpers are in op::mpi. The tables of lists stay global - OP_set_halos and
 * OP_map_halos, and their device copies - beside OP_set_list and OP_part_list.
 *
 * As in op_mpi_core.h, the parts that need <mpi.h> are left out when
 * OP_MPI_CORE_NOMPI is defined, which is how nvcc compiles the GPU library.
 */

#include <op_lib_core.h>
#include <op_mpi_core.h>

#include <memory>
#include <vector>

namespace op::mpi {

/* One halo list: for each neighbour rank, a contiguous block of `list`.
 *
 *   ranks      ascending and unique, one per neighbour
 *   sizes[i]   how many entries ranks[i] has, at list[disps[i] .. disps[i] + sizes[i]); > 0
 *   disps      the prefix sum of sizes, disps[0] == 0
 *   list       size() entries in all; null when there are none
 *
 * A value: it owns its arrays, and moving it moves them. What an entry of `list`
 * means depends on which list this is - see SetHalo and MapHalo. */
struct HaloList {
  op_set set = nullptr;
  std::vector<int> ranks;
  std::vector<idx_l_t> sizes;
  std::vector<idx_l_t> disps;
  std::unique_ptr<idx_l_t[]> list;

  int ranks_size() const { return (int)ranks.size(); }
  idx_l_t size() const { return ranks.empty() ? 0 : disps.back() + sizes.back(); }

  /* Named for how a list is built rather than for which list it becomes: an
     export list and the nonexec import list are both built from pairs, and
     transpose (below) turns either kind into the other. */

  /* From (rank, index) pairs, n_ints ints in all: each rank's indices sorted and
     deduplicated, ranks that end up with none left out. */
  static HaloList from_pairs(op_set set, const int *pairs, int n_ints);

  /* From groups already formed, taking ownership of all three: ranks ascending
     and unique, sizes[i] > 0 entries of list for ranks[i], in that order. */
  static HaloList from_groups(op_set set, std::vector<int> ranks, std::vector<idx_l_t> sizes,
                              std::unique_ptr<idx_l_t[]> list);
};

/* A set's four halo lists. Each dat on the set holds its halo after its owned
   elements: import_exec's elements, then import_nonexec's, each rank's block
   landing contiguously at its disps. */
struct SetHalo {
  HaloList export_exec;     // local indices of owned elements a neighbour executes over
  HaloList import_exec;     // the elements received to execute over, in halo order: each
                            // one's current local index on its owner
  HaloList export_nonexec;  // local indices of owned elements a neighbour only reads
  HaloList import_nonexec;  // as import_exec, for elements only read
};

/* The partial-exchange lists of a map, empty unless OP_map_partial_exchange. */
struct MapHalo {
  HaloList export_nonexec;  // local indices of the owned elements each neighbour needs
                            // through this map
  HaloList import_nonexec;  // local indices of the halo elements this map reaches
};

/* A halo list's entries in device memory, freed with it. The deleter is defined
   by the GPU library, so this header needs no GPU headers. */
struct DeviceFree {
  void operator()(idx_l_t *p) const;
};
using DeviceList = std::unique_ptr<idx_l_t, DeviceFree>;

/* The device copies of the lists the unified exchange gathers and scatters
   through, uploaded by op_mv_halo_list_device and null until then (or for good,
   in a CPU library). A full exchange scatters by offset, so a set's import lists
   stay on the host. */
struct DeviceSetHalo {
  DeviceList export_exec, export_nonexec;
};
struct DeviceMapHalo {
  DeviceList export_nonexec, import_nonexec;
};

#ifndef OP_MPI_CORE_NOMPI

/* The list on the other side: each rank's block goes to that rank, and the result
   holds what every rank sent here, grouped by sender. An export list gives the
   matching import list and an import list the matching export list. The senders
   are discovered by a sparse exchange, not a collective. */
HaloList transpose(const HaloList &list, MPI_Comm comm);

/* For a count each rank holds part of, in rank order: the total over all ranks,
   and this rank's offset - the sum over the ranks below it. One collective each,
   nothing sized by the number of ranks. */
inline idx_g_t sum_over_ranks(idx_g_t n, MPI_Comm comm) {
  idx_g_t total = 0;
  MPI_Allreduce(&n, &total, 1, get_mpi_type(&n), MPI_SUM, comm);
  return total;
}

inline idx_g_t sum_below_rank(idx_g_t n, MPI_Comm comm) {
  idx_g_t below = 0;
  MPI_Exscan(&n, &below, 1, get_mpi_type(&n), MPI_SUM, comm);
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank == 0 ? 0 : below;  // MPI_Exscan leaves rank 0's result undefined
}

/* Send each neighbour the rows - row_bytes each, of `rows` - that exp lists for
   it, and receive the rows imp lists into `into`, grouped as imp lists them. exp
   and imp must be each other's transpose over comm. Rows are counted in a datatype
   of one row, so a block is not capped at 2 GB. Collective over comm. */
void exchange_rows(MPI_Comm comm, const char *rows, std::size_t row_bytes, const HaloList &exp,
                   const HaloList &imp, char *into);

/* Move a set's rows - its part of a dat, a mapping table or g_index - to the ranks
   elem_part gives them. exp lists the rows leaving, by destination, and imp what
   arrives, by source. The result holds the rows this rank keeps, in order, then
   those received. Allocated with xmalloc; null when empty. Collective over comm. */
char *migrate_rows(MPI_Comm comm, const char *rows, std::size_t row_bytes, int n_rows, const int *elem_part,
                   int my_rank, const HaloList &exp, const HaloList &imp);

#endif /* OP_MPI_CORE_NOMPI */

}  // namespace op::mpi

extern std::vector<op::mpi::SetHalo> OP_set_halos;        // by set index; empty until halo creation
extern std::vector<op::mpi::MapHalo> OP_map_halos;        // by map index; empty until halo creation
extern std::vector<op::mpi::DeviceSetHalo> OP_set_halos_d;  // by set index
extern std::vector<op::mpi::DeviceMapHalo> OP_map_halos_d;  // by map index

#endif /* __OP_MPI_HALO_H */
