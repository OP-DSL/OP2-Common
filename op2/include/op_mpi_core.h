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

#ifndef __OP_MPI_CORE_H
#define __OP_MPI_CORE_H

#include <memory>
#include <vector>

/*
 * op_mpi_core.h
 *
 * Headder file for the OP2 Distributed memory (MPI) halo creation,
 * halo exchange and support utility routines/functions
 *
 * written by: Gihan R. Mudalige, (Started 01-03-2011)
 */
#ifndef OP_MPI_CORE_NOMPI
#include <mpi.h>

/** Define the root MPI process **/
#ifdef MPI_ROOT
#undef MPI_ROOT
#endif
#define MPI_ROOT 0

/** extern variables for halo creation and exchange**/
extern MPI_Comm OP_MPI_WORLD;
extern MPI_Comm OP_MPI_GLOBAL;

#endif /* OP_MPI_CORE_NOMPI */
// Structs that don't need the MPI include

/*******************************************************************************
* MPI halo list data type
*******************************************************************************/

/* One halo list: for each neighbour rank, a contiguous block of `list`.
 *
 *   ranks      ascending and unique, one per neighbour
 *   sizes[i]   how many entries ranks[i] has, at list[disps[i] .. disps[i] + sizes[i]); > 0
 *   disps      the prefix sum of sizes, disps[0] == 0
 *   list       size() entries in all; null when there are none
 *
 * A value: it owns its arrays, and moving it moves them. What an entry of `list`
 * means depends on which list this is - see SetHalo and MapHalo. Build one with
 * the halo_list_from_* functions. */
struct HaloList {
  op_set set = nullptr;
  std::vector<int> ranks;
  std::vector<idx_l_t> sizes;
  std::vector<idx_l_t> disps;
  std::unique_ptr<idx_l_t[]> list;

  int ranks_size() const { return (int)ranks.size(); }
  idx_l_t size() const { return ranks.empty() ? 0 : disps.back() + sizes.back(); }
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

/*******************************************************************************
* Data structures related to MPI level partitioning
*******************************************************************************/

// struct to hold the partition information for each set
typedef struct {
  // set to which this partition info blongs to
  op_set set;
  // global index of each element held in this MPI process
  idx_g_t *g_index;
  // partition to which each element belongs
  int *elem_part;
  // indicates if this set is partitioned 1 if partitioned 0 if not
  int is_partitioned;
} part_core;

typedef part_core *part;

/*******************************************************************************
* Data structure to hold mpi communications of an op_dat
*******************************************************************************/
#define NAMESIZE 20
typedef struct {
  // name of this op_dat
  char name[NAMESIZE];
  // size of this op_dat
  int size;
  // index of this op_dat
  int index;
  // total number of times this op_dat was halo exported
  int count;
  // total number of bytes halo exported for this op_dat in this kernel
  idx_g_t bytes;
} op_dat_mpi_comm_info_core;

typedef op_dat_mpi_comm_info_core *op_dat_mpi_comm_info;

/*******************************************************************************
* Data Type to hold MPI performance measures
*******************************************************************************/

typedef struct {

  // name of kernel
  char name[NAMESIZE];
  // total time spent in this kernel (compute + comm - overlap)
  double time;
  // number of times this kernel is called
  int count;
  // number of op_dat indices in this kernel
  int num_indices;
  // array to hold all the op_dat_mpi_comm_info structs for this kernel
  op_dat_mpi_comm_info *comm_info;
  // capacity of comm_info array
  int cap;
} op_mpi_kernel;

/** external variables **/

extern int OP_part_index;
extern part *OP_part_list;
extern idx_g_t **orig_part_range;

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
extern std::vector<DeviceSetHalo> OP_set_halos_d; // by set index
extern std::vector<DeviceMapHalo> OP_map_halos_d; // by map index

// Structs and functions that use MPI definitions
#ifndef OP_MPI_CORE_NOMPI

template <typename T>
MPI_Datatype get_mpi_type() {
  if (std::is_same<T, long long>::value) {
    return MPI_LONG_LONG;
  } else if (std::is_same<T, int>::value) {
    return MPI_INT;
  } else if (std::is_same<T, idx_g_t>::value) {
    return MPI_LONG_LONG;
  } else {
    throw std::runtime_error("Unsupported type");
  }
  // #endif
}

template <typename T>
MPI_Datatype get_mpi_type(T*) {
  return get_mpi_type<T>();
}

/* For a count each rank holds part of, in rank order: the total over all ranks,
   and this rank's offset - the sum over the ranks below it. One collective each,
   nothing sized by the number of ranks. */
inline idx_g_t op_mpi_total(idx_g_t n, MPI_Comm comm) {
  idx_g_t total = 0;
  MPI_Allreduce(&n, &total, 1, get_mpi_type(&n), MPI_SUM, comm);
  return total;
}

inline idx_g_t op_mpi_offset(idx_g_t n, MPI_Comm comm) {
  idx_g_t below = 0;
  MPI_Exscan(&n, &below, 1, get_mpi_type(&n), MPI_SUM, comm);
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank == 0 ? 0 : below;  // MPI_Exscan leaves rank 0's result undefined
}

/* An MPI datatype for one element of a dat, so a transfer's count is elements,
   not bytes: an int byte count stops at 2 GB, which one neighbour's block of a
   large dat can pass. Freed with the scope; MPI lets a pending transfer finish
   with a datatype that has been freed. */
struct DatElementType {
  MPI_Datatype type;
  explicit DatElementType(op_dat dat) {
    MPI_Type_contiguous(dat->size, MPI_BYTE, &type);
    MPI_Type_commit(&type);
  }
  ~DatElementType() { MPI_Type_free(&type); }
  DatElementType(const DatElementType &) = delete;
  DatElementType &operator=(const DatElementType &) = delete;
  operator MPI_Datatype() const { return type; }
};

/* Build a HaloList. Named for how a list is built rather than for which list it
   becomes: an export list and the nonexec import list are both built from pairs,
   and halo_list_transpose turns either kind into the other. */

/* From (rank, index) pairs, n_ints ints in all: each rank's indices sorted and
   deduplicated, ranks that end up with none left out. */
HaloList halo_list_from_pairs(op_set set, const int *pairs, int n_ints);

/* From groups already formed, taking ownership of all three: ranks ascending and
   unique, sizes[i] > 0 entries of list for ranks[i], in that order. */
HaloList halo_list_from_groups(op_set set, std::vector<int> ranks,
                               std::vector<idx_l_t> sizes,
                               std::unique_ptr<idx_l_t[]> list);

/* The list on the other side: each rank's block goes to that rank, and the result
   holds what every rank sent here, grouped by sender. An export list gives the
   matching import list and an import list the matching export list. The senders
   are discovered by a sparse exchange, not a collective. */
HaloList halo_list_transpose(op_set set, const HaloList &list, MPI_Comm comm);

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************
* Utility function prototypes
*******************************************************************************/

void decl_partition(op_set set, idx_g_t *g_index, int *partition);

void get_part_range(idx_g_t **part_range, int my_rank, int comm_size,
                    MPI_Comm Comm);

int get_partition(idx_g_t global_index, idx_g_t *part_range, idx_l_t *local_index,
                  int comm_size, op_set set);

idx_g_t get_global_index(idx_l_t local_index, int partition, idx_g_t *part_range,
                     int comm_size);

int is_onto_map(op_map map);

/*******************************************************************************
* Core MPI lib function prototypes
*******************************************************************************/

void op_halo_create();

void op_halo_permap_create();

/* Refill every set's import lists from their owners' export lists, so each entry
   is again the element's current local index on its owner. An owner that
   reorders its elements rewrites only its own export lists; call this after it
   does. The layout is unchanged. Collective over OP_MPI_WORLD. */
void op_halo_refresh_imports();

void op_halo_destroy();

op_dat op_mpi_get_data(op_dat dat);

void fetch_data_hdf5(op_dat dat, char *usr_ptr, int low, int high);

void mpi_timing_output();

void op_mpi_exit();

/* Stops the job if a halo exchange is still waiting for its wait. */
void op_mpi_unified_exit();

void print_dat_to_txtfile_mpi(op_dat dat, const char *file_name);

void print_dat_to_binfile_mpi(op_dat dat, const char *file_name);

/* Scatter user data from the original block partition into the current
 * (repartitioned) layout of dat. ptr holds local_size elements, which must
 * match this rank's original block-partitioned set size. */
void op_mpi_put_data(op_dat dat, void *ptr, size_t local_size);

void op_mpi_init(int argc, char **argv, int diags, MPI_Fint global,
                 MPI_Fint local);
void op_mpi_init_soa(int argc, char **argv, int diags, MPI_Fint global,
                     MPI_Fint local, int soa);

/* Defined in op_mpi_decl.c, may need to be put in a seperate headder file */
size_t op_mv_halo_device(op_set set, op_dat dat);

/* Defined in op_mpi_decl.c, may need to be put in a seperate headder file */
size_t op_mv_halo_list_device();

void partition(const char *lib_name, const char *lib_routine, op_set prime_set,
               op_map prime_map, op_dat coords);

/******************************************************************************
* Custom partitioning wrapper prototypes
*******************************************************************************/

void op_partition_random(op_set primary_set);

void op_partition_external(op_set primary_set, op_dat partvec);

void op_partition_inertial(op_dat x);


#ifdef HAVE_PARMETIS
/*******************************************************************************
* ParMetis wrapper prototypes
*******************************************************************************/

void op_partition_geom(op_dat coords);

void op_partition_geomkway(op_dat coords, op_map primary_map);

#endif

#if defined(HAVE_KAHIP) || defined(HAVE_PARMETIS)
/*******************************************************************************
* K-way partitioning prototype
*******************************************************************************/

void op_partition_kway(op_map primary_map, bool use_kahip);

#endif

#ifdef HAVE_PTSCOTCH
/*******************************************************************************
* PT-SCOTCH wrapper prototypes
*******************************************************************************/

void op_partition_ptscotch(op_map primary_map);
#endif

void op_move_to_device();

#ifdef __cplusplus
}
#endif

#endif /* OP_MPI_CORE_NOMPI */
#endif /* __OP_MPI_CORE_H */
