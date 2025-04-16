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

typedef struct {
  // set related to this list
  op_set set;
  // number of elements in this list (local elements & indices)
  idx_l_t size;
  // MPI ranks to be exported to or imported from
  int *ranks;
  // number of MPI neighbors to be exported to or imported from
  int ranks_size;
  // displacements for the starting point of each rank's element list (local elements & indices)
  idx_l_t *disps;
  // number of elements exported to or imported from each ranks (local elements & indices)
  idx_l_t *sizes;
  // the list of all elements (local elements & indices)
  idx_l_t *list;
} halo_list_core;

typedef halo_list_core *halo_list;

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

// Structs and functions that use MPI definitions
#ifndef OP_MPI_CORE_NOMPI

/*******************************************************************************
* Buffer struct used in non-blocking mpi halo sends/receives
*******************************************************************************/

typedef struct {
  // buffer holding exec halo to be exported;
  char *buf_exec;
  // buffer holding nonexec halo to be exported;
  char *buf_nonexec;
  // pointed to hold the MPI_Reqest for sends
  MPI_Request *s_req;
  // pointed to hold the MPI_Reqest for receives
  MPI_Request *r_req;
  // number of send MPI_Reqests in flight at a given time for this op_dat
  int s_num_req;
  // number of receive MPI_Reqests in flight at a given time for this op_dat
  int r_num_req;
} op_mpi_buffer_core;

typedef op_mpi_buffer_core *op_mpi_buffer;

#endif /* OP_MPI_CORE_NOMPI */

/** external variables **/

extern int OP_part_index;
extern part *OP_part_list;
extern idx_g_t **orig_part_range;

/** export list on the device **/

extern idx_l_t **export_exec_list_d;
extern idx_l_t **export_exec_list_disps_d;
extern idx_l_t **export_nonexec_list_d;
extern idx_l_t **export_nonexec_list_disps_d;
extern idx_l_t **export_nonexec_list_partial_d;
extern idx_l_t **import_nonexec_list_partial_d;
extern idx_l_t *set_import_buffer_size;
extern idx_l_t **import_exec_list_disps_d;
extern idx_l_t **import_nonexec_list_disps_d;

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
                  int comm_size);

idx_g_t get_global_index(idx_l_t local_index, int partition, idx_g_t *part_range,
                     int comm_size);

void find_neighbors_set(halo_list List, int *neighbors, int *sizes,
                        int *ranks_size, int my_rank, int comm_size,
                        MPI_Comm Comm);

void create_list(int *list, int *ranks, int *disps, int *sizes, int *ranks_size,
                 int *total, int *temp_list, int size, int comm_size,
                 int my_rank);

void create_export_list(op_set set, int *temp_list, halo_list h_list, int size,
                        int comm_size, int my_rank);

void create_import_list(op_set set, int *temp_list, halo_list h_list,
                        int total_size, int *ranks, int *sizes, int ranks_size,
                        int comm_size, int my_rank);

int is_onto_map(op_map map);

/*******************************************************************************
* Core MPI lib function prototypes
*******************************************************************************/

void op_halo_create();

void op_halo_permap_create();

void op_halo_destroy();

op_dat op_mpi_get_data(op_dat dat);

void fetch_data_hdf5(op_dat dat, char *usr_ptr, int low, int high);

void mpi_timing_output();

void op_mpi_exit();

void print_dat_to_txtfile_mpi(op_dat dat, const char *file_name);

void print_dat_to_binfile_mpi(op_dat dat, const char *file_name);

void op_mpi_put_data(op_dat dat);

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

void op_partition_meshkway(op_map primary_map); // does not work
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

/*******************************************************************************
* External functions defined in op_mpi_(cuda)_rt_support.c
*******************************************************************************/

void op_exchange_halo(op_arg *arg, int exec_flag);
void op_exchange_halo_partial(op_arg *arg, int exec_flag);
void op_wait_all(op_arg *arg);
int op_mpi_test(op_arg *arg);
void op_exchange_halo_cuda(op_arg *arg, int exec_flag);
void op_exchange_halo_partial_cuda(op_arg *arg, int exec_flag);
void op_wait_all_cuda(op_arg *arg);

void op_download_buffer_async(char *send_buffer_device, char *send_buffer_host, unsigned size_send);
void op_upload_buffer_async  (char *recv_buffer_device, char *recv_buffer_host, unsigned size_recv);
void op_download_buffer_sync();
void op_gather_record();
void op_scatter_sync();
void op_gather_sync();

#ifdef __cplusplus
}
#endif

#endif /* OP_MPI_CORE_NOMPI */
#endif /* __OP_MPI_CORE_H */
