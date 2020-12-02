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

#include <mpi.h>

// use uthash from - http://troydhanson.github.com/uthash/
#include <uthash.h>

/** Define the root MPI process **/
#ifdef MPI_ROOT
#undef MPI_ROOT
#endif
#define MPI_ROOT 0

/*******************************************************************************
* MPI halo list data type
*******************************************************************************/

typedef struct {
  // set related to this list
  op_set set;
  // number of elements in this list
  int size;
  // MPI ranks to be exported to or imported from
  int *ranks;
  // number of MPI neighbors to be exported to or imported from
  int ranks_size;
  // displacements for the starting point of each rank's element list
  int *disps;
  // number of elements exported to or imported from each ranks
  int *sizes;
  // the list of all elements
  int *list;
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
  int *g_index;
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
  int bytes;
} op_dat_mpi_comm_info_core;

typedef op_dat_mpi_comm_info_core *op_dat_mpi_comm_info;

/*******************************************************************************
* Data Type to hold MPI performance measures
*******************************************************************************/

typedef struct {

  UT_hash_handle hh; // with this variable uthash makes this structure hashable

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

/** external variables **/

extern int OP_part_index;
extern part *OP_part_list;
extern int **orig_part_range;

/** export list on the device **/

extern int **export_exec_list_d;
extern int **export_exec_list_disps_d;
extern int **export_nonexec_list_d;
extern int **export_nonexec_list_disps_d;
extern int **export_nonexec_list_partial_d;
extern int **import_nonexec_list_partial_d;
extern int *set_import_buffer_size;
extern int **import_exec_list_disps_d;
extern int **import_nonexec_list_disps_d;

/*******************************************************************************
* Data Type to hold sliding planes info
*******************************************************************************/

typedef struct {
  int index;
  int coupling_group_size;
  int *coupling_proclist;

  int num_ifaces;
  int *iface_list;

  int *nprocs_per_int;
  int **proclist_per_int;
  int **nodelist_send_size;
  int ***nodelist_send;

  int max_data_size;
  char ***send_buf;
  MPI_Request **requests;
  MPI_Status **statuses;

  char *OP_global_buffer;
  int OP_global_buffer_size;

  int gbl_num_ifaces;
  int *gbl_iface_list;
  int *nprocs_per_gint;
  int **proclist_per_gint;

  int gbl_offset;
  op_map cellsToNodes;
  op_dat coords;
  op_dat mark;
} op_export_core;

typedef op_export_core *op_export_handle;

typedef struct {
  int index;
  int nprocs;
  int *proclist;
  int gbl_offset;
  op_dat coords;
  op_dat mark;
  int max_dat_size;
  int num_my_ifaces;
  int *iface_list;
  int *nprocs_per_int;
  int **proclist_per_int;
  int *node_size_per_int;
  int **nodelist_per_int;
  char ***recv_buf;
  int *recv2int;
  int *recv2proc;
  MPI_Request *requests;
  MPI_Status *statuses;
  double *interp_dist;

} op_import_core;

typedef op_import_core *op_import_handle;

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************
* Utility function prototypes
*******************************************************************************/

void decl_partition(op_set set, int *g_index, int *partition);

void get_part_range(int **part_range, int my_rank, int comm_size,
                    MPI_Comm Comm);

int get_partition(int global_index, int *part_range, int *local_index,
                  int comm_size);

int get_global_index(int local_index, int partition, int *part_range,
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
void op_mv_halo_device(op_set set, op_dat dat);

/* Defined in op_mpi_decl.c, may need to be put in a seperate headder file */
void op_mv_halo_list_device();

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

void op_partition_kway(op_map primary_map);

void op_partition_geomkway(op_dat coords, op_map primary_map);

void op_partition_meshkway(op_map primary_map); // does not work
#endif

#ifdef HAVE_PTSCOTCH
/*******************************************************************************
* PT-SCOTCH wrapper prototypes
*******************************************************************************/

void op_partition_ptscotch(op_map primary_map);
#endif

/*******************************************************************************
* Sliding planes functionality
*******************************************************************************/
op_export_handle op_export_init(int nprocs, int *proclist, op_map cellsToNodes,
                                op_set sp_nodes, op_dat coords, op_dat mark);
void op_export_data(op_export_handle handle, op_dat dat);
op_import_handle op_import_init(op_export_handle exp_handle, op_dat coords,
                                op_dat mark);
void op_inc_theta(op_export_handle handle, int *sp_id, double *dtheta1,
                  double *dtheta2);
void op_import_data(op_import_handle handle, op_dat dat);
void op_theta_init(op_export_handle handle, int *sp_id, double *dtheta1,
                   double *dtheta2, double *alpha);

void op_move_to_device();
#ifdef __cplusplus
}
#endif

/*******************************************************************************
* External functions defined in op_mpi_(cuda)_rt_support.c
*******************************************************************************/

void op_exchange_halo(op_arg *arg, int exec_flag);
void op_exchange_halo_partial(op_arg *arg, int exec_flag);
void op_wait_all(op_arg *arg);
void op_exchange_halo_cuda(op_arg *arg, int exec_flag);
void op_exchange_halo_partial_cuda(op_arg *arg, int exec_flag);
void op_wait_all_cuda(op_arg *arg);

void op_download_buffer_async(char *send_buffer_device, char *send_buffer_host, unsigned size_send);
void op_upload_buffer_async  (char *recv_buffer_device, char *recv_buffer_host, unsigned size_recv);
void op_download_buffer_sync();
void op_scatter_sync();

#endif /* __OP_MPI_CORE_H */
