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

#ifndef __OP_LIB_MPI_H
#define __OP_LIB_MPI_H

/*
 * written by: Gihan R. Mudalige, 01-03-2011
 */

#include <op_lib_core.h>
#include <op_mpi_core.h>

/** extern variables for halo creation and exchange**/
extern MPI_Comm OP_MPI_WORLD;
extern MPI_Comm OP_MPI_GLOBAL;

extern halo_list *OP_export_exec_list; // EEH list
extern halo_list *OP_import_exec_list; // IEH list

extern halo_list *OP_import_nonexec_list; // INH list
extern halo_list *OP_export_nonexec_list; // ENH list

extern halo_list* OP_aug_export_exec_lists[OP_NHALOS_MAX];
extern halo_list* OP_aug_import_exec_lists[OP_NHALOS_MAX];

extern halo_list* OP_aug_export_nonexec_lists[OP_NHALOS_MAX];
extern halo_list* OP_aug_import_nonexec_lists[OP_NHALOS_MAX];

extern halo_list *OP_merged_import_exec_list;
extern halo_list *OP_merged_export_exec_list;

extern halo_list *OP_merged_import_nonexec_list;
extern halo_list *OP_merged_export_nonexec_list;

extern halo_list *OP_merged_import_exec_nonexec_list;
extern halo_list *OP_merged_export_exec_nonexec_list;

extern MPI_Request *grp_send_requests;
extern MPI_Request *grp_recv_requests;

extern char *grp_send_buffer;
extern char *grp_recv_buffer;

extern char *grp_send_buffer_h;
extern char *grp_recv_buffer_h;

extern char *grp_send_buffer_d;
extern char *grp_recv_buffer_d;

extern int grp_tag;

extern int*** OP_aug_map_ptr_list;
extern int** OP_map_ptr_list;

extern int OP_part_index;
extern part *OP_part_list;
extern int **orig_part_range;

/** variables for partial halo exchanges **/
extern int *OP_map_partial_exchange;
extern halo_list *OP_import_nonexec_permap;
extern halo_list *OP_export_nonexec_permap;

#ifdef __cplusplus
extern "C" {
#endif

/** Gather halo data in buffer on the device **/
void gather_data_to_buffer(op_arg arg, halo_list exp_exec_list,
                           halo_list exp_nonexec_list, int my_rank);
void gather_data_to_buffer_chained(int nargs, op_arg *args, int exec_flag, 
                                    int exp_rank_count, int* buf_pos, int* send_sizes, int* total_buf_size, int my_rank);
void gather_data_to_buffer_partial(op_arg arg, halo_list exp_nonexec_list);
void scatter_data_from_buffer(op_arg arg);
void scatter_data_from_buffer_partial(op_arg arg);

void scatter_data_from_buffer_ptr_cuda_chained(op_arg* args, int nargs, int rank_count, char *buffer, int exec_flag, int my_rank);

op_set op_decl_set_hdf5(char const *file, char const *name);
op_map op_decl_map_hdf5(op_set from, op_set to, int dim, char const *file,
                        char const *name);
op_dat op_decl_dat_hdf5(op_set set, int dim, char const *type, char const *file,
                        char const *name);

void op_get_const_hdf5(char const *name, int dim, char const *type,
                       char *const_data, char const *file_name);

void op_dump_to_hdf5(char const *file_name);
void op_write_const_hdf5(char const *name, int dim, char const *type,
                         char *const_data, char const *file_name);

int op_mpi_add_nhalos(op_halo_info halo_info, int nhalos);
int op_mpi_add_nhalos_set(op_set set, int nhalos);
int op_mpi_add_nhalos_map(op_map map, int nhalos);
int op_mpi_add_nhalos_map_calc(op_map map, int nhalos);
int op_mpi_add_nhalos_dat(op_dat dat, int nhalos);
int get_nhalos(op_arg *arg);
int is_arg_valid(op_arg* arg, int exec_flag, int dirtybit_val);
int get_dirty_args(int nargs, op_arg *args, int exec_flag, op_arg* dirty_args, int dirtybit_val);
int is_nonexec_halo_required(op_arg *arg, int nhalos, int halo_id);
int are_dirtybits_clear(op_arg *arg);
void unset_dirtybit(op_arg *arg);

#ifdef __cplusplus
}
#endif

#endif /* __OP_LIB_MPI_H */
