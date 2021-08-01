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

#ifndef __OP_MPI_SLOPE_CORE_H
#define __OP_MPI_SLOPE_CORE_H

#ifdef SLOPE
    #define op_halo_create(x) op_halo_create_slope(x)
    #define op_halo_destroy() op_halo_destroy_slope()
#endif

void op_halo_create_slope();
void op_halo_destroy_slope();
void op_single_halo_destroy(halo_list* h_list);
halo_list merge_halo_lists(halo_list h_list1, halo_list h_list2, int comm_size, int my_rank);

int inverse_map_values(int* map_values, int map_dim, int map_size, int** inv_map_out, int** inv_offset_out);
int inverse_map(op_map map, int** inv_map_out, int** inv_offset_out);
int* get_map_values_from_elements(op_map map, int* indices, int indices_count);
halo_list* extend_halo_list(halo_list* h_list, int my_rank, int comm_size);

void step5_1(int **part_range, int my_rank, int comm_size);
void step7_1(int **part_range, int my_rank, int comm_size);
void step7_2(int **part_range, int my_rank, int comm_size);
void step7_3(int **part_range, int my_rank, int comm_size);
void step8_1(int **part_range, int my_rank, int comm_size);

int get_nonexec_size(op_set set, int* to_sets, int* to_set_to_exec_max, int* to_set_to_nonexec_max);
int get_exec_size(op_set set, int* to_sets, int* to_set_to_core_max, int* to_set_to_exec_max);
int get_core_size(op_set set, int* to_sets, int* to_set_to_core_max);
void calculate_max_values(op_set from_set, op_set to_set, int map_dim, int* map_values,
int* to_sets, int* to_set_to_core_max, int* to_set_to_exec_max, int* to_set_to_nonexec_max, int my_rank);
int find_element_in(int* arr, int element);
int get_max_value(int* arr, int from, int to);

#endif /* __OP_MPI_SLOPE_CORE_H */
