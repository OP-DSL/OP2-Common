/*
  Open source copyright declaration based on BSD open source template:
  http://www.opensource.org/licenses/bsd-license.php

* Copyright (c) 2009, Mike Giles
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
 * op_mpi_core.h
 * 
 * Headder file for the OP2 Distributed memory (MPI) halo creation, 
 * halo exchange and support utility routines/functions 
 *
 * written by: Gihan R. Mudalige, (Started 01-03-2011)
 */

#include <mpi.h>
 
/** Define the root MPI process **/
#ifdef MPI_ROOT
#undef MPI_ROOT
#endif
#define MPI_ROOT 0



/*******************************************************************************
* MPI halo list data type
*******************************************************************************/
typedef struct {
 op_set set;        //set related to this list
 int    size;       //number of elements in this list                                
 int    *ranks;     //MPI ranks to be exported to or imported from
 int    ranks_size; //number of MPI neighbors 
                    //to be exported to or imported from
 int    *disps;     //displacements for the starting point of each 
                    //rank's element list 
 int    *sizes;     //number of elements exported to or imported 
                    //from each ranks
 int    *list;      //the list of all elements 
} halo_list_core;

typedef halo_list_core * halo_list;



/*******************************************************************************
* Data structures related to MPI level partitioning
*******************************************************************************/
//struct to hold the partition information for each set
typedef struct
{
  op_set set;   //set to which this partition info blongs to 
  int *g_index; //global index of each element held in 
                //this MPI process
  int *elem_part;//partition to which each element belongs
  int is_partitioned; //indicates if this set is partitioned 
                      //1 if partitioned 0 if not
} part_core;

typedef part_core *part;



/*******************************************************************************
* Data Type to hold MPI performance measures
*******************************************************************************/
typedef struct 
{
  char const  *name;   // name of kernel 
  double      time;    //total time spent in this 
                       //kernel (compute+comm-overlapping)
  int         count;   //number of times this kernel is called
  int*        op_dat_indices;  //array to hold op_dat index of 
                               //each op_dat used in MPI halo 
                               //exports for this kernel
  int         num_indices; //number of op_dat indices
  int*        tot_count;   //total number of times this op_dat was 
                           //halo exported within this kernel
  int*        tot_bytes;   //total number of bytes halo exported 
                           //for this op_dat in this kernel  
} op_mpi_kernel;



/*******************************************************************************
* Buffer struct used in non-blocking mpi halo sends/receives
*******************************************************************************/
typedef struct {
 int         dat_index;    //index of the op_dat to which this buffer belongs
 char        *buf_exec;    //buffer holding exec halo to be exported;
 char        *buf_nonexec; //buffer holding nonexec halo to be exported;
 MPI_Request *s_req;       //pointed to hold the MPI_Reqest for sends
 MPI_Request *r_req;       //pointed to hold the MPI_Reqest for receives
 int         s_num_req;    //number of send MPI_Reqests in flight at a given 
                           //time for this op_dat
 int         r_num_req;    //number of receive MPI_Reqests in flight at a given  
                           //time for this op_dat
} op_mpi_buffer_core;

typedef op_mpi_buffer_core *op_mpi_buffer;


/** extern variables **/
extern int OP_part_index;
extern part *OP_part_list;
extern int** orig_part_range;




/*******************************************************************************
* Utility function prototypes
*******************************************************************************/
void decl_partition(op_set set, int* g_index, int* partition);

void get_part_range(int** part_range, int my_rank, int comm_size, MPI_Comm Comm);

int get_partition(int global_index, int* part_range, int* local_index, int comm_size);

int get_global_index(int local_index, int partition, int* part_range, int comm_size);

void find_neighbors_set(halo_list List, int* neighbors, int* sizes, 
	int* ranks_size, int my_rank, int comm_size, MPI_Comm Comm);

void create_list(int* list, int* ranks, int* disps, int* sizes, int* ranks_size, 
    int* total, int* temp_list, int size, int comm_size, int my_rank);

void create_export_list(op_set set, int* temp_list, halo_list h_list, int size, 
    int comm_size, int my_rank);

void create_import_list(op_set set, int* temp_list, halo_list h_list, int total_size, 
    int* ranks, int* sizes, int ranks_size, int comm_size, int my_rank);

int is_onto_map(op_map map);



/*******************************************************************************
* Core MPI lib function prototypes
*******************************************************************************/
void op_halo_create();

void op_halo_destroy();

int exchange_halo(op_arg arg);

void wait_all(op_arg arg);

void set_dirtybit(op_arg arg);

op_dat op_mpi_get_data(op_dat dat);

void global_reduce(op_arg* arg);

void op_mpi_timing_output();

void op_mpi_exit();

int op_mpi_perf_time(const char* name, double time);

void op_mpi_perf_comm(int kernel_index, op_arg arg);

void print_dat_tofile(op_dat dat, const char *file_name);

void print_dat_tobinfile(op_dat dat, const char *file_name);

void reset_halo(op_arg arg);
 
