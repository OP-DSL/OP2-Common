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


#include <op_lib_core.h>
#include <op_lib_c.h>
#include <op_util.h>

//mpi header
#include <mpi.h>

#include <op_mpi_core.h>

//
//MPI Communicator for halo creation and exchange
//

MPI_Comm OP_MPI_WORLD;

//
//MPI Halo related global variables
//

halo_list *OP_export_exec_list;//EEH list
halo_list *OP_import_exec_list;//IEH list

halo_list *OP_import_nonexec_list;//INH list
halo_list *OP_export_nonexec_list;//ENH list

//
//global array to hold dirty_bits for op_dats
//


/*table holding MPI performance of each loop
  (accessed via a hash of loop name) */

op_mpi_kernel op_mpi_kernel_tab[HASHSIZE];


//
//global variables to hold partition information on an MPI rank
//

int OP_part_index = 0;
part *OP_part_list;

//
//Save original partition ranges
//

int** orig_part_range = NULL;

/*******************************************************************************
 * Routine to declare partition information for a given set
 *******************************************************************************/

void decl_partition(op_set set, int* g_index, int* partition)
{
  part p = (part) xmalloc(sizeof(part_core));
  p->set = set;
  p->g_index = g_index;
  p->elem_part = partition;
  p->is_partitioned = 0;
  OP_part_list[set->index] = p;
  OP_part_index++;
}

/*******************************************************************************
 * Routine to get partition range on all mpi ranks for all sets
 *******************************************************************************/

void get_part_range(int** part_range, int my_rank, int comm_size, MPI_Comm Comm)
{
  (void)my_rank;
  for(int s=0; s<OP_set_index; s++) {
    op_set set=OP_set_list[s];

    int* sizes = (int *)xmalloc(sizeof(int)*comm_size);
    MPI_Allgather(&set->size, 1, MPI_INT, sizes, 1, MPI_INT, Comm);

    part_range[set->index] = (int *)xmalloc(2*comm_size*sizeof(int));

    int disp = 0;
    for(int i = 0; i<comm_size; i++){
      part_range[set->index][2*i] = disp;
      disp = disp + sizes[i] - 1;
      part_range[set->index][2*i+1] = disp;
      disp++;
#ifdef DEBUG
      if(my_rank == MPI_ROOT)
        printf("range of %10s in rank %d: %d-%d\n",set->name,i,
            part_range[set->index][2*i], part_range[set->index][2*i+1]);
#endif
    }
    free(sizes);
  }
}

/*******************************************************************************
 * Routine to get partition (i.e. mpi rank) where global_index is located and
 * its local index
 *******************************************************************************/

int get_partition(int global_index, int* part_range, int* local_index,
                  int comm_size)
{
  for(int i = 0; i<comm_size; i++)
  {
    if (global_index >= part_range[2*i] &&
        global_index <= part_range[2*i+1])
    {
      *local_index = global_index -  part_range[2*i];
      return i;
    }
  }
  return 0;
}

/*******************************************************************************
 * Routine to convert a local index in to a global index
 *******************************************************************************/

int get_global_index(int local_index, int partition, int* part_range,
                     int comm_size)
{
  (void)comm_size;
  int g_index = part_range[2*partition]+local_index;
#ifdef DEBUG
  if(g_index > part_range[2*(comm_size-1)+1])
    printf("Global index larger than set size\n");
#endif
  return g_index;
}

/*******************************************************************************
 * Routine to find the MPI neighbors given a halo list
 *******************************************************************************/

void find_neighbors_set(halo_list List, int* neighbors, int* sizes,
                        int* ranks_size, int my_rank, int comm_size, MPI_Comm Comm)
{
  int* temp = (int*)xmalloc(comm_size*sizeof(int));
  int* r_temp = (int*)xmalloc(comm_size*comm_size*sizeof(int));

  for(int r = 0;r<comm_size*comm_size;r++)r_temp[r] = -99;
  for(int r = 0;r<comm_size;r++)temp[r] = -99;

  int n = 0;

  for(int r =0; r<comm_size; r++)
  {
    if(List->ranks[r]>=0) temp[List->ranks[r]] = List->sizes[r];
  }

  MPI_Allgather( temp, comm_size, MPI_INT, r_temp,
      comm_size,MPI_INT,Comm);

  for(int i=0; i<comm_size; i++)
  {
    if(i != my_rank)
    {
      if( r_temp[i*comm_size+my_rank] > 0)
      {
        neighbors[n] = i;
        sizes[n] = r_temp[i*comm_size+my_rank];
        n++;
      }
    }
  }
  *ranks_size = n;
  free(temp);free(r_temp);
}

/*******************************************************************************
 * Routine to create a generic halo list
 * (used in both import and export list creation)
 *******************************************************************************/

void create_list(int* list, int* ranks, int* disps, int* sizes, int* ranks_size,
    int* total, int* temp_list, int size, int comm_size, int my_rank)
{
  (void)my_rank;
  int index = 0;
  int total_size = 0;
  if(size < 0)printf("problem\n");
  //negative values set as an initialisation
  for(int r = 0;r<comm_size;r++)
  {
    disps[r] = ranks[r] = -99;
    sizes[r] = 0;
  }
  for(int r = 0;r<comm_size;r++)
  {
    sizes[index] = disps[index] = 0;

    int* temp = (int *)xmalloc((size/2)*sizeof(int));
    for(int i = 0;i<size;i=i+2)
    {
      if(temp_list[i]==r)
        temp[sizes[index]++] = temp_list[i+1];
    }
    if(sizes[index]>0)
    {
      ranks[index] = r;
      //sort temp,
      quickSort(temp,0,sizes[index]-1);
      //eliminate duplicates in temp
      sizes[index] = removeDups(temp, sizes[index]);
      total_size = total_size + sizes[index];

      if(index > 0)
        disps[index] = disps[index-1] +  sizes[index-1];
      //add to end of exp_list
      for(int e = 0;e<sizes[index];e++)
        list[disps[index]+e] = temp[e];

      index++;
    }
    free(temp);
  }

  *total = total_size;
  *ranks_size = index;
}

/*******************************************************************************
 * Routine to create an export list
 *******************************************************************************/

void create_export_list(op_set set, int* temp_list, halo_list h_list, int size,
    int comm_size, int my_rank)
{
  int* ranks = (int *)xmalloc(comm_size*sizeof(int));
  int* list = (int *)xmalloc((size/2)*sizeof(int));
  int* disps = (int *)xmalloc(comm_size*sizeof(int));
  int* sizes = (int *)xmalloc(comm_size*sizeof(int));

  int ranks_size = 0;
  int total_size = 0;

  create_list(list, ranks, disps, sizes, &ranks_size, &total_size,
      temp_list, size, comm_size, my_rank);


  h_list->set = set;
  h_list->size = total_size;
  h_list->ranks = ranks;
  h_list->ranks_size = ranks_size;
  h_list->disps = disps;
  h_list->sizes = sizes;
  h_list->list = list;
}

/*******************************************************************************
 * Routine to create an import list
 *******************************************************************************/

void create_import_list(op_set set, int* temp_list, halo_list h_list,
                        int total_size, int* ranks, int* sizes, int ranks_size,
                        int comm_size, int my_rank)
{
  (void)my_rank;
  int* disps = (int *)xmalloc(comm_size*sizeof(int));
  disps[0] = 0;
  for(int i=0; i<ranks_size; i++)
  {
    if(i>0)disps[i] = disps[i-1]+sizes[i-1];
  }

  h_list->set = set;
  h_list->size = total_size;
  h_list->ranks = ranks;
  h_list->ranks_size = ranks_size;
  h_list->disps = disps;
  h_list->sizes = sizes;
  h_list->list = temp_list;
}

/*******************************************************************************
 * Routine to create an nonexec-import list (only a wrapper)
 *******************************************************************************/

static void create_nonexec_import_list(op_set set, int* temp_list, halo_list h_list,
                                       int size, int comm_size, int my_rank)
{
  create_export_list(set, temp_list, h_list, size, comm_size, my_rank);
}

/*******************************************************************************
 * Routine to create an nonexec-export list (only a wrapper)
 *******************************************************************************/

static void create_nonexec_export_list(op_set set, int* temp_list, halo_list h_list,
                                       int total_size, int* ranks, int* sizes,
                                       int ranks_size, int comm_size, int my_rank)
{
  create_import_list(set, temp_list, h_list, total_size, ranks, sizes,
      ranks_size, comm_size, my_rank);
}

/*******************************************************************************
 * Check if a given op_map is an on-to map from the from-set to the to_set
 *******************************************************************************/

int is_onto_map(op_map map)
{
  //create new communicator
  int my_rank, comm_size;
  MPI_Comm OP_CHECK_WORLD;
  MPI_Comm_dup(MPI_COMM_WORLD, &OP_CHECK_WORLD);
  MPI_Comm_rank(OP_CHECK_WORLD, &my_rank);
  MPI_Comm_size(OP_CHECK_WORLD, &comm_size);

  // Compute global partition range information for each set
  int** part_range = (int **)xmalloc(OP_set_index*sizeof(int*));
  get_part_range(part_range,my_rank,comm_size, OP_CHECK_WORLD);

  //mak a copy of the to-set elements of the map
  int* to_elem_copy = (int *)xmalloc(map->from->size*map->dim*sizeof(int));
  memcpy(to_elem_copy,(void *)map->map,map->from->size*map->dim*sizeof(int));

  //sort and remove duplicates from to_elem_copy
  quickSort(to_elem_copy, 0, map->from->size*map->dim - 1);
  int to_elem_copy_size = removeDups(to_elem_copy, map->from->size*map->dim);
  to_elem_copy = (int *)xrealloc(to_elem_copy,to_elem_copy_size*sizeof(int));

  //go through the to-set element range that this local MPI process holds
  //and collect the to-set elements not found in to_elem_copy
  int cap = 100; int count = 0;
  int* not_found = (int *)xmalloc(sizeof(int)*cap);
  for(int i = 0; i < map->to->size; i++)
  {
    int g_index = get_global_index(i, my_rank,
        part_range[map->to->index], comm_size);
    if(binary_search(to_elem_copy, i, 0, to_elem_copy_size-1) < 0)
    {
      //add to not_found list
      if(count >= cap)
      {
        cap = cap*2;
        not_found = (int *)xrealloc(not_found, cap*sizeof(int));
      }
      not_found[count++] = g_index;
    }
  }

  //
  //allreduce this not_found to form a global_not_found list
  //
  int recv_count[comm_size];
  MPI_Allgather(&count, 1, MPI_INT, recv_count, 1, MPI_INT, OP_CHECK_WORLD);

  //discover global size of the not_found_list
  int g_count = 0;
  for(int i = 0; i< comm_size; i++)g_count += recv_count[i];

  //prepare for an allgatherv
  int disp = 0;
  int* displs = (int *)xmalloc(comm_size*sizeof(int));
  for(int i = 0; i<comm_size; i++)
  {
    displs[i] =   disp;
    disp = disp + recv_count[i];
  }

  //allocate memory to hold the global_not_found list
  int *global_not_found = (int *)xmalloc(sizeof(int)*g_count);

  MPI_Allgatherv(not_found,count,MPI_INT, global_not_found,recv_count,displs,
      MPI_INT, OP_CHECK_WORLD);
  free(not_found);free(displs);

  //sort and remove duplicates of the global_not_found list
  if(g_count > 0)
  {
    quickSort(global_not_found, 0, g_count-1);
    g_count = removeDups(global_not_found, g_count);
    global_not_found = (int *)xrealloc(global_not_found, g_count*sizeof(int));
  }
  else
  {
    //nothing in the global_not_found list .. i.e. this is an on to map
    free(global_not_found);free(to_elem_copy);//free(displs);
    for(int i = 0; i<OP_set_index; i++)free(part_range[i]);free(part_range);
    return 1;
  }

  //see if any element in the global_not_found is found in the local map-copy
  //and add it to a "found" list
  cap = 100; count = 0;
  int* found = (int *)xmalloc(sizeof(int)*cap);
  for(int i = 0; i < g_count; i++)
  {
    if(binary_search(to_elem_copy, global_not_found[i], 0,
          to_elem_copy_size-1) >= 0)
    {
      //add to found list
      if(count >= cap)
      {
        cap = cap*2;
        found = (int *)xrealloc(found, cap*sizeof(int));
      }
      found[count++] = global_not_found[i];
    }
  }
  free(global_not_found);

  //
  //allreduce the "found" elements to form a global_found list
  //
  //recv_count[comm_size];
  MPI_Allgather(&count, 1, MPI_INT, recv_count, 1, MPI_INT, OP_CHECK_WORLD);

  //discover global size of the found_list
  int g_found_count = 0;
  for(int i = 0; i< comm_size; i++)g_found_count += recv_count[i];

  //prepare for an allgatherv
  disp = 0;
  displs = (int *)xmalloc(comm_size*sizeof(int));
  for(int i = 0; i<comm_size; i++)
  {
    displs[i] =   disp;
    disp = disp + recv_count[i];
  }

  //allocate memory to hold the global_found list
  int *global_found = (int *)xmalloc(sizeof(int)*g_found_count);

  MPI_Allgatherv(found,count,MPI_INT, global_found,recv_count,displs,
      MPI_INT, OP_CHECK_WORLD);
  free(found);

  //sort global_found list and remove duplicates
  if(g_count > 0)
  {
    quickSort(global_found, 0, g_found_count-1);
    g_found_count = removeDups(global_found, g_found_count);
    global_found = (int *)xrealloc(global_found, g_found_count*sizeof(int));
  }

  //if the global_found list size is smaller than the globla_not_found list size
  //then map is not an on_to map
  int result = 0;
  if(g_found_count == g_count)
    result = 1;

  free(global_found);free(displs);
  for(int i = 0; i<OP_set_index; i++)free(part_range[i]);free(part_range);
  MPI_Comm_free(&OP_CHECK_WORLD);
  free(to_elem_copy);

  return result;
}

/*******************************************************************************
 * Main MPI halo creation routine
 *******************************************************************************/

void op_halo_create()
{
  //declare timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  double time;
  double max_time;
  op_timers(&cpu_t1, &wall_t1); //timer start for list create

  //create new communicator for OP mpi operation
  int my_rank, comm_size;
  MPI_Comm_dup(MPI_COMM_WORLD, &OP_MPI_WORLD);
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_WORLD, &comm_size);

  /* Compute global partition range information for each set*/
  int** part_range = (int **)xmalloc(OP_set_index*sizeof(int*));
  get_part_range(part_range,my_rank,comm_size, OP_MPI_WORLD);

  //save this partition range information if it is not already saved during
  //a call to some partitioning routine
  if(orig_part_range == NULL)
  {
    orig_part_range = (int **)xmalloc(OP_set_index*sizeof(int*));
    for(int s = 0; s< OP_set_index; s++)
    {
      op_set set=OP_set_list[s];
      orig_part_range[set->index] = (int *)xmalloc(2*comm_size*sizeof(int));
      for(int j = 0; j<comm_size; j++){
        orig_part_range[set->index][2*j] = part_range[set->index][2*j];
        orig_part_range[set->index][2*j+1] = part_range[set->index][2*j+1];
      }
    }
  }

  OP_export_exec_list = (halo_list *)xmalloc(OP_set_index*sizeof(halo_list));

  /*----- STEP 1 - Construct export lists for execute set elements and related
    mapping table entries -----*/

  //declare temporaty scratch variables to hold set export lists and mapping
  //table export lists
  int s_i;
  int* set_list;

  int cap_s = 1000; //keep track of the temp array capacities


  for(int s=0; s<OP_set_index; s++){ //for each set
    op_set set=OP_set_list[s];

    //create a temporaty scratch space to hold export list for this set
    s_i = 0;cap_s = 1000;
    set_list = (int *)xmalloc(cap_s*sizeof(int));

    for(int e=0; e<set->size;e++){//for each elment of this set
      for(int m=0; m<OP_map_index; m++) { //for each maping table
        op_map map=OP_map_list[m];

        if(compare_sets(map->from,set)==1) //need to select mappings
          //FROM this set
        {
          int part, local_index;
          for(int j=0; j<map->dim; j++) { //for each element
            //pointed at by this entry
            part = get_partition(map->map[e*map->dim+j],
                part_range[map->to->index],&local_index,comm_size);
            if(s_i>=cap_s)
            {
              cap_s = cap_s*2;
              set_list = (int *)xrealloc(set_list,cap_s*sizeof(int));
            }

            if(part != my_rank){
              set_list[s_i++] = part; //add to set export list
              set_list[s_i++] = e;
            }
          }
        }
      }
    }

    //create set export list
    //printf("creating set export list for set %10s of size %d\n",
    //set->name,s_i);
    halo_list h_list= (halo_list)xmalloc(sizeof(halo_list_core));
    create_export_list(set,set_list, h_list, s_i, comm_size, my_rank);
    OP_export_exec_list[set->index] = h_list;
    free(set_list);//free temp list
  }

  /*---- STEP 2 - construct import lists for mappings and execute sets------*/

  OP_import_exec_list = (halo_list *)xmalloc(OP_set_index*sizeof(halo_list));

  int *neighbors, *sizes;
  int ranks_size;

  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];

    //-----Discover neighbors-----
    ranks_size = 0;
    neighbors = (int *)xmalloc(comm_size*sizeof(int));
    sizes = (int *)xmalloc(comm_size*sizeof(int));

    halo_list list = OP_export_exec_list[set->index];

    find_neighbors_set(list,neighbors,sizes,&ranks_size,my_rank,
        comm_size, OP_MPI_WORLD);
    MPI_Request request_send[list->ranks_size];

    int* rbuf, cap = 0, index = 0;

    for(int i=0; i<list->ranks_size; i++) {
      //printf("export from %d to %d set %10s, list of size %d \n",
      //my_rank,list->ranks[i],set->name,list->sizes[i]);
      int* sbuf = &list->list[list->disps[i]];
      MPI_Isend( sbuf,  list->sizes[i],  MPI_INT, list->ranks[i], s,
          OP_MPI_WORLD, &request_send[i] );
    }

    for(int i=0; i< ranks_size; i++) cap = cap + sizes[i];
    int* temp = (int *)xmalloc(cap*sizeof(int));

    //import this list from those neighbors
    for(int i=0; i<ranks_size; i++) {
      //printf("import from %d to %d set %10s, list of size %d\n",
      //neighbors[i], my_rank, set->name, sizes[i]);
      rbuf = (int *)xmalloc(sizes[i]*sizeof(int));
      MPI_Recv(rbuf, sizes[i], MPI_INT, neighbors[i],s, OP_MPI_WORLD,
          MPI_STATUSES_IGNORE );
      memcpy(&temp[index],(void *)&rbuf[0],sizes[i]*sizeof(int));
      index = index + sizes[i];
      free(rbuf);
    }

    MPI_Waitall(list->ranks_size,request_send, MPI_STATUSES_IGNORE );

    //create import lists
    //printf("creating importlist with number of neighbors %d\n",ranks_size);
    halo_list h_list= (halo_list)xmalloc(sizeof(halo_list_core));
    create_import_list(set, temp, h_list, index,neighbors, sizes,
        ranks_size, comm_size, my_rank);
    OP_import_exec_list[set->index] = h_list;
  }

  /*--STEP 3 -Exchange mapping table entries using the import/export lists--*/

  for(int m=0; m<OP_map_index; m++) { //for each maping table
    op_map map=OP_map_list[m];
    halo_list i_list = OP_import_exec_list[map->from->index];
    halo_list e_list = OP_export_exec_list[map->from->index];

    MPI_Request request_send[e_list->ranks_size];

    //prepare bits of the mapping tables to be exported
    int** sbuf = (int **)xmalloc(e_list->ranks_size*sizeof(int *));

    for(int i=0; i < e_list->ranks_size; i++) {
      sbuf[i] = (int *)xmalloc(e_list->sizes[i]*map->dim*sizeof(int));
      for(int j = 0; j < e_list->sizes[i]; j++)
      {
        for(int p = 0; p < map->dim; p++)
        {
          sbuf[i][j*map->dim+p] =
            map->map[map->dim*(e_list->list[e_list->disps[i]+j])+p];
        }
      }
      //printf("\n export from %d to %d map %10s, number of elements of size %d | sending:\n ",
      //    my_rank,e_list.ranks[i],map.name,e_list.sizes[i]);
      MPI_Isend(sbuf[i],  map->dim*e_list->sizes[i],  MPI_INT,
          e_list->ranks[i], m, OP_MPI_WORLD, &request_send[i]);
    }

    //prepare space for the incomming mapping tables - realloc each
    //mapping tables in each mpi process
    OP_map_list[map->index]->map = (int *)xrealloc(OP_map_list[map->index]->map,
        (map->dim*(map->from->size+i_list->size))*sizeof(int));

    int init = map->dim*(map->from->size);
    for(int i=0; i<i_list->ranks_size; i++) {
      //printf("\n imported on to %d map %10s, number of elements of size %d | recieving: ",
      //    my_rank, map->name, i_list->size);
      MPI_Recv(&(OP_map_list[map->index]->
            map[init+i_list->disps[i]*map->dim]),
          map->dim*i_list->sizes[i], MPI_INT, i_list->ranks[i], m,
          OP_MPI_WORLD, MPI_STATUSES_IGNORE);
    }

    MPI_Waitall(e_list->ranks_size,request_send, MPI_STATUSES_IGNORE );
    for(int i=0; i < e_list->ranks_size; i++) free(sbuf[i]); free(sbuf);
  }

  /*-- STEP 4 - Create import lists for non-execute set elements using mapping
    table entries including the additional mapping table entries --*/

  OP_import_nonexec_list = (halo_list *)xmalloc(OP_set_index*sizeof(halo_list));
  OP_export_nonexec_list = (halo_list *)xmalloc(OP_set_index*sizeof(halo_list));

  //declare temporaty scratch variables to hold non-exec set export lists
  s_i = 0;
  set_list = NULL;
  cap_s = 1000; //keep track of the temp array capacity

  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];
    halo_list exec_set_list=OP_import_exec_list[set->index];

    //create a temporaty scratch space to hold nonexec export list for this set
    s_i = 0;
    set_list = (int *)xmalloc(cap_s*sizeof(int));

    for(int m=0; m<OP_map_index; m++) { //for each maping table
      op_map map=OP_map_list[m];
      halo_list exec_map_list=OP_import_exec_list[map->from->index];

      if(compare_sets(map->to,set)==1) //need to select mappings TO this set
      {
        //for each entry in this mapping table: original+execlist
        int len = map->from->size+exec_map_list->size;
        for(int e = 0; e<len; e++)
        {
          int part;
          int local_index;
          for(int j=0; j < map->dim; j++) { //for each element pointed
            //at by this entry
            part = get_partition(map->map[e*map->dim+j],
                part_range[map->to->index],&local_index,comm_size);

            if(s_i>=cap_s)
            {
              cap_s = cap_s*2;
              set_list = (int *)xrealloc(set_list,cap_s*sizeof(int));
            }

            if(part != my_rank)
            {
              int found = -1;
              //check in exec list
              int rank = binary_search(exec_set_list->ranks,
                  part, 0, exec_set_list->ranks_size-1);

              if(rank >= 0)
              {
                found = binary_search(exec_set_list->list,
                    local_index, exec_set_list->disps[rank],
                    exec_set_list->disps[rank]+
                    exec_set_list->sizes[rank]-1);
              }

              if(found < 0){
                // not in this partition and not found in
                //exec list
                //add to non-execute set_list
                set_list[s_i++] = part;
                set_list[s_i++] = local_index;
              }
            }
          }
        }
      }
    }

    //create non-exec set import list
    //printf("creating non-exec import list of size %d\n",s_i);
    halo_list h_list= (halo_list)xmalloc(sizeof(halo_list_core));
    create_nonexec_import_list(set,set_list, h_list, s_i, comm_size, my_rank);
    free(set_list);//free temp list
    OP_import_nonexec_list[set->index] = h_list;
  }


  /*----------- STEP 5 - construct non-execute set export lists -------------*/

  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];

    //-----Discover neighbors-----
    ranks_size = 0;
    neighbors = (int* )xmalloc(comm_size*sizeof(int));
    sizes = (int* )xmalloc(comm_size*sizeof(int));

    halo_list list=OP_import_nonexec_list[set->index];
    find_neighbors_set(list,neighbors,sizes,&ranks_size,my_rank,
        comm_size, OP_MPI_WORLD);

    MPI_Request request_send[list->ranks_size];
    int* rbuf, cap = 0, index = 0;

    for(int i=0; i<list->ranks_size; i++) {
      //printf("import to %d from %d set %10s, nonexec list of size %d | sending:\n",
      //    my_rank,list->ranks[i],set->name,list->sizes[i]);
      int* sbuf = &list->list[list->disps[i]];
      MPI_Isend( sbuf,  list->sizes[i],  MPI_INT, list->ranks[i], s,
          OP_MPI_WORLD, &request_send[i] );
    }

    for(int i=0; i< ranks_size; i++) cap = cap + sizes[i];
    int* temp = (int* )xmalloc(cap*sizeof(int));

    //export this list to those neighbors
    for(int i=0; i<ranks_size; i++) {
      //printf("export to %d from %d set %10s, list of size %d | recieving:\n",
      //    neighbors[i], my_rank, set->name, sizes[i]);
      rbuf = (int* )xmalloc(sizes[i]*sizeof(int));
      MPI_Recv(rbuf, sizes[i], MPI_INT, neighbors[i],s, OP_MPI_WORLD,
          MPI_STATUSES_IGNORE );
      memcpy(&temp[index],(void *)&rbuf[0],sizes[i]*sizeof(int));
      index = index + sizes[i];
      free(rbuf);
    }

    MPI_Waitall(list->ranks_size,request_send, MPI_STATUSES_IGNORE );

    //create import lists
    //printf("creating nonexec set export list with number of neighbors %d\n",ranks_size);
    halo_list h_list= (halo_list)xmalloc(sizeof(halo_list_core));
    create_nonexec_export_list(set, temp, h_list, index, neighbors, sizes,
        ranks_size, comm_size, my_rank);
    OP_export_nonexec_list[set->index] = h_list;
  }


  /*-STEP 6 - Exchange execute set elements/data using the import/export lists--*/

  for(int s=0; s<OP_set_index; s++){ //for each set
    op_set set=OP_set_list[s];
    halo_list i_list = OP_import_exec_list[set->index];
    halo_list e_list = OP_export_exec_list[set->index];

    //for each data array
    op_dat_entry *item; int d = -1; //d is just simply the tag for mpi comms
    TAILQ_FOREACH(item, &OP_dat_list, entries) {
      d++; //increase tag to do mpi comm for the next op_dat
      op_dat dat = item->dat;

      if(compare_sets(set,dat->set)==1)//if this data array is defined on this set
      {
        //printf("on rank %d, The data array is %10s\n",my_rank,dat->name);
        MPI_Request request_send[e_list->ranks_size];

        //prepare execute set element data to be exported
        char** sbuf = (char** )xmalloc(e_list->ranks_size*sizeof(char *));

        for(int i=0; i < e_list->ranks_size; i++) {
          sbuf[i] = (char *)xmalloc(e_list->sizes[i]*dat->size);
          for(int j = 0; j < e_list->sizes[i]; j++)
          {
            int set_elem_index = e_list->list[e_list->disps[i]+j];
            memcpy(&sbuf[i][j*dat->size],
                (void *)&dat->data[dat->size*(set_elem_index)],
                dat->size);
          }
          //printf("export from %d to %d data %10s, number of elements of size %d | sending:\n ",
          //    my_rank,e_list->ranks[i],dat->name,e_list->sizes[i]);
          MPI_Isend(sbuf[i],  dat->size*e_list->sizes[i],
              MPI_CHAR, e_list->ranks[i],
              d, OP_MPI_WORLD, &request_send[i]);
        }

        //prepare space for the incomming data - realloc each
        //data array in each mpi process
        dat->data = (char *)xrealloc(dat->data,(set->size+i_list->size)*dat->size);

        int init = set->size*dat->size;
        for(int i=0; i<i_list->ranks_size; i++) {
          MPI_Recv(&(dat->data[init+i_list->disps[i]*dat->size]),
              dat->size*i_list->sizes[i],
              MPI_CHAR, i_list->ranks[i], d,
              OP_MPI_WORLD, MPI_STATUSES_IGNORE);
        }

        MPI_Waitall(e_list->ranks_size,request_send,
            MPI_STATUSES_IGNORE );
        for(int i=0; i<e_list->ranks_size; i++) free(sbuf[i]);
        free(sbuf);
        //printf("imported on to %d data %10s, number of elements of size %d | recieving:\n ",
        //    my_rank, dat->name, i_list->size);
      }

    }
  }

  /*-STEP 7 - Exchange non-execute set elements/data using the import/export lists--*/

  for(int s=0; s<OP_set_index; s++){ //for each set
    op_set set=OP_set_list[s];
    halo_list i_list = OP_import_nonexec_list[set->index];
    halo_list e_list = OP_export_nonexec_list[set->index];

    //for each data array
    op_dat_entry *item; int d = -1; //d is just simply the tag for mpi comms
    TAILQ_FOREACH(item, &OP_dat_list, entries) {
      d++; //increase tag to do mpi comm for the next op_dat
      op_dat dat = item->dat;

      if(compare_sets(set,dat->set)==1)//if this data array is
        //defined on this set
      {
        //printf("on rank %d, The data array is %10s\n",my_rank,dat->name);
        MPI_Request request_send[e_list->ranks_size];

        //prepare non-execute set element data to be exported
        char** sbuf = (char** )xmalloc(e_list->ranks_size*sizeof(char *));

        for(int i=0; i < e_list->ranks_size; i++) {
          sbuf[i] = (char *)xmalloc(e_list->sizes[i]*dat->size);
          for(int j = 0; j < e_list->sizes[i]; j++)
          {
            int set_elem_index = e_list->list[e_list->disps[i]+j];
            memcpy(&sbuf[i][j*dat->size],
                (void *)&dat->data[dat->size*(set_elem_index)],dat->size);
          }
          MPI_Isend(sbuf[i],  dat->size*e_list->sizes[i],
              MPI_CHAR, e_list->ranks[i],
              d, OP_MPI_WORLD, &request_send[i]);
        }

        //prepare space for the incomming nonexec-data - realloc each
        //data array in each mpi process
        halo_list exec_i_list = OP_import_exec_list[set->index];

        dat->data = (char *)xrealloc(dat->data,
          (set->size+exec_i_list->size+i_list->size)*dat->size);

        int init = (set->size+exec_i_list->size)*dat->size;
        for(int i=0; i < i_list->ranks_size; i++) {
          MPI_Recv(&(dat->data[init+i_list->disps[i]*dat->size]),
              dat->size*i_list->sizes[i],
              MPI_CHAR, i_list->ranks[i], d,
              OP_MPI_WORLD, MPI_STATUSES_IGNORE);
        }

        MPI_Waitall(e_list->ranks_size,request_send, MPI_STATUSES_IGNORE );
        for(int i=0; i < e_list->ranks_size; i++) free(sbuf[i]); free(sbuf);
      }
    }
  }





  /*-STEP 8 ----------------- Renumber Mapping tables-----------------------*/

  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];

    for(int m=0; m<OP_map_index; m++) { //for each maping table
      op_map map=OP_map_list[m];

      if(compare_sets(map->to,set)==1) //need to select mappings TO this set
      {
        halo_list exec_set_list=OP_import_exec_list[set->index];
        halo_list nonexec_set_list=OP_import_nonexec_list[set->index];

        halo_list exec_map_list=OP_import_exec_list[map->from->index];

        //for each entry in this mapping table: original+execlist
        int len = map->from->size+exec_map_list->size;
        for(int e = 0; e < len; e++)
        {
          for(int j=0; j < map->dim; j++) { //for each element
            //pointed at by this entry
            int part;
            int local_index = 0;
            part = get_partition(map->map[e*map->dim+j],
                part_range[map->to->index],&local_index,comm_size);

            if(part == my_rank)
            {
              OP_map_list[map->index]->
                map[e*map->dim+j] = local_index;
            }
            else
            {
              int found = -1;
              //check in exec list
              int rank1 = binary_search(exec_set_list->ranks,
                  part, 0, exec_set_list->ranks_size-1);
              //check in nonexec list
              int rank2 = binary_search(nonexec_set_list->ranks,
                  part, 0, nonexec_set_list->ranks_size-1);

              if(rank1 >=0)
              {
                found = binary_search(exec_set_list->list,
                    local_index, exec_set_list->disps[rank1],
                    exec_set_list->disps[rank1]+
                    exec_set_list->sizes[rank1]-1);
                if(found>=0)
                {
                  OP_map_list[map->index]->map[e*map->dim+j] =
                    found + map->to->size ;
                }
              }

              if(rank2 >=0 && found <0)
              {
                found = binary_search(nonexec_set_list->list,
                    local_index, nonexec_set_list->disps[rank2],
                    nonexec_set_list->disps[rank2]+
                    nonexec_set_list->sizes[rank2]-1);
                if(found>=0)
                {
                  OP_map_list[map->index]->map[e*map->dim+j] =
                    found + set->size + exec_set_list->size;
                }
              }

              if(found < 0)
                printf("ERROR: Set %10s Element %d needed on rank %d \
                    from partition %d\n",
                    set->name, local_index, my_rank, part );
            }
          }
        }
      }
    }
  }



  /*-STEP 9 ---------------- Create MPI send Buffers-----------------------*/

  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;

    op_mpi_buffer mpi_buf= (op_mpi_buffer)xmalloc(sizeof(op_mpi_buffer_core));

    halo_list exec_e_list = OP_export_exec_list[dat->set->index];
    halo_list nonexec_e_list = OP_export_nonexec_list[dat->set->index];

    mpi_buf->buf_exec = (char *)xmalloc((exec_e_list->size)*dat->size);
    mpi_buf->buf_nonexec = (char *)xmalloc((nonexec_e_list->size)*dat->size);

    halo_list exec_i_list = OP_import_exec_list[dat->set->index];
    halo_list nonexec_i_list = OP_import_nonexec_list[dat->set->index];

    mpi_buf->s_req = (MPI_Request *)xmalloc(sizeof(MPI_Request)*
        (exec_e_list->ranks_size + nonexec_e_list->ranks_size));
    mpi_buf->r_req = (MPI_Request *)xmalloc(sizeof(MPI_Request)*
        (exec_i_list->ranks_size + nonexec_i_list->ranks_size));

    mpi_buf->s_num_req = 0;
    mpi_buf->r_num_req = 0;
    dat->mpi_buffer = mpi_buf;
  }


  //set dirty bits of all data arrays to 0
  //for each data array
  item = NULL;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    dat->dirtybit= 0;
  }


  /*-STEP 10 -------------------- Separate core elements------------------------*/

  int** core_elems = (int **)xmalloc(OP_set_index*sizeof(int *));
  int** exp_elems = (int **)xmalloc(OP_set_index*sizeof(int *));

  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];

    halo_list exec = OP_export_exec_list[set->index];
    halo_list nonexec = OP_export_nonexec_list[set->index];

    if(exec->size > 0)
    {
      exp_elems[set->index]= (int *)xmalloc(exec->size*sizeof(int));
      memcpy(exp_elems[set->index], exec->list, exec->size*sizeof(int));
      quickSort(exp_elems[set->index], 0, exec->size-1);

      int num_exp = removeDups(exp_elems[set->index], exec->size);
      core_elems[set->index] = (int *)xmalloc(set->size*sizeof(int ));
      int count = 0;
      for(int e=0; e < set->size;e++){//for each elment of this set

        if((binary_search(exp_elems[set->index], e, 0, num_exp-1) < 0))
        {
          core_elems[set->index][count++] = e;
        }
      }
      quickSort(core_elems[set->index], 0, count-1);

      if(count+num_exp != set->size) printf("sizes not equal\n");
      set->core_size = count;

      //for each data array defined on this set seperate its elements
      op_dat_entry *item;
      TAILQ_FOREACH(item, &OP_dat_list, entries) {
      op_dat dat = item->dat;

        if(compare_sets(set,dat->set)==1)//if this data array is
          //defined on this set
        {
          char* new_dat = (char* )xmalloc(set->size*dat->size);
          for(int i = 0; i<count; i++)
          {
            memcpy(&new_dat[i*dat->size],
                &dat->data[core_elems[set->index][i]*dat->size],
                dat->size);
          }
          for(int i = 0; i< num_exp; i++)
          {
            memcpy(&new_dat[(count+i)*dat->size],
                &dat->data[exp_elems[set->index][i]*dat->size],
                dat->size);
          }
          memcpy(&dat->data[0],&new_dat[0], set->size*dat->size);
          free(new_dat);
        }
      }

      //for each mapping defined from this set seperate its elements
      for(int m=0; m<OP_map_index; m++) { //for each set
        op_map map=OP_map_list[m];

        if(compare_sets(map->from,set)==1)//if this mapping is
          //defined from this set
        {
          int* new_map = (int *)xmalloc(set->size*map->dim*sizeof(int));
          for(int i = 0; i<count; i++)
          {
            memcpy(&new_map[i*map->dim],
                &map->map[core_elems[set->index][i]*map->dim],
                map->dim*sizeof(int));
          }
          for(int i = 0; i<num_exp; i++)
          {
            memcpy(&new_map[(count+i)*map->dim],
                &map->map[exp_elems[set->index][i]*map->dim],
                map->dim*sizeof(int));
          }
          memcpy(&map->map[0],&new_map[0],
              set->size*map->dim*sizeof(int));
          free(new_map);
        }
      }

      for(int i  = 0; i< exec->size;i++)
      {
        int index = binary_search(exp_elems[set->index],
            exec->list[i], 0, num_exp-1);
        if(index < 0)
          printf("Problem in seperating core elements - exec list\n");
        else exec->list[i] = count + index;
      }

      for(int i  = 0; i< nonexec->size;i++)
      {
        int index = binary_search(core_elems[set->index],
            nonexec->list[i], 0, count-1);
        if (index < 0)
        {
          index = binary_search(exp_elems[set->index],
              nonexec->list[i], 0, num_exp-1);
          if(index < 0)
            printf("Problem in seperating core elements - nonexec list\n");
          else nonexec->list[i] = count + index;
        }
        else nonexec->list[i] = index;
      }
    }
    else
    {
      core_elems[set->index] = (int *)xmalloc(set->size*sizeof(int ));
      exp_elems[set->index] = (int *)xmalloc(0*sizeof(int ));
      for(int e=0; e < set->size;e++){//for each elment of this set
        core_elems[set->index][e] = e;
      }
      set->core_size = set->size;
    }
  }

  //now need to renumber mapping tables as the elements are seperated
  for(int m=0; m<OP_map_index; m++) { //for each set
    op_map map=OP_map_list[m];

    halo_list exec_map_list=OP_import_exec_list[map->from->index];
    //for each entry in this mapping table: original+execlist
    int len = map->from->size+exec_map_list->size;
    for(int e = 0; e < len; e++)
    {
      for(int j=0; j < map->dim; j++) { //for each element pointed
        //at by this entry
        if(map->map[e*map->dim+j] < map->to->size)
        {
          int index = binary_search(core_elems[map->to->index],
              map->map[e*map->dim+j],
              0, map->to->core_size-1);
          if(index < 0)
          {
            index = binary_search(exp_elems[map->to->index],
                map->map[e*map->dim+j],
                0, (map->to->size) - (map->to->core_size) -1);
            if(index < 0)
              printf("Problem in seperating core elements - \
                  renumbering map\n");
            else OP_map_list[map->index]->map[e*map->dim+j] =
              map->to->core_size + index;
          }
          else OP_map_list[map->index]->map[e*map->dim+j] = index;
        }
      }
    }
  }


  /*-STEP 11 ----------- Save the original set element indexes------------------*/

  //if OP_part_list is empty, (i.e. no previous partitioning done) then
  //create it and store the seperation of elements using core_elems
  //and exp_elems
  if(OP_part_index != OP_set_index)
  {
    //allocate memory for list
    OP_part_list = (part *)xmalloc(OP_set_index*sizeof(part));

    for(int s=0; s<OP_set_index; s++) { //for each set
      op_set set=OP_set_list[s];
      //printf("set %s size = %d\n", set.name, set.size);
      int *g_index = (int *)xmalloc(sizeof(int)*set->size);
      int *partition = (int *)xmalloc(sizeof(int)*set->size);
      for(int i = 0; i< set->size; i++)
      {
        g_index[i] = get_global_index(i,my_rank,
            part_range[set->index],comm_size);
        partition[i] = my_rank;
      }
      decl_partition(set, g_index, partition);

      //combine core_elems and exp_elems to one memory block
      int* temp = (int *)xmalloc(sizeof(int)*set->size);
      memcpy(&temp[0], core_elems[set->index],
          set->core_size*sizeof(int));
      memcpy(&temp[set->core_size], exp_elems[set->index],
          (set->size - set->core_size)*sizeof(int));

      //update OP_part_list[set->index]->g_index
      for(int i = 0; i<set->size; i++)
      {
        temp[i] = OP_part_list[set->index]->g_index[temp[i]];
      }
      free(OP_part_list[set->index]->g_index);
      OP_part_list[set->index]->g_index = temp;
    }
  }
  else //OP_part_list exists (i.e. a partitioning has been done)
    //update the seperation of elements
  {
    for(int s=0; s<OP_set_index; s++) { //for each set
      op_set set=OP_set_list[s];

      //combine core_elems and exp_elems to one memory block
      int* temp = (int *)xmalloc(sizeof(int)*set->size);
      memcpy(&temp[0], core_elems[set->index],
          set->core_size*sizeof(int));
      memcpy(&temp[set->core_size], exp_elems[set->index],
          (set->size - set->core_size)*sizeof(int));

      //update OP_part_list[set->index]->g_index
      for(int i = 0; i<set->size; i++)
      {
        temp[i] = OP_part_list[set->index]->g_index[temp[i]];
      }
      free(OP_part_list[set->index]->g_index);
      OP_part_list[set->index]->g_index = temp;
    }
  }

  /*for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];
    printf("Original Index for set %s\n", set->name);
    for(int i=0; i<set->size; i++ )
    printf(" %d",OP_part_list[set->index]->g_index[i]);
    }*/

  //set up exec and nonexec sizes
  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];
    set->exec_size = OP_import_exec_list[set->index]->size;
    set->nonexec_size = OP_import_nonexec_list[set->index]->size;
  }

  /*-STEP 12 ---------- Clean up and Compute rough halo size numbers------------*/

  for(int i = 0; i<OP_set_index; i++)
  { free(part_range[i]);
    free(core_elems[i]); free(exp_elems[i]);
  }
  free(part_range);
  free(exp_elems); free(core_elems);

  op_timers(&cpu_t2, &wall_t2);  //timer stop for list create
  //compute import/export lists creation time
  time = wall_t2-wall_t1;
  MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);

  //compute avg/min/max set sizes and exec sizes accross the MPI universe
  int avg_size = 0, min_size = 0, max_size = 0;
  for(int s = 0; s< OP_set_index; s++){
    op_set set=OP_set_list[s];

    //number of set elements first
    MPI_Reduce(&set->size, &avg_size,1, MPI_INT, MPI_SUM,
        MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(&set->size, &min_size,1, MPI_INT, MPI_MIN,
        MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(&set->size, &max_size,1, MPI_INT, MPI_MAX,
        MPI_ROOT, OP_MPI_WORLD);

    if(my_rank == MPI_ROOT)
    {
      printf("Num of %8s (avg | min | max)\n",set->name);
      printf("total elems         %10d %10d %10d\n",avg_size/comm_size, min_size, max_size);
    }

    avg_size = 0;min_size = 0; max_size = 0;


    //number of OWNED elements second
    MPI_Reduce(&set->core_size,
        &avg_size,1, MPI_INT, MPI_SUM, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(&set->core_size,
        &min_size,1, MPI_INT, MPI_MIN, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(&set->core_size,
        &max_size,1, MPI_INT, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);

    if(my_rank == MPI_ROOT)
    {
      printf("core elems         %10d %10d %10d \n",avg_size/comm_size, min_size, max_size);
    }
    avg_size = 0;min_size = 0; max_size = 0;


    //number of exec halo elements third
    MPI_Reduce(&OP_import_exec_list[set->index]->size,
        &avg_size,1, MPI_INT, MPI_SUM, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(&OP_import_exec_list[set->index]->size,
        &min_size,1, MPI_INT, MPI_MIN, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(&OP_import_exec_list[set->index]->size,
        &max_size,1, MPI_INT, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);

    if(my_rank == MPI_ROOT)
    {
      printf("exec halo elems     %10d %10d %10d \n", avg_size/comm_size, min_size, max_size);
    }
    avg_size = 0;min_size = 0; max_size = 0;

    //number of non-exec halo elements fourth
    MPI_Reduce(&OP_import_nonexec_list[set->index]->size,
        &avg_size,1, MPI_INT, MPI_SUM, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(&OP_import_nonexec_list[set->index]->size,
        &min_size,1, MPI_INT, MPI_MIN, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(&OP_import_nonexec_list[set->index]->size,
        &max_size,1, MPI_INT, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);

    if(my_rank == MPI_ROOT)
    {
      printf("non-exec halo elems %10d %10d %10d \n", avg_size/comm_size, min_size, max_size);
    }
    avg_size = 0;min_size = 0; max_size = 0;
    if(my_rank == MPI_ROOT)
    {
      printf("-----------------------------------------------------\n");
    }
  }

  if(my_rank == MPI_ROOT)
  {
    printf("\n\n");
  }

  //compute avg/min/max number of MPI neighbors per process accross the MPI universe
  avg_size = 0, min_size = 0, max_size = 0;
  for(int s = 0; s< OP_set_index; s++){
    op_set set=OP_set_list[s];

    //number of exec halo neighbors first
    MPI_Reduce(&OP_import_exec_list[set->index]->ranks_size,
        &avg_size,1, MPI_INT, MPI_SUM, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(&OP_import_exec_list[set->index]->ranks_size,
        &min_size,1, MPI_INT, MPI_MIN, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(&OP_import_exec_list[set->index]->ranks_size,
        &max_size,1, MPI_INT, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);

    if(my_rank == MPI_ROOT)
    {
      printf("MPI neighbors for exchanging %8s (avg | min | max)\n",set->name);
      printf("exec halo elems     %4d %4d %4d\n",avg_size/comm_size, min_size, max_size);


    }
    avg_size = 0;min_size = 0; max_size = 0;

    //number of non-exec halo neighbors second
    MPI_Reduce(&OP_import_nonexec_list[set->index]->ranks_size,
        &avg_size,1, MPI_INT, MPI_SUM, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(&OP_import_nonexec_list[set->index]->ranks_size,
        &min_size,1, MPI_INT, MPI_MIN, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(&OP_import_nonexec_list[set->index]->ranks_size,
        &max_size,1, MPI_INT, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);

    if(my_rank == MPI_ROOT)
    {
      printf("non-exec halo elems %4d %4d %4d\n",avg_size/comm_size, min_size, max_size);
    }
    avg_size = 0;min_size = 0; max_size = 0;
    if(my_rank == MPI_ROOT)
    {
      printf("-----------------------------------------------------\n");
    }
  }

  //compute average worst case halo size in Bytes
  int tot_halo_size = 0;
  for(int s = 0; s< OP_set_index; s++){
    op_set set=OP_set_list[s];

    op_dat_entry *item;
    TAILQ_FOREACH(item, &OP_dat_list, entries) {
      op_dat dat = item->dat;

      if(compare_sets(dat->set,set)==1)
      {
        halo_list exec_imp = OP_import_exec_list[set->index];
        halo_list nonexec_imp= OP_import_nonexec_list[set->index];
        tot_halo_size = tot_halo_size + exec_imp->size*dat->size +
          nonexec_imp->size*dat->size;
      }
    }
  }
  int avg_halo_size;
  MPI_Reduce(&tot_halo_size, &avg_halo_size,1, MPI_INT, MPI_SUM,
      MPI_ROOT, OP_MPI_WORLD);

  //print performance results
  if(my_rank == MPI_ROOT)
  {
    printf("Max total halo creation time = %lf\n",max_time);
    printf("Average (worst case) Halo size = %d Bytes\n",
        avg_halo_size/comm_size);
  }

  //initialise hash table that keeps track of the communication performance of
  //each of the kernels executed
  for (int i=0;i<HASHSIZE;i++) op_mpi_kernel_tab[i].count = 0;
}

/*******************************************************************************
 * Routine to Clean-up all MPI halos(called at the end of an OP2 MPI application)
 *******************************************************************************/

void op_halo_destroy()
{
  //remove halos from op_dats
  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    dat->data =(char *)xrealloc(dat->data,dat->set->size*dat->size);
  }

  //free lists
  for(int s = 0; s< OP_set_index; s++){
    op_set set=OP_set_list[s];

    free(OP_import_exec_list[set->index]->ranks);
    free(OP_import_exec_list[set->index]->disps);
    free(OP_import_exec_list[set->index]->sizes);
    free(OP_import_exec_list[set->index]->list);
    free(OP_import_exec_list[set->index]);

    free(OP_import_nonexec_list[set->index]->ranks);
    free(OP_import_nonexec_list[set->index]->disps);
    free(OP_import_nonexec_list[set->index]->sizes);
    free(OP_import_nonexec_list[set->index]->list);
    free(OP_import_nonexec_list[set->index]);

    free(OP_export_exec_list[set->index]->ranks);
    free(OP_export_exec_list[set->index]->disps);
    free(OP_export_exec_list[set->index]->sizes);
    free(OP_export_exec_list[set->index]->list);
    free(OP_export_exec_list[set->index]);

    free(OP_export_nonexec_list[set->index]->ranks);
    free(OP_export_nonexec_list[set->index]->disps);
    free(OP_export_nonexec_list[set->index]->sizes);
    free(OP_export_nonexec_list[set->index]->list);
    free(OP_export_nonexec_list[set->index]);

  }
  free(OP_import_exec_list);free(OP_import_nonexec_list);
  free(OP_export_exec_list);free(OP_export_nonexec_list);

  item = NULL;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    free(((op_mpi_buffer)(dat->mpi_buffer))->buf_exec);
    free(((op_mpi_buffer)(dat->mpi_buffer))->buf_nonexec);
    free(((op_mpi_buffer)(dat->mpi_buffer))->s_req);
    free(((op_mpi_buffer)(dat->mpi_buffer))->r_req);
  }

  MPI_Comm_free(&OP_MPI_WORLD);
}

/*******************************************************************************
 * Routine to set the dirty bit for an MPI Halo after halo exchange
 *******************************************************************************/

static void set_dirtybit(op_arg* arg)
{
  op_dat dat = arg->dat;

  if((arg->argtype == OP_ARG_DAT) &&
    (arg->acc == OP_INC || arg->acc == OP_WRITE || arg->acc == OP_RW))
    dat->dirtybit = 1;
}



void op_mpi_reduce_float(op_arg* arg, float* data)
{
  (void)data;
  if(arg->argtype == OP_ARG_GBL && arg->acc != OP_READ)
  {
    float result;
    if(arg->acc == OP_INC)//global reduction
    {
      MPI_Allreduce((float *)arg->data, &result, arg->dim, MPI_FLOAT,
          MPI_SUM, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(float)*arg->dim);
    }
    else if(arg->acc == OP_MAX)//global maximum
    {
      MPI_Allreduce((float *)arg->data, &result, arg->dim, MPI_FLOAT,
          MPI_MAX, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(float)*arg->dim);;
    }
    else if(arg->acc == OP_MIN)//global minimum
    {
      MPI_Allreduce((float *)arg->data, &result, arg->dim, MPI_FLOAT,
          MPI_MIN, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(float)*arg->dim);
    }
  }
}

void op_mpi_reduce_double(op_arg* arg, double* data)
{
  (void)data;
  if(arg->argtype == OP_ARG_GBL && arg->acc != OP_READ)
  {
    double result;
    if(arg->acc == OP_INC)//global reduction
    {
      MPI_Allreduce((double *)arg->data, (double *)&result, arg->dim, MPI_DOUBLE,
          MPI_SUM, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(double)*arg->dim);
    }
    else if(arg->acc == OP_MAX)//global maximum
    {
      MPI_Allreduce((double *)arg->data, &result, arg->dim, MPI_DOUBLE,
          MPI_MAX, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(double)*arg->dim);;
    }
    else if(arg->acc == OP_MIN)//global minimum
    {
      MPI_Allreduce((double *)arg->data, &result, arg->dim, MPI_DOUBLE,
          MPI_MIN, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(double)*arg->dim);
    }
  }
}

void op_mpi_reduce_int(op_arg* arg, int* data)
{
  (void)data;
  int result;

  if(arg->argtype == OP_ARG_GBL && arg->acc != OP_READ)
  {
    if(arg->acc == OP_INC)//global reduction
    {
      MPI_Allreduce((int *)arg->data, &result, arg->dim, MPI_INT,
          MPI_SUM, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(int)*arg->dim);
    }
    else if(arg->acc == OP_MAX)//global maximum
    {
      MPI_Allreduce((int *)arg->data, &result, arg->dim, MPI_INT,
          MPI_MAX, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(int)*arg->dim);;
    }
    else if(arg->acc == OP_MIN)//global minimum
    {
      MPI_Allreduce((int *)arg->data, &result, arg->dim, MPI_INT,
          MPI_MIN, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(int)*arg->dim);
    }
  }
}



/*******************************************************************************
 * MPI Global reduce of an op_arg
 *******************************************************************************/

void global_reduce(op_arg *arg)
{
  if(strcmp("double",arg->type)==0)
  {
    double result;
    if(arg->acc == OP_INC)//global reduction
    {
      MPI_Allreduce((double *)arg->data, &result, 1, MPI_DOUBLE,
          MPI_SUM, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(double));
    }
    else if(arg->acc == OP_MAX)//global maximum
    {
      MPI_Allreduce((double *)arg->data, &result, 1, MPI_DOUBLE,
          MPI_MAX, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(double));
    }
    else if(arg->acc == OP_MIN)//global minimum
    {
      MPI_Allreduce((double *)arg->data, &result, 1, MPI_DOUBLE,
          MPI_MIN, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(double));
    }
  }
  else if(strcmp("float",arg->type)==0)
  {
    float result;
    if(arg->acc == OP_INC)//global reduction
    {
      MPI_Allreduce((float *)arg->data, &result, 1, MPI_FLOAT,
          MPI_SUM, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(float));
    }
    else if(arg->acc == OP_MAX)//global maximum
    {
      MPI_Allreduce((float *)arg->data, &result, 1, MPI_FLOAT,
          MPI_MAX, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(float));;
    }
    else if(arg->acc == OP_MIN)//global minimum
    {
      MPI_Allreduce((float *)arg->data, &result, 1, MPI_FLOAT,
          MPI_MIN, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(float));
    }
  }
  else if(strcmp("int",arg->type)==0)
  {
    int result;
    if(arg->acc == OP_INC)//global reduction
    {
      MPI_Allreduce((int *)arg->data, &result,1, MPI_INT,
          MPI_SUM, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(int));
    }
    else if(arg->acc == OP_MAX)//global maximum
    {
      MPI_Allreduce((int *)arg->data, &result, 1, MPI_INT,
          MPI_MAX, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(int));;
    }
    else if(arg->acc == OP_MIN)//global minimum
    {
      MPI_Allreduce((int *)arg->data, &result, 1, MPI_INT,
          MPI_MIN, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(int));
    }
  }
}

/*******************************************************************************
 * Routine to get a copy of the data held in a distributed op_dat
 *******************************************************************************/

op_dat op_mpi_get_data(op_dat dat)
{
  //create new communicator for fetching
  int my_rank, comm_size;
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_WORLD, &comm_size);

  //
  //make a copy of the distributed op_dat on to a distributed temporary op_dat
  //
  op_dat temp_dat = (op_dat) xmalloc(sizeof(op_dat_core));
  char *data = (char *)xmalloc(dat->set->size*dat->size);
  memcpy(data, dat->data, dat->set->size*dat->size);

  //
  //use orig_part_range to fill in OP_part_list[set->index]->elem_part with
  //original partitioning information
  //
  for(int i = 0; i < dat->set->size; i++)
  {
    int local_index;
    OP_part_list[dat->set->index]->elem_part[i] =
      get_partition(OP_part_list[dat->set->index]->g_index[i],
          orig_part_range[dat->set->index], &local_index, comm_size);
  }


  halo_list pe_list;
  halo_list pi_list;

  //
  //create export list
  //
  part p= OP_part_list[dat->set->index];
  int count = 0;int cap = 1000;
  int *temp_list = (int *)xmalloc(cap*sizeof(int));

  for(int i = 0; i < dat->set->size; i++)
  {
    if(p->elem_part[i] != my_rank)
    {
      if(count>=cap)
      {
        cap = cap*2;
        temp_list = (int *)xrealloc(temp_list, cap*sizeof(int));
      }
      temp_list[count++] = p->elem_part[i];
      temp_list[count++] = i;
    }
  }

  pe_list = (halo_list) xmalloc(sizeof(halo_list_core));
  create_export_list(dat->set, temp_list, pe_list, count, comm_size, my_rank);
  free(temp_list);


  //
  //create import list
  //
  int *neighbors, *sizes;
  int ranks_size;

  //-----Discover neighbors-----
  ranks_size = 0;
  neighbors = (int *)xmalloc(comm_size*sizeof(int));
  sizes = (int *)xmalloc(comm_size*sizeof(int));

  find_neighbors_set(pe_list, neighbors, sizes, &ranks_size,
      my_rank, comm_size, OP_MPI_WORLD);
  MPI_Request request_send[pe_list->ranks_size];

  int* rbuf;
  cap = 0; count = 0;

  for(int i=0; i<pe_list->ranks_size; i++) {
    int* sbuf = &pe_list->list[pe_list->disps[i]];
    MPI_Isend( sbuf,  pe_list->sizes[i],  MPI_INT, pe_list->ranks[i], 1,
        OP_MPI_WORLD, &request_send[i] );
  }

  for(int i=0; i< ranks_size; i++) cap = cap + sizes[i];
  temp_list = (int *)xmalloc(cap*sizeof(int));

  for(int i=0; i<ranks_size; i++) {
    rbuf = (int *)xmalloc(sizes[i]*sizeof(int));
    MPI_Recv(rbuf, sizes[i], MPI_INT, neighbors[i], 1, OP_MPI_WORLD,
        MPI_STATUSES_IGNORE );
    memcpy(&temp_list[count],(void *)&rbuf[0],sizes[i]*sizeof(int));
    count = count + sizes[i];
    free(rbuf);
  }

  MPI_Waitall(pe_list->ranks_size,request_send, MPI_STATUSES_IGNORE );
  pi_list = (halo_list) xmalloc(sizeof(halo_list_core));
  create_import_list(dat->set, temp_list, pi_list, count,
      neighbors, sizes, ranks_size, comm_size, my_rank);


  //
  //migrate the temp "data" array to the original MPI ranks
  //

  //prepare bits of the data array to be exported
  char** sbuf_char = (char **)xmalloc(pe_list->ranks_size*sizeof(char *));

  for(int i=0; i < pe_list->ranks_size; i++) {
    sbuf_char[i] = (char *)xmalloc(pe_list->sizes[i]*dat->size);
    for(int j = 0; j<pe_list->sizes[i]; j++)
    {
      int index = pe_list->list[pe_list->disps[i]+j];
      memcpy(&sbuf_char[i][j*dat->size],
          (void *)&data[dat->size*(index)],dat->size);
    }
    MPI_Isend(sbuf_char[i], dat->size*pe_list->sizes[i],
        MPI_CHAR, pe_list->ranks[i],
        dat->index, OP_MPI_WORLD, &request_send[i]);
  }

  char *rbuf_char = (char *)xmalloc(dat->size*pi_list->size);
  for(int i=0; i < pi_list->ranks_size; i++) {
    MPI_Recv(&rbuf_char[pi_list->disps[i]*dat->size],dat->size*pi_list->sizes[i],
        MPI_CHAR, pi_list->ranks[i], dat->index, OP_MPI_WORLD,
        MPI_STATUSES_IGNORE);
  }

  MPI_Waitall(pe_list->ranks_size,request_send, MPI_STATUSES_IGNORE );
  for(int i=0; i < pe_list->ranks_size; i++) free(sbuf_char[i]);
  free(sbuf_char);

  //delete the data entirs that has been sent and create a
  //modified data array
  char* new_dat = (char *)xmalloc(dat->size*(dat->set->size+pi_list->size));

  count = 0;
  for(int i = 0; i < dat->set->size;i++)//iterate over old set size
  {
    if(OP_part_list[dat->set->index]->elem_part[i] == my_rank)
    {
      memcpy(&new_dat[count*dat->size],
          (void *)&data[dat->size*i],dat->size);
      count++;
    }
  }

  memcpy(&new_dat[count*dat->size],(void *)rbuf_char,dat->size*pi_list->size);
  count = count+pi_list->size;
  new_dat = (char *)xrealloc(new_dat,dat->size*count);
  free(rbuf_char);
  free(data);
  data = new_dat;

  //
  //make a copy of the original g_index and migrate that also to the original
  //MPI process
  //
  //prepare bits of the original g_index array to be exported
  int** sbuf = (int **)xmalloc(pe_list->ranks_size*sizeof(int *));

  //send original g_index values to relevant mpi processes
  for(int i=0; i < pe_list->ranks_size; i++) {
    sbuf[i] = (int *)xmalloc(pe_list->sizes[i]*sizeof(int));
    for(int j = 0; j<pe_list->sizes[i]; j++)
    {
      sbuf[i][j] = OP_part_list[dat->set->index]->
        g_index[pe_list->list[pe_list->disps[i]+j]];
    }
    MPI_Isend(sbuf[i],  pe_list->sizes[i],
        MPI_INT, pe_list->ranks[i],
        dat->index, OP_MPI_WORLD, &request_send[i]);
  }

  rbuf = (int *)xmalloc(sizeof(int)*pi_list->size);

  //receive original g_index values from relevant mpi processes
  for(int i=0; i < pi_list->ranks_size; i++) {
    MPI_Recv(&rbuf[pi_list->disps[i]],pi_list->sizes[i],
        MPI_INT, pi_list->ranks[i], dat->index,
        OP_MPI_WORLD, MPI_STATUSES_IGNORE);
  }
  MPI_Waitall(pe_list->ranks_size,request_send, MPI_STATUSES_IGNORE );
  for(int i=0; i < pe_list->ranks_size; i++) free(sbuf[i]); free(sbuf);

  //delete the g_index entirs that has been sent and create a
  //modified g_index
  int* new_g_index = (int *)xmalloc(sizeof(int)*(dat->set->size+pi_list->size));

  count = 0;
  for(int i = 0; i < dat->set->size;i++)//iterate over old size of the g_index array
  {
    if(OP_part_list[dat->set->index]->elem_part[i] == my_rank)
    {
      new_g_index[count] = OP_part_list[dat->set->index]->g_index[i];
      count++;
    }
  }

  memcpy(&new_g_index[count],(void *)rbuf,sizeof(int)*pi_list->size);
  count = count+pi_list->size;
  new_g_index = (int *)xrealloc(new_g_index,sizeof(int)*count);
  free(rbuf);

  //
  //sort elements in temporaty data according to new_g_index
  //
  quickSort_dat(new_g_index,data, 0,count-1, dat->size);

  //cleanup
  free(pe_list->ranks);free(pe_list->disps);
  free(pe_list->sizes);free(pe_list->list);
  free(pe_list);
  free(pi_list->ranks);free(pi_list->disps);
  free(pi_list->sizes);free(pi_list->list);
  free(pi_list);
  free(new_g_index);

  //remember that the original set size is now given by count
  op_set set = (op_set) malloc(sizeof(op_set_core));
  set->index = dat->set->index;
  set->size  = count;
  set->name  = dat->set->name;

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
 * Routine to put (modify) a the data held in a distributed op_dat
 *******************************************************************************/

void op_mpi_put_data(op_dat dat)
{
  (void)dat;
  //the op_dat in parameter list is modified
  //need the orig_part_range and OP_part_list

  //need to do some checks to see if the input op_dat has the same dimensions
  //and other values as the internal op_dat
}

/*******************************************************************************
 * Debug/Diagnostics Routine to initialise import halo data to NaN
 *******************************************************************************/

static void op_reset_halo(op_arg* arg)
{
  op_dat dat = arg->dat;

  if((arg->argtype == OP_ARG_DAT) &&
    (arg->acc == OP_READ || arg->acc == OP_RW ) &&
    (dat->dirtybit == 1))
  {
    //printf("Resetting Halo of data array %10s\n",dat->name);
    halo_list imp_exec_list = OP_import_exec_list[dat->set->index];
    halo_list imp_nonexec_list = OP_import_nonexec_list[dat->set->index];

    // initialise import halo data to NaN
    int double_count = imp_exec_list->size*dat->size/sizeof(double);
    double_count +=  imp_nonexec_list->size*dat->size/sizeof(double);
    double* NaN = (double *)xmalloc(double_count* sizeof(double));
    for(int i = 0; i<double_count; i++) NaN[i] = (double)NAN;//0.0/0.0;

    int init = dat->set->size*dat->size;
    memcpy(&(dat->data[init]), NaN,
      dat->size*imp_exec_list->size + dat->size*imp_nonexec_list->size);
    free(NaN);
  }
}

/*******************************************************************************
 * Routine to output performance measures
 *******************************************************************************/

void mpi_timing_output()
{
  int my_rank, comm_size;
  MPI_Comm OP_MPI_IO_WORLD;
  MPI_Comm_dup(MPI_COMM_WORLD, &OP_MPI_IO_WORLD);
  MPI_Comm_rank(OP_MPI_IO_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_IO_WORLD, &comm_size);

  int count;

  count = 0;
  int tot_count = 0;
  for (int n=0; n<HASHSIZE; n++) {
    MPI_Allreduce(&op_mpi_kernel_tab[n].count,&count, 1, MPI_INT, MPI_SUM, OP_MPI_IO_WORLD);
    tot_count += count;
  }

  if(tot_count > 0)
  {
    double tot_time;
    double avg_time;

    printf("___________________________________________________\n");
    printf("Performance information on rank %d\n", my_rank);
    printf("Kernel        Count  total time(sec)  Avg time(sec)  \n");
    for (int n=0; n<HASHSIZE; n++) {
    if (op_mpi_kernel_tab[n].count>0) {
        printf("%-10s  %6d       %10.4f      %10.4f    \n",
          op_mpi_kernel_tab[n].name,op_mpi_kernel_tab[n].count,
          op_mpi_kernel_tab[n].time,op_mpi_kernel_tab[n].time/op_mpi_kernel_tab[n].count);

#ifdef COMM_PERF
        if(op_mpi_kernel_tab[n].num_indices>0)
        {
          printf("halo exchanges:  ");
          for(int i = 0; i<op_mpi_kernel_tab[n].num_indices; i++)
            printf("%10s ",op_mpi_kernel_tab[n].comm_info[i]->name);
          printf("\n");
          printf("       count  :  ");
          for(int i = 0; i<op_mpi_kernel_tab[n].num_indices; i++)
            printf("%10d ",op_mpi_kernel_tab[n].comm_info[i]->count);printf("\n");
          printf("total(Kbytes) :  ");
          for(int i = 0; i<op_mpi_kernel_tab[n].num_indices; i++)
            printf("%10d ",op_mpi_kernel_tab[n].comm_info[i]->bytes/1024);printf("\n");
          printf("average(bytes):  ");
          for(int i = 0; i<op_mpi_kernel_tab[n].num_indices; i++)
            printf("%10d ",op_mpi_kernel_tab[n].comm_info[i]->bytes/
                op_mpi_kernel_tab[n].comm_info[i]->count );printf("\n");
        }
        else
        {
          printf("halo exchanges:  %10s\n","NONE");
        }
        printf("---------------------------------------------------\n");
#endif

      }
    }
    printf("___________________________________________________\n");

    if(my_rank == MPI_ROOT)
    {
      printf("___________________________________________________\n");
      printf("\nKernel        Count   Max time(sec)   Avg time(sec)  \n");
    }
    for (int n=0; n<HASHSIZE; n++) {
      MPI_Reduce(&op_mpi_kernel_tab[n].count,&count, 1, MPI_INT, MPI_MAX, MPI_ROOT, OP_MPI_IO_WORLD);
      MPI_Reduce(&op_mpi_kernel_tab[n].time,&avg_time, 1, MPI_DOUBLE, MPI_SUM, MPI_ROOT, OP_MPI_IO_WORLD);
      MPI_Reduce(&op_mpi_kernel_tab[n].time,&tot_time, 1, MPI_DOUBLE, MPI_MAX, MPI_ROOT, OP_MPI_IO_WORLD);

      if(my_rank == MPI_ROOT && count > 0)
      {
        printf("%-10s  %6d       %10.4f      %10.4f    \n",
            op_mpi_kernel_tab[n].name,
            count,
            tot_time,
            (avg_time)/comm_size);
      }
      tot_time = avg_time = 0.0;
    }
  }
  MPI_Comm_free(&OP_MPI_IO_WORLD);
}

/*******************************************************************************
 * Routine to measure timing for an op_par_loop / kernel
 *******************************************************************************/

int op_mpi_perf_time(const char* name, double time)
{
  int kernel_index = op2_hash(name);
  if(op_mpi_kernel_tab[kernel_index].count == 0)
  {
    op_mpi_kernel_tab[kernel_index].name = name;
    op_mpi_kernel_tab[kernel_index].num_indices = 0;
    op_mpi_kernel_tab[kernel_index].time     = 0.0;
  }

  op_mpi_kernel_tab[kernel_index].count    += 1;
  op_mpi_kernel_tab[kernel_index].time     += time;

  return kernel_index;
}

#ifdef COMM_PERF

/*******************************************************************************
 * Routine to linear search comm_info array in an op_mpi_kernel for an op_dat
 *******************************************************************************/
int search_op_mpi_kernel(op_dat dat, op_mpi_kernel kernal, int num_indices)
{
   for(int i = 0; i<num_indices; i++)
     if(strcmp(kernal.comm_info[i]->name, dat->name) == 0 &&
       kernal.comm_info[i]->size == dat->size &&
       kernal.comm_info[i]->index == dat->index )
       return i;

   return -1;
}

/*******************************************************************************
 * Routine to measure MPI message sizes exchanged in an op_par_loop / kernel
 *******************************************************************************/

void op_mpi_perf_comm(int kernel_index, op_dat dat)
{
  halo_list exp_exec_list = OP_export_exec_list[dat->set->index];
  halo_list exp_nonexec_list = OP_export_nonexec_list[dat->set->index];

  int tot_halo_size = (exp_exec_list->size + exp_nonexec_list->size) * dat->size;

  int num_indices = op_mpi_kernel_tab[kernel_index].num_indices;

  if(num_indices == 0)
  {
    //set capcity of comm_info array
    op_mpi_kernel_tab[kernel_index].cap = 20;

    op_dat_mpi_comm_info dat_comm = (op_dat_mpi_comm_info) xmalloc(sizeof(op_dat_mpi_comm_info_core));
    op_mpi_kernel_tab[kernel_index].comm_info = (op_dat_mpi_comm_info*)
    xmalloc(sizeof(op_dat_mpi_comm_info *)*op_mpi_kernel_tab[kernel_index].cap);

    //initialize
    dat_comm->name = dat->name;
    dat_comm->size = dat->size;
    dat_comm->index = dat->index;
    dat_comm->count = 0;
    dat_comm->bytes = 0;

    //add first values
    dat_comm->count += 1;
    dat_comm->bytes += tot_halo_size;

    op_mpi_kernel_tab[kernel_index].comm_info[num_indices] = dat_comm;
    op_mpi_kernel_tab[kernel_index].num_indices++;
  }
  else
  {
    int index = search_op_mpi_kernel(dat, op_mpi_kernel_tab[kernel_index], num_indices);

    if(index < 0)
    {
      //increase capacity of comm_info array
      if(num_indices >= op_mpi_kernel_tab[kernel_index].cap)
      {
        op_mpi_kernel_tab[kernel_index].cap = op_mpi_kernel_tab[kernel_index].cap*2;
        op_mpi_kernel_tab[kernel_index].comm_info = (op_dat_mpi_comm_info*)
        xrealloc(op_mpi_kernel_tab[kernel_index].comm_info,
          sizeof(op_dat_mpi_comm_info *)*op_mpi_kernel_tab[kernel_index].cap);
      }

      op_dat_mpi_comm_info dat_comm =
      (op_dat_mpi_comm_info) xmalloc(sizeof(op_dat_mpi_comm_info_core));

      //initialize
      dat_comm->name = dat->name;
      dat_comm->size = dat->size;
      dat_comm->index = dat->index;
      dat_comm->count = 0;
      dat_comm->bytes = 0;

      //add first values
      dat_comm->count += 1;
      dat_comm->bytes += tot_halo_size;

      op_mpi_kernel_tab[kernel_index].comm_info[num_indices] = dat_comm;
      op_mpi_kernel_tab[kernel_index].num_indices++;
    }
    else
    {
      op_mpi_kernel_tab[kernel_index].comm_info[index]->count += 1;
      op_mpi_kernel_tab[kernel_index].comm_info[index]->bytes += tot_halo_size;
    }
  }
}
#endif

/*******************************************************************************
 * Routine to exit an op2 mpi application -
 *******************************************************************************/

void op_mpi_exit()
{
  //cleanup performance data - need to do this in some op_mpi_exit() routine
#ifdef COMM_PERF
  for (int n=0; n<HASHSIZE; n++) {
    for(int i = 0; i<op_mpi_kernel_tab[n].num_indices; i++)
      free(op_mpi_kernel_tab[n].comm_info[i]);
  }
#endif

  //free memory allocated to halos and mpi_buffers
  op_halo_destroy();
  //return all op_dats, op_maps back to original element order
  op_partition_reverse();
  //print each mpi process's timing info for each kernel

}

int op_mpi_halo_exchanges(op_set set, int nargs, op_arg *args) {
  int size = set->size;
  int direct_flag = 1;

  //check if this is a direct loop
  for (int n=0; n<nargs; n++)
    if(args[n].argtype == OP_ARG_DAT && args[n].idx != -1)
      direct_flag = 0;

  if (direct_flag == 1) return size;

  //not a direct loop ...
  for (int n=0; n<nargs; n++) {
    if(args[n].argtype == OP_ARG_DAT)
      op_exchange_halo(&args[n]);

    if(args[n].idx != -1 && args[n].acc != OP_READ)
      size = set->size + set->exec_size;
  }
  return size;
}

void op_mpi_set_dirtybit(int nargs, op_arg *args) {

  for (int n=0; n<nargs; n++) {
    if(args[n].argtype == OP_ARG_DAT)
    {
      set_dirtybit(&args[n]);
    }
  }
}

void op_mpi_wait_all(int nargs, op_arg *args) {
  for (int n=0; n<nargs; n++) {
    op_wait_all(&args[n]);
  }
}

void op_mpi_reset_halos(int nargs, op_arg *args) {
  for (int n=0; n<nargs; n++) {
    op_reset_halo(&args[n]);
  }
}

void op_mpi_global_reduction(int nargs, op_arg *args) {
  for (int n=0; n<nargs; n++) {
    if (args[n].argtype == OP_ARG_GBL && args[n].acc!=OP_READ) global_reduce(&args[n]);
  }
}

void op_mpi_barrier() {

}

#ifdef COMM_PERF
void op_mpi_perf_comms(int k_i, int nargs, op_arg *args) {
  for (int n=0; n<nargs; n++) {
    if (args[n].argtype == OP_ARG_DAT && args[n].sent == 1)
      op_mpi_perf_comm(k_i, (&args[n])->dat);
  }
}
#endif

/*******************************************************************************
 * Write a op_dat to a named ASCI file
 *******************************************************************************/

void print_dat_tofile(op_dat dat, const char *file_name)
{
  //create new communicator for output
  int rank, comm_size;
  MPI_Comm OP_MPI_IO_WORLD;
  MPI_Comm_dup(MPI_COMM_WORLD, &OP_MPI_IO_WORLD);
  MPI_Comm_rank(OP_MPI_IO_WORLD, &rank);
  MPI_Comm_size(OP_MPI_IO_WORLD, &comm_size);

  //compute local number of elements in dat
  int count = dat->set->size;

  if(strcmp(dat->type,"double") == 0)
  {
    double *l_array = (double *) xmalloc(dat->dim*(count)*sizeof(double));
    memcpy(l_array, (void *)&(dat->data[0]),
        dat->size*count);

    int l_size = count;
    int elem_size = dat->dim;
    int* recevcnts = (int *) xmalloc(comm_size*sizeof(int));
    int* displs = (int *) xmalloc(comm_size*sizeof(int));
    int disp = 0;
    double *g_array = 0;

    MPI_Allgather(&l_size, 1, MPI_INT, recevcnts, 1, MPI_INT, OP_MPI_IO_WORLD);

    int g_size = 0;
    for(int i = 0; i<comm_size; i++)
    {
      g_size += recevcnts[i];
      recevcnts[i] = elem_size*recevcnts[i];
    }
    for(int i = 0; i<comm_size; i++)
    {
      displs[i] = disp;
      disp = disp + recevcnts[i];
    }
    if(rank==MPI_ROOT) g_array = (double *) xmalloc(elem_size*g_size*sizeof(double));
    MPI_Gatherv(l_array, l_size*elem_size, MPI_DOUBLE, g_array, recevcnts,
        displs, MPI_DOUBLE, MPI_ROOT, OP_MPI_IO_WORLD);


    if(rank==MPI_ROOT)
    {
      FILE *fp;
      if ( (fp = fopen(file_name,"w")) == NULL) {
        printf("can't open file %s\n",file_name);
        MPI_Abort(OP_MPI_IO_WORLD, -1);
      }

      if (fprintf(fp,"%d %d\n",g_size, elem_size)<0)
      {
        printf("error writing to %s\n",file_name);
        MPI_Abort(OP_MPI_IO_WORLD, -1);
      }

      for(int i = 0; i< g_size; i++)
      {
        for(int j = 0; j < elem_size; j++ )
        {
          if (fprintf(fp,"%lf ",g_array[i*elem_size+j])<0)
          {
            printf("error writing to %s\n",file_name);
            MPI_Abort(OP_MPI_IO_WORLD, -1);
          }
        }
        fprintf(fp,"\n");
      }
      fclose(fp);
      free(g_array);
    }
    free(l_array);free(recevcnts);free(displs);
  }
  else if(strcmp(dat->type,"float") == 0)
  {
    float *l_array = (float *) xmalloc(dat->dim*(count)*sizeof(float));
    memcpy(l_array, (void *)&(dat->data[0]),
        dat->size*count);

    int l_size = count;
    int elem_size = dat->dim;
    int* recevcnts = (int *) xmalloc(comm_size*sizeof(int));
    int* displs = (int *) xmalloc(comm_size*sizeof(int));
    int disp = 0;
    float *g_array = 0;

    MPI_Allgather(&l_size, 1, MPI_INT, recevcnts, 1, MPI_INT, OP_MPI_IO_WORLD);

    int g_size = 0;
    for(int i = 0; i<comm_size; i++)
    {
      g_size += recevcnts[i];
      recevcnts[i] = elem_size*recevcnts[i];
    }
    for(int i = 0; i<comm_size; i++)
    {
      displs[i] = disp;
      disp = disp + recevcnts[i];
    }

    if(rank==MPI_ROOT) g_array = (float *) xmalloc(elem_size*g_size*sizeof(float));
    MPI_Gatherv(l_array, l_size*elem_size, MPI_FLOAT, g_array, recevcnts,
        displs, MPI_FLOAT, MPI_ROOT, OP_MPI_IO_WORLD);


    if(rank==MPI_ROOT)
    {
      FILE *fp;
      if ( (fp = fopen(file_name,"w")) == NULL) {
        printf("can't open file %s\n",file_name);
        MPI_Abort(OP_MPI_IO_WORLD, -1);
      }

      if (fprintf(fp,"%d %d\n",g_size, elem_size)<0)
      {
        printf("error writing to %s\n",file_name);
        MPI_Abort(OP_MPI_IO_WORLD, -1);
      }

      for(int i = 0; i< g_size; i++)
      {
        for(int j = 0; j < elem_size; j++ )
        {
          if (fprintf(fp,"%f ",g_array[i*elem_size+j])<0)
          {
            printf("error writing to %s\n",file_name);
            MPI_Abort(OP_MPI_IO_WORLD, -1);
          }
        }
        fprintf(fp,"\n");
      }
      fclose(fp);
      free(g_array);
    }
    free(l_array);free(recevcnts);free(displs);
  }
  else if(strcmp(dat->type,"int") == 0)
  {
    int *l_array = (int *) xmalloc(dat->dim*(count)*sizeof(int));
    memcpy(l_array, (void *)&(dat->data[0]),
        dat->size*count);

    int l_size = count;
    int elem_size = dat->dim;
    int* recevcnts = (int *) xmalloc(comm_size*sizeof(int));
    int* displs = (int *) xmalloc(comm_size*sizeof(int));
    int disp = 0;
    int *g_array = 0;

    MPI_Allgather(&l_size, 1, MPI_INT, recevcnts, 1, MPI_INT, OP_MPI_IO_WORLD);

    int g_size = 0;
    for(int i = 0; i<comm_size; i++)
    {
      g_size += recevcnts[i];
      recevcnts[i] = elem_size*recevcnts[i];
    }
    for(int i = 0; i<comm_size; i++)
    {
      displs[i] = disp;
      disp = disp + recevcnts[i];
    }

    if(rank==MPI_ROOT) g_array = (int *) xmalloc(elem_size*g_size*sizeof(int));
    MPI_Gatherv(l_array, l_size*elem_size, MPI_INT, g_array, recevcnts,
        displs, MPI_INT, MPI_ROOT, OP_MPI_IO_WORLD);


    if(rank==MPI_ROOT)
    {
      FILE *fp;
      if ( (fp = fopen(file_name,"w")) == NULL) {
        printf("can't open file %s\n",file_name);
        MPI_Abort(OP_MPI_IO_WORLD, -1);
      }

      if (fprintf(fp,"%d %d\n",g_size, elem_size)<0)
      {
        printf("error writing to %s\n",file_name);
        MPI_Abort(OP_MPI_IO_WORLD, -1);
      }

      for(int i = 0; i< g_size; i++)
      {
        for(int j = 0; j < elem_size; j++ )
        {
          if (fprintf(fp,"%d ",g_array[i*elem_size+j])<0)
          {
            printf("error writing to %s\n",file_name);
            MPI_Abort(OP_MPI_IO_WORLD, -1);
          }
        }
        fprintf(fp,"\n");
      }
      fclose(fp);
      free(g_array);
    }
    free(l_array);free(recevcnts);free(displs);
  }
  else
  {
    printf("Unknown type %s, cannot be written to file %s\n",dat->type,file_name);
  }

  MPI_Comm_free(&OP_MPI_IO_WORLD);

}

/*******************************************************************************
 * Write a op_dat to a named Binary file
 *******************************************************************************/

void print_dat_tobinfile(op_dat dat, const char *file_name)
{
  //create new communicator for output
  int rank, comm_size;
  MPI_Comm OP_MPI_IO_WORLD;
  MPI_Comm_dup(MPI_COMM_WORLD, &OP_MPI_IO_WORLD);
  MPI_Comm_rank(OP_MPI_IO_WORLD, &rank);
  MPI_Comm_size(OP_MPI_IO_WORLD, &comm_size);

  //compute local number of elements in dat
  int count = dat->set->size;

  if(strcmp(dat->type,"double") == 0)
  {
    double *l_array  = (double *) xmalloc(dat->dim*(count)*sizeof(double));
    memcpy(l_array, (void *)&(dat->data[0]),
        dat->size*count);

    int l_size = count;
    size_t elem_size = dat->dim;
    int* recevcnts = (int *) xmalloc(comm_size*sizeof(int));
    int* displs = (int *) xmalloc(comm_size*sizeof(int));
    int disp = 0;
    double *g_array = 0;

    MPI_Allgather(&l_size, 1, MPI_INT, recevcnts, 1, MPI_INT, OP_MPI_IO_WORLD);

    int g_size = 0;
    for(int i = 0; i<comm_size; i++)
    {
      g_size += recevcnts[i];
      recevcnts[i] =   elem_size*recevcnts[i];
    }
    for(int i = 0; i<comm_size; i++)
    {
      displs[i] =   disp;
      disp = disp + recevcnts[i];
    }
    if(rank==MPI_ROOT) g_array  = (double *) xmalloc(elem_size*g_size*sizeof(double));
    MPI_Gatherv(l_array, l_size*elem_size, MPI_DOUBLE, g_array, recevcnts,
        displs, MPI_DOUBLE, MPI_ROOT, OP_MPI_IO_WORLD);


    if(rank==MPI_ROOT)
    {
      FILE *fp;
      if ( (fp = fopen(file_name,"wb")) == NULL) {
        printf("can't open file %s\n",file_name);
        MPI_Abort(OP_MPI_IO_WORLD, -1);
      }

      if (fwrite(&g_size, sizeof(int),1, fp)<1)
      {
        printf("error writing to %s",file_name);
        MPI_Abort(OP_MPI_IO_WORLD, -1);
      }
      if (fwrite(&elem_size, sizeof(int),1, fp)<1)
      {
        printf("error writing to %s\n",file_name);
        MPI_Abort(OP_MPI_IO_WORLD, -1);
      }

      for(int i = 0; i< g_size; i++)
      {
        if (fwrite( &g_array[i*elem_size], sizeof(double), elem_size, fp ) < elem_size)
        {
          printf("error writing to %s\n",file_name);
          MPI_Abort(OP_MPI_IO_WORLD, -1);
        }
      }
      fclose(fp);
      free(g_array);
    }
    free(l_array);free(recevcnts);free(displs);
  }
  else if(strcmp(dat->type,"float") == 0)
  {
    float *l_array  = (float *) xmalloc(dat->dim*(count)*sizeof(float));
    memcpy(l_array, (void *)&(dat->data[0]),
        dat->size*count);

    int l_size = count;
    size_t elem_size = dat->dim;
    int* recevcnts = (int *) xmalloc(comm_size*sizeof(int));
    int* displs = (int *) xmalloc(comm_size*sizeof(int));
    int disp = 0;
    float *g_array = 0;

    MPI_Allgather(&l_size, 1, MPI_INT, recevcnts, 1, MPI_INT, OP_MPI_IO_WORLD);

    int g_size = 0;
    for(int i = 0; i<comm_size; i++)
    {
      g_size += recevcnts[i];
      recevcnts[i] =   elem_size*recevcnts[i];
    }
    for(int i = 0; i<comm_size; i++)
    {
      displs[i] =   disp;
      disp = disp + recevcnts[i];
    }
    if(rank==MPI_ROOT) g_array  = (float *) xmalloc(elem_size*g_size*sizeof(float));
    MPI_Gatherv(l_array, l_size*elem_size, MPI_FLOAT, g_array, recevcnts,
        displs, MPI_FLOAT, MPI_ROOT, OP_MPI_IO_WORLD);


    if(rank==MPI_ROOT)
    {
      FILE *fp;
      if ( (fp = fopen(file_name,"wb")) == NULL) {
        printf("can't open file %s\n",file_name);
        MPI_Abort(OP_MPI_IO_WORLD, -1);
      }

      if (fwrite(&g_size, sizeof(int),1, fp)<1)
      {
        printf("error writing to %s",file_name);
        MPI_Abort(OP_MPI_IO_WORLD, -1);
      }
      if (fwrite(&elem_size, sizeof(int),1, fp)<1)
      {
        printf("error writing to %s\n",file_name);
        MPI_Abort(OP_MPI_IO_WORLD, -1);
      }

      for(int i = 0; i< g_size; i++)
      {
        if (fwrite( &g_array[i*elem_size], sizeof(float), elem_size, fp ) < elem_size)
        {
          printf("error writing to %s\n",file_name);
          MPI_Abort(OP_MPI_IO_WORLD, -1);
        }
      }
      fclose(fp);
      free(g_array);
    }
    free(l_array);free(recevcnts);free(displs);
  }
  else if(strcmp(dat->type,"int") == 0)
  {
    int *l_array  = (int *) xmalloc(dat->dim*(count)*sizeof(int));
    memcpy(l_array, (void *)&(dat->data[0]),
        dat->size*count);

    int l_size = count;
    size_t elem_size = dat->dim;
    int* recevcnts = (int *) xmalloc(comm_size*sizeof(int));
    int* displs = (int *) xmalloc(comm_size*sizeof(int));
    int disp = 0;
    int *g_array = 0;

    MPI_Allgather(&l_size, 1, MPI_INT, recevcnts, 1, MPI_INT, OP_MPI_IO_WORLD);

    int g_size = 0;
    for(int i = 0; i<comm_size; i++)
    {
      g_size += recevcnts[i];
      recevcnts[i] =   elem_size*recevcnts[i];
    }
    for(int i = 0; i<comm_size; i++)
    {
      displs[i] =   disp;
      disp = disp + recevcnts[i];
    }
    if(rank==MPI_ROOT) g_array  = (int *) xmalloc(elem_size*g_size*sizeof(int));
    MPI_Gatherv(l_array, l_size*elem_size, MPI_INT, g_array, recevcnts,
        displs, MPI_INT, MPI_ROOT, OP_MPI_IO_WORLD);

    if(rank==MPI_ROOT)
    {
      FILE *fp;
      if ( (fp = fopen(file_name,"wb")) == NULL) {
        printf("can't open file %s\n",file_name);
        MPI_Abort(OP_MPI_IO_WORLD, -1);
      }

      if (fwrite(&g_size, sizeof(int),1, fp)<1)
      {
        printf("error writing to %s",file_name);
        MPI_Abort(OP_MPI_IO_WORLD, -1);
      }
      if (fwrite(&elem_size, sizeof(int),1, fp)<1)
      {
        printf("error writing to %s\n",file_name);
        MPI_Abort(OP_MPI_IO_WORLD, -1);
      }

      for(int i = 0; i< g_size; i++)
      {
        if (fwrite( &g_array[i*elem_size], sizeof(int), elem_size, fp ) < elem_size)
        {
          printf("error writing to %s\n",file_name);
          MPI_Abort(OP_MPI_IO_WORLD, -1);
        }
      }
      fclose(fp);
      free(g_array);
    }
    free(l_array);free(recevcnts);free(displs);
  }
  else
  {
    printf("Unknown type %s, cannot be written to file %s\n",dat->type,file_name);
  }

  MPI_Comm_free(&OP_MPI_IO_WORLD);
}

/*******************************************************************************
 * Get the global size of a set
 *******************************************************************************/

int op_get_size(op_set set)
{
  int my_rank, comm_size;
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_WORLD, &comm_size);
  int* sizes = (int *)malloc(sizeof(int)*comm_size);
  int g_size = 0;
  MPI_Allgather(&set->size, 1, MPI_INT, sizes, 1, MPI_INT, OP_MPI_WORLD);
  for(int i = 0; i<comm_size; i++)g_size = g_size + sizes[i];
  free(sizes);

  return g_size;
}

