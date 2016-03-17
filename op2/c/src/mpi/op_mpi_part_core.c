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

//mpi header
#include <mpi.h>

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_util.h>

//ptscotch header
#ifdef HAVE_PTSCOTCH
#include <ptscotch.h>

#endif

//parmetis header
#ifdef HAVE_PARMETIS
#include <parmetis.h>
#ifdef PARMETIS_VER_4
typedef idx_t idxtype;
#endif
#endif

#include <op_lib_mpi.h>


extern int *OP_map_partial_exchange;//flag for each map ..
                                    //used for checking if partial halo exchanges
                                    //are to be performed


//
//MPI Communicator for partitioning
//

MPI_Comm OP_PART_WORLD;

/*******************************************************************************
 * Utility function to find the number of times a value appears in an array
 *******************************************************************************/

static int frequencyof(int value, int* array, int size)
{
  int frequency = 0;
  for(int i = 0; i<size; i++)
  {
    if(array[i] == value) frequency++;
  }
  return frequency;
}

/*******************************************************************************
 * Utility function to find the mode of a set of numbers in an array
 *******************************************************************************/

static int find_mode(int* array, int size)
{
  int count = 0, mode = array[0], current;
  for(int i=0; i<size; i++)
  {
    if (i>0 && array[i]==array[i-1]) continue;
    current = frequencyof(array[i], array, size);
    if(count< current)
    {
      count = current;
      mode = array[i];
    }
  }
  return mode;
}

/*******************************************************************************
 * Utility function to see if a target op_set is held in an op_set array
 *******************************************************************************/

static int compare_all_sets(op_set target_set, op_set other_sets[], int size)
{
  for(int i = 0; i < size; i++)
  {
    if(compare_sets(target_set, other_sets[i])==1)return i;
  }
  return -1;
}

/*******************************************************************************
 * Special routine to create export list during partitioning map->to set
 * from map_>from set in partition_to_set()
 *******************************************************************************/

static int* create_exp_list_2(op_set set, int* temp_list, halo_list h_list,
    int* part_list, int size, int comm_size, int my_rank)
{
  (void)my_rank;
  int* ranks = (int *) xmalloc(comm_size*sizeof(int));
  int* to_list = (int *) xmalloc((size/3)*sizeof(int));
  part_list = (int *) xmalloc((size/3)*sizeof(int));
  int* disps = (int *) xmalloc(comm_size*sizeof(int));
  int* sizes = (int *) xmalloc(comm_size*sizeof(int));

  int index = 0; int total_size = 0;

  //negative values set as an initialisation
  for(int r = 0;r<comm_size;r++)
  {
    disps[r] = ranks[r] = -99;
    sizes[r] = 0;
  }

  for(int r = 0;r<comm_size;r++)
  {
    sizes[index] = 0;
    disps[index] = 0;
    int* temp_to = (int *) xmalloc((size/3)*sizeof(int));
    int* temp_part = (int *) xmalloc((size/3)*sizeof(int));

    for(int i = 0;i<size;i=i+3)
    {
      if(temp_list[i]==r)
      {
        temp_to[sizes[index]] = temp_list[i+1];
        temp_part[sizes[index]] = temp_list[i+2];
        sizes[index]++;
      }
    }

    if(sizes[index]>0)
    {
      ranks[index] = r;
      //no sorting
      total_size = total_size + sizes[index];
      //no eleminating duplicates
      if(index > 0)
        disps[index] = disps[index-1] +  sizes[index-1];

      //add to end of t_list and p_list
      for(int e = 0;e<sizes[index];e++)
      {
        to_list[disps[index]+e] = temp_to[e];
        part_list[disps[index]+e] = temp_part[e];
      }
      index++;
    }
    op_free(temp_to);
    op_free(temp_part);
  }


  h_list->set = set;
  h_list->size = total_size;
  h_list->ranks = ranks;
  h_list->ranks_size = index;
  h_list->disps = disps;
  h_list->sizes = sizes;
  h_list->list = to_list;

  return part_list;
}

/*******************************************************************************
 * Special routine to create import list during partitioning map->to set
 * from map_>from set in partition_to_set()
 *******************************************************************************/

static void create_imp_list_2(op_set set, int* temp_list, halo_list h_list,
    int total_size, int* ranks, int* sizes, int ranks_size, int comm_size,
    int my_rank)
{
  (void)my_rank;
  int* disps = (int *) xmalloc(comm_size*sizeof(int));
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
 * Routine to use a partitioned map->to set to partition the map->from set
 *******************************************************************************/

static int partition_from_set(op_map map, int my_rank, int comm_size, int** part_range)
{
  (void)my_rank;
  part p_set = OP_part_list[map->to->index];

  int cap = 100; int count = 0;
  int* temp_list = (int*) xmalloc(cap*sizeof(int));

  halo_list pi_list = (halo_list) xmalloc(sizeof(halo_list_core));

  //go through the map and build an import list of the non-local "to" elements
  for(int i = 0; i < map->from->size; i++)
  {
    int part, local_index;
    for(int j = 0; j<map->dim; j++)
    {
      part = get_partition(map->map[i*map->dim+j],
          part_range[map->to->index],&local_index,comm_size);
      if(count>=cap)
      {
        cap = cap*2;
        temp_list = (int *)xrealloc(temp_list,cap*sizeof(int));
      }

      if(part != my_rank)
      {
        temp_list[count++] = part;
        temp_list[count++] = local_index;
      }
    }
  }
  create_export_list(map->to,temp_list, pi_list, count, comm_size, my_rank);
  op_free(temp_list);

  //now, discover neighbors and create export list of "to" elements
  int ranks_size = 0;

  int* neighbors = (int *)xmalloc(comm_size*sizeof(int));
  int* sizes = (int *)xmalloc(comm_size*sizeof(int));

  halo_list pe_list = (halo_list) xmalloc(sizeof(halo_list_core));
  find_neighbors_set(pi_list, neighbors, sizes, &ranks_size, my_rank,
      comm_size, OP_PART_WORLD);

  //  MPI_Request request_send[pi_list->ranks_size];
  MPI_Request* request_send = (MPI_Request *)xmalloc(pi_list->ranks_size*sizeof(MPI_Request));

  int* rbuf;
  cap = 0; count = 0;

  for(int i=0; i < pi_list->ranks_size; i++) {
    int* sbuf = &pi_list->list[pi_list->disps[i]];
    MPI_Isend( sbuf,  pi_list->sizes[i],  MPI_INT, pi_list->ranks[i], 1,
        OP_PART_WORLD, &request_send[i] );
  }

  for(int i=0; i< ranks_size; i++) cap = cap + sizes[i];
  temp_list = (int *)xmalloc(cap*sizeof(int));

  for(int i=0; i<ranks_size; i++) {
    rbuf = (int *)xmalloc(sizes[i]*sizeof(int));
    MPI_Recv(rbuf, sizes[i], MPI_INT, neighbors[i], 1, OP_PART_WORLD,
        MPI_STATUS_IGNORE );
    memcpy(&temp_list[count],(void *)&rbuf[0],sizes[i]*sizeof(int));
    count = count + sizes[i];
    op_free(rbuf);
  }
  MPI_Waitall(pi_list->ranks_size,request_send, MPI_STATUSES_IGNORE );
  create_import_list(map->to,temp_list, pe_list, count,
      neighbors, sizes, ranks_size, comm_size, my_rank);

  //use the import and export lists to exchange partition information of
  //this "to" set
  MPI_Request request_send_p[pe_list->ranks_size];

  //first - prepare partition information of the "to" set element to be exported
  int** sbuf = (int **)xmalloc(pe_list->ranks_size*sizeof(int *));
  for(int i = 0; i < pe_list->ranks_size; i++)
  {
    //printf("export to %d from rank %d set %s of size %d\n",
    //   pe_list->ranks[i], my_rank, map->to->name, pe_list->sizes[i] );
    sbuf[i] = (int *)xmalloc(pe_list->sizes[i]*sizeof(int));
    for(int j = 0; j<pe_list->sizes[i]; j++)
    {
      int elem = pe_list->list[pe_list->disps[i]+j];
      sbuf[i][j] = p_set->elem_part[elem];
    }
    MPI_Isend(sbuf[i],  pe_list->sizes[i],  MPI_INT, pe_list->ranks[i],
        2, OP_PART_WORLD, &request_send_p[i]);
  }

  //second - prepare space for the incomming partition information of the "to" set
  int* imp_part = (int *)xmalloc(sizeof(int)*pi_list->size);

  //third - receive
  for(int i=0; i<pi_list->ranks_size; i++) {
    //printf("import from %d to rank %d set %s of size %d\n",
    //    pi_list->ranks[i], my_rank, map->to->name, pi_list->sizes[i] );
    MPI_Recv(&imp_part[pi_list->disps[i]],
        pi_list->sizes[i], MPI_INT, pi_list->ranks[i], 2,
        OP_PART_WORLD, MPI_STATUS_IGNORE);

  }
  MPI_Waitall(pe_list->ranks_size,request_send_p, MPI_STATUSES_IGNORE );
  for(int i = 0; i<pe_list->ranks_size; i++) op_free(sbuf[i]);op_free(sbuf);

  //allocate memory to hold the partition details for the set thats going to be
  //partitioned
  int* partition = (int *)xmalloc(sizeof(int)*map->from->size);

  //go through the mapping table and the imported partition information and
  //partition the "from" set
  for(int i = 0; i<map->from->size; i++)
  {
    int part, local_index;
    int found_parts[map->dim];
    for(int j = 0; j < map->dim; j++)
    {
      part = get_partition(map->map[i*map->dim+j],
          part_range[map->to->index],&local_index,comm_size);

      if(part == my_rank)
        found_parts[j] = p_set->elem_part[local_index];
      else //get partition information from imported data
      {
        int r = binary_search(pi_list->ranks,part,0,pi_list->ranks_size-1);
        if(r >= 0)
        {
          int elem = binary_search(&pi_list->list[pi_list->disps[r]],
              local_index,0,pi_list->sizes[r]-1);
          if(elem >= 0)
            found_parts[j] = imp_part[pi_list->disps[r]+elem];
          else
          {
            printf("Element %d not found in partition import list\n",
                local_index);
            MPI_Abort(OP_PART_WORLD, 2);
          }
        }
        else
        {
          printf("Rank %d not found in partition import list\n", part);
          MPI_Abort(OP_PART_WORLD, 2);
        }
      }
    }
    partition[i] = find_mode(found_parts, map->dim);
  }

  OP_part_list[map->from->index]->elem_part = partition;
  OP_part_list[map->from->index]->is_partitioned = 1;

  //cleanup
  op_free(imp_part);
  op_free(pi_list->list);op_free(pi_list->ranks);op_free(pi_list->sizes);
  op_free(pi_list->disps);op_free(pi_list);
  op_free(pe_list->list);op_free(pe_list->ranks);op_free(pe_list->sizes);
  op_free(pe_list->disps);op_free(pe_list);

  free(request_send);

  return 1;
}

/*******************************************************************************
 * Routine to use the partitioned map->from set to partition the map->to set
 *******************************************************************************/

static int partition_to_set(op_map map, int my_rank, int comm_size, int** part_range)
{
  part p_set = OP_part_list[map->from->index];

  int cap = 300; int count = 0;
  int* temp_list = (int *)xmalloc(cap*sizeof(int));

  halo_list pe_list = (halo_list) xmalloc(sizeof(halo_list_core));
  int* part_list_e = NULL; //corresponding "to" element's partition infomation
  //exported to an mpi rank

  //go through the map and if any element pointed to by a mapping table entry
  //(i.e. a "from" set element) is in a foreign partition, add the partition
  // of the from element to be exported to that mpi foreign process
  //also collect information about the local "to" elements
  for(int i = 0; i < map->from->size; i++)
  {
    int part;
    int local_index;

    for(int j=0; j < map->dim; j++)
    {
      part = get_partition(map->map[i*map->dim+j],
          part_range[map->to->index],&local_index,comm_size);

      if(part != my_rank)
      {
        if(count>=cap)
        {
          cap = cap*3;
          temp_list = (int *)xrealloc(temp_list,cap*sizeof(int));
        }

        temp_list[count++] = part; //curent partition (i.e. mpi rank)
        temp_list[count++] = local_index;//map->map[i*map->dim+j];//global index
        temp_list[count++] = p_set->elem_part[i]; //new partition
      }
    }
  }

  part_list_e = create_exp_list_2(map->to,temp_list, pe_list, part_list_e,
      count, comm_size, my_rank);
  op_free(temp_list);


  int ranks_size = 0;
  int* neighbors = (int *)xmalloc(comm_size*sizeof(int));
  int* sizes = (int *)xmalloc(comm_size*sizeof(int));

  //to_part_list tpi_list;
  halo_list pi_list = (halo_list) xmalloc(sizeof(halo_list_core));
  int* part_list_i = NULL; //corresponding "to" element's partition infomation
  //imported from an mpi rank

  find_neighbors_set(pe_list,neighbors,sizes,&ranks_size,my_rank,
      comm_size, OP_PART_WORLD);

  //  MPI_Request request_send_t[pe_list->ranks_size];
  MPI_Request* request_send_t = (MPI_Request *)xmalloc(pe_list->ranks_size*sizeof(MPI_Request));
  //MPI_Request request_send_p[pe_list->ranks_size];
  MPI_Request* request_send_p = (MPI_Request *)xmalloc(pe_list->ranks_size*sizeof(MPI_Request));

  int *rbuf_t, *rbuf_p;
  cap = 0; count = 0;

  for(int i=0; i < pe_list->ranks_size; i++) {
    int* sbuf_t = &pe_list->list[pe_list->disps[i]];
    int* sbuf_p = &part_list_e[pe_list->disps[i]];
    MPI_Isend( sbuf_t,  pe_list->sizes[i],  MPI_INT, pe_list->ranks[i], 1,
        OP_PART_WORLD, &request_send_t[i] );
    MPI_Isend( sbuf_p,  pe_list->sizes[i],  MPI_INT, pe_list->ranks[i], 2,
        OP_PART_WORLD, &request_send_p[i] );
  }

  for(int i=0; i< ranks_size; i++) cap = cap + sizes[i];
  int* temp_list_t = (int *)xmalloc(cap*sizeof(int));
  part_list_i = (int *)xmalloc(cap*sizeof(int));

  for(int i=0; i<ranks_size; i++) {
    rbuf_t = (int *)xmalloc(sizes[i]*sizeof(int));
    rbuf_p = (int *)xmalloc(sizes[i]*sizeof(int));

    MPI_Recv(rbuf_t, sizes[i], MPI_INT, neighbors[i], 1, OP_PART_WORLD,
        MPI_STATUS_IGNORE );
    MPI_Recv(rbuf_p, sizes[i], MPI_INT, neighbors[i], 2, OP_PART_WORLD,
        MPI_STATUS_IGNORE );
    memcpy(&temp_list_t[count],(void *)&rbuf_t[0],sizes[i]*sizeof(int));
    memcpy(&part_list_i[count],(void *)&rbuf_p[0],sizes[i]*sizeof(int));
    count = count + sizes[i];
    op_free(rbuf_t);
    op_free(rbuf_p);
  }
  MPI_Waitall(pe_list->ranks_size,request_send_t, MPI_STATUSES_IGNORE );
  MPI_Waitall(pe_list->ranks_size,request_send_p, MPI_STATUSES_IGNORE );

  create_imp_list_2(map->to, temp_list_t, pi_list, count,
      neighbors, sizes, ranks_size, comm_size, my_rank);


  //-----go through local mapping table as well as the imported information
  //and partition the "to" set
  cap = map->to->size;
  count = 0;
  int *to_elems = (int *)xmalloc(sizeof(int)*cap);
  int *parts = (int *)xmalloc(sizeof(int)*cap);

  //--first the local mapping table
  int local_index;
  int part;
  for(int i = 0; i < map->from->size; i++)
  {
    for(int j=0; j < map->dim; j++)
    {
      part = get_partition(map->map[i*map->dim+j],
          part_range[map->to->index], &local_index,comm_size);
      if(part == my_rank)
      {
        if(count>=cap)
        {
          cap = cap*2;
          parts = (int *)xrealloc(parts,sizeof(int)*cap);
          to_elems = (int *)xrealloc(to_elems,sizeof(int)*cap);
        }
        to_elems[count] = local_index;
        parts[count++] = p_set->elem_part[i];
      }
    }
  }

  //copy pi_list.list and part_list_i to to_elems and parts
  if(count+pi_list->size > 0)
  {
    to_elems = (int *)xrealloc(to_elems, sizeof(int)*(count+pi_list->size));
    parts = (int *)xrealloc(parts, sizeof(int)*(count+pi_list->size));
  }

  memcpy(&to_elems[count],(void *)&pi_list->list[0],pi_list->size*sizeof(int));
  memcpy(&parts[count],(void *)&part_list_i[0],pi_list->size*sizeof(int));

  int *partition = (int *)xmalloc(sizeof(int)*map->to->size);
  for(int i = 0; i < map->to->size; i++){partition[i] = -99;}

  count = count+pi_list->size;

  //sort both to_elems[] and correspondingly parts[] arrays
  if(count > 0)quickSort_2(to_elems, parts, 0, count-1);

  if (count > comm_size*10) {
    int* part_counter = (int*)xmalloc(comm_size*sizeof(int));
    for(int i = 0; i < count;) {
      memset(part_counter, 0, comm_size*sizeof(int));
      int curr = to_elems[i];
      do{
        part_counter[parts[i]]++;
        i++;
        if(i>=count) break;
      } while(curr == to_elems[i]);
      int maxpos=0;
      for (int j = 0; j < comm_size;j++)
        if (part_counter[maxpos] < part_counter[j]) maxpos=j;
      partition[curr] = maxpos;
    }
    free(part_counter);
  } else {
    int* found_parts;
    for(int i = 0; i<count;)
    {
      int curr = to_elems[i];
      int c = 0; cap = map->dim;
      found_parts = (int *)xmalloc(sizeof(int)*cap);

      do{
        if(c>=cap)
        {
          cap = cap*2;
          found_parts = (int *)xrealloc(found_parts, sizeof(int)*cap);
        }
        found_parts[c++] =  parts[i];
        i++;
        if(i>=count) break;
      } while(curr == to_elems[i]);

      partition[curr] = find_mode(found_parts, c);
      op_free(found_parts);
    }
  }

  if(count+pi_list->size > 0)
  {
    op_free(to_elems);op_free(parts);
  }

  //check if this "from" set is an "on to" set
  //need to check this globally on all processors
  int ok = 1;
  for(int i = 0; i < map->to->size; i++)
  {
    if(partition[i]<0)
    {
      if (OP_diags>2)
      {
        printf("on rank %d: Map %s is not an an on-to mapping \
            from set %s to set %s\n", my_rank, map->name,
            map->from->name,map->to->name);
      }
      //return -1;
      ok = -1;
      break;
    }
  }

  //check if globally this map was giving us an on-to set mapping
  int* global_ok_array = (int*)xmalloc(comm_size*sizeof(int));
  for(int r = 0;r<comm_size;r++)global_ok_array[r] = 1;
  MPI_Allgather( &ok, 1, MPI_INT,  global_ok_array,
      1,MPI_INT,OP_PART_WORLD);
  int result = 1;
  for(int r = 0;r<comm_size;r++)
  {
    if(global_ok_array[r]<0)
    {
      //printf("Rank %d reported problem partitioning\n",r);
      result = -1;
    }
  }
  op_free(global_ok_array);

  if(result == 1)
  {
    OP_part_list[map->to->index]->elem_part = partition;
    OP_part_list[map->to->index]->is_partitioned = 1;
  }
  else
  {
    op_free(partition);
  }

  //cleanup
  op_free(pi_list->list);op_free(pi_list->ranks);op_free(pi_list->sizes);
  op_free(pi_list->disps);op_free(pi_list);
  op_free(pe_list->list);op_free(pe_list->ranks);op_free(pe_list->sizes);
  op_free(pe_list->disps); op_free(pe_list);
  op_free(part_list_i);op_free(part_list_e);

  free(request_send_p);
  free(request_send_t);

  return result;
}

/*******************************************************************************
 * Routine to partition all secondary sets using primary set partition
 *******************************************************************************/

static void partition_all(op_set primary_set, int my_rank, int comm_size)
{
  // Compute global partition range information for each set
  int** part_range = (int **)xmalloc(OP_set_index*sizeof(int*));
  get_part_range(part_range,my_rank,comm_size, OP_PART_WORLD);

  int sets_partitioned = 1;
  int maps_used = 0;

  op_set all_partitioned_sets[OP_set_index];
  int all_used_maps[OP_map_index];
  for(int i = 0; i<OP_map_index; i++) { all_used_maps[i] = -1;}

  //begin with the partitioned primary set
  all_partitioned_sets[0] = OP_set_list[primary_set->index];

  int error = 0;
  while(sets_partitioned < OP_set_index && error == 0)
  {
    int cost[OP_map_index];
    for(int i = 0; i<OP_map_index; i++) cost[i] = 99;

    //compute a "cost" associated with using each mapping table
    for(int m=0; m<OP_map_index; m++)
    {
      op_map map=OP_map_list[m];

      if(linear_search(all_used_maps,map->index,0,maps_used-1)<0)// if not used before
      {
        part to_set = OP_part_list[map->to->index];
        part from_set = OP_part_list[map->from->index];

        //partitioning a set using a mapping from a partitioned set costs
        //more than partitioning a set using a mapping to a partitioned set
        //i.e. preferance is given to the latter over the former
        if(from_set->is_partitioned == 1 &&
            compare_all_sets(map->from,all_partitioned_sets, sets_partitioned)>=0)
          cost[map->index] = 2;
        else if(to_set->is_partitioned == 1 &&
            compare_all_sets(map->to,all_partitioned_sets,sets_partitioned)>=0)
          cost[map->index] = (map->dim == 1 ? 0 : 1);
      }
    }

    while(1)
    {
      int selected = min(cost, OP_map_index);

      if(selected >= 0)
      {
        op_map map=OP_map_list[selected];

        //partition using this map
        part to_set = OP_part_list[map->to->index];
        part from_set = OP_part_list[map->from->index];

        if(to_set->is_partitioned == 1 && from_set->is_partitioned == 0)
        {
          if( partition_from_set(map, my_rank, comm_size, part_range) > 0)
          {
            all_partitioned_sets[sets_partitioned++] = map->from;
            all_used_maps[maps_used++] = map->index;
            break;
          }
          else //partitioning unsuccessful with this map- find another map
            cost[selected] = 99;
        }
        else if(from_set->is_partitioned == 1 && to_set->is_partitioned == 0)
        {
          if( partition_to_set(map, my_rank, comm_size, part_range) > 0)
          {
            all_partitioned_sets[sets_partitioned++] = map->to;
            all_used_maps[maps_used++] = map->index;
            break;
          }
          else //partitioning unsuccessful with this map - find another map
            cost[selected] = 99;
        }
        else {
          cost[selected] = 99;
        }
      }
      else //partitioning error;
      {
        printf("On rank %d: Partitioning error\n",my_rank);
        error = 1; break;
      }
    }
  }

  if(my_rank==MPI_ROOT)
  {
    int die = 0;
    printf("Sets partitioned = %d\n",sets_partitioned);
    if(sets_partitioned != OP_set_index)
    {
      for(int s=0; s<OP_set_index; s++) { //for each set
        op_set set = OP_set_list[s];
        part P=OP_part_list[set->index];
        if(P->is_partitioned != 1)
        {
          printf("Unable to find mapping between primary set and %s \n",
              P->set->name);
          if (P->set->size != 0) die = 1;
        }
      }
      if (die) {
        printf("Partitioning aborted !\n");
        MPI_Abort(OP_PART_WORLD, 1);
      }
    }
  }

  for(int i = 0; i<OP_set_index; i++)op_free(part_range[i]);op_free(part_range);
}

/*******************************************************************************
 * Routine to renumber mapping table entries with new partition's indexes
 *******************************************************************************/

static void renumber_maps(int my_rank, int comm_size)
{
  //get partition rage information
  int** part_range = (int **)xmalloc(OP_set_index*sizeof(int*));
  get_part_range(part_range,my_rank,comm_size, OP_PART_WORLD);

  //find elements of the "to" set thats not in this local process
  for(int m=0; m<OP_map_index; m++) { //for each maping table
    op_map map=OP_map_list[m];

    int cap = 1000; int count = 0;
    int* req_list = (int *)xmalloc(cap*sizeof(int));

    for(int i = 0; i< map->from->size; i++)
    {
      int local_index;
      for(int j=0; j<map->dim; j++)
      {
        local_index = binary_search(OP_part_list[map->to->index]->g_index,
            map->map[i*map->dim+j], 0, map->to->size-1);

        if(count>=cap)
        {
          cap = cap*2;
          req_list = (int *)xrealloc(req_list, cap*sizeof(int));
        }

        if(local_index < 0) // not in this partition
        {
          //store the global index of the element
          req_list[count++] = map->map[i*map->dim+j];
        }
      }
    }
    //sort and remove duplicates
    if(count > 0)
    {
      quickSort(req_list, 0, count-1);
      count = removeDups(req_list, count);
      req_list = (int *)xrealloc(req_list, count*sizeof(int));
    }

    //do an allgather to findout how many elements that each process will
    //be requesting partition information about
    int recv_count[comm_size];
    MPI_Allgather(&count, 1, MPI_INT, recv_count, 1, MPI_INT, OP_PART_WORLD);

    //discover global size of these required elements
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

    //allocate memory to hold the global indexes of elements requiring partition details
    int *g_index = (int *)xmalloc(sizeof(int)*g_count);

    MPI_Allgatherv(req_list,count,MPI_INT, g_index,recv_count,displs,
        MPI_INT, OP_PART_WORLD);
    op_free(req_list);

    if(g_count > 0)
    {
      quickSort(g_index, 0, g_count-1);
      g_count = removeDups(g_index, g_count);
      g_index = (int *)xrealloc(g_index, g_count*sizeof(int));
    }

    //printf("on rank %d map %s needs set %s : before g_count = %d\n",
    //    my_rank, map->name, map->to->name, g_count);

    //go through the recieved global g_index array and see if any local element's
    //partition details are requested by some foreign process
    int *exp_index = (int *)xmalloc(sizeof(int)*g_count);
    int *exp_g_index = (int *)xmalloc(sizeof(int)*g_count);

    int exp_count = 0;
    for(int i = 0; i<g_count; i++)
    {
      int local_index = binary_search(OP_part_list[map->to->index]->g_index,
          g_index[i],0,map->to->size-1);
      int global_index;
      if(local_index >= 0)
      {
        exp_g_index[exp_count] = g_index[i];

        global_index = get_global_index(local_index, my_rank,
            part_range[map->to->index], comm_size);
        exp_index[exp_count++] = global_index;
      }
    }
    op_free(g_index);

    //realloc exp_index, exp_g_index
    exp_index = (int *)xrealloc(exp_index,sizeof(int)*exp_count);
    exp_g_index = (int *)xrealloc(exp_g_index,sizeof(int)*exp_count);

    //now export to every MPI rank, these partition info with an all-to-all
    MPI_Allgather(&exp_count, 1, MPI_INT, recv_count, 1, MPI_INT, OP_PART_WORLD);
    disp = 0; op_free(displs);
    displs = (int *)xmalloc(comm_size*sizeof(int));

    for(int i = 0; i<comm_size; i++)
    {
      displs[i] =   disp;
      disp = disp + recv_count[i];
    }

    //allocate memory to hold the incomming partition details and allgatherv
    g_count = 0;
    for(int i = 0; i< comm_size; i++)g_count += recv_count[i];
    int *all_imp_index = (int *)xmalloc(sizeof(int)*g_count);
    g_index = (int *)xmalloc(sizeof(int)*g_count);

    //printf("on rank %d map %s need set %s: After g_count = %d\n",
    //    my_rank, map.name,map.to.name,g_count);

    MPI_Allgatherv(exp_g_index,exp_count,MPI_INT, g_index,recv_count,displs,
        MPI_INT, OP_PART_WORLD);

    MPI_Allgatherv(exp_index,exp_count,MPI_INT, all_imp_index,recv_count,
        displs, MPI_INT, OP_PART_WORLD);

    op_free(exp_index);
    op_free(exp_g_index);

    //sort all_imp_index according to g_index array
    if(g_count > 0)quickSort_2(g_index, all_imp_index, 0, g_count-1);

    //now we hopefully have all the informattion required to renumber this map
    //so now, again go through each entry of this mapping table and renumber
    for(int i = 0; i< map->from->size; i++)
    {
      int local_index, global_index;
      for(int j=0; j < map->dim; j++)
      {
        local_index = binary_search(OP_part_list[map->to->index]->g_index,
            map->map[i*map->dim+j], 0, map->to->size-1);

        if(local_index < 0) // not in this partition
        {
          //need to search through g_index array
          int found = binary_search(g_index,map->map[i*map->dim+j],
              0, g_count-1);
          if(found < 0) printf("Problem in renumbering\n");
          else
          {
            OP_map_list[map->index]-> map[i*map->dim+j] =
              all_imp_index[found];
          }
        }
        else //in this partition
        {
          global_index = get_global_index(local_index, my_rank,
              part_range[map->to->index], comm_size);
          OP_map_list[map->index]->map[i*map->dim+j] = global_index;
        }
      }
    }

    op_free(g_index);
    op_free(displs);
    op_free(all_imp_index);
  }
  for(int i = 0; i<OP_set_index; i++)op_free(part_range[i]);op_free(part_range);
}

/*******************************************************************************
 * Routine to perform data migration to new partitions
 *******************************************************************************/

static void migrate_all(int my_rank, int comm_size)
{
  /*--STEP 1 - Create Imp/Export Lists for reverse migrating elements ----------*/

  //create imp/exp lists for reverse migration
  halo_list pe_list[OP_set_index]; //export list for each set
  halo_list pi_list[OP_set_index]; //import list for each set

  //create partition export lists
  int* temp_list; int count, cap;

  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];
    part p= OP_part_list[set->index];

    //create a temporaty scratch space to hold export list for this set's
    //partition information
    count = 0;cap = 1000;
    temp_list = (int *)xmalloc(cap*sizeof(int));

    for(int i = 0; i < set->size; i++)
    {
      if(p->elem_part[i] != my_rank)
      {
        if(count>=cap)
        {
          cap = cap*2;
          temp_list = (int *)xrealloc(temp_list, cap*sizeof(int));
        }
        temp_list[count++] = p->elem_part[i];
        temp_list[count++] = i;//part.g_index[i];
      }
    }
    //create partition export list
    pe_list[set->index] = (halo_list) xmalloc(sizeof(halo_list_core));
    create_export_list(set, temp_list, pe_list[set->index],
        count, comm_size, my_rank);
    op_free(temp_list);
  }

  //create partition import lists
  int *neighbors, *sizes;
  int ranks_size;

  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];

    halo_list exp = pe_list[set->index];

    //-----Discover neighbors-----
    ranks_size = 0;
    neighbors = (int *)xmalloc(comm_size*sizeof(int));
    sizes = (int *)xmalloc(comm_size*sizeof(int));

    find_neighbors_set(exp, neighbors, sizes, &ranks_size,
        my_rank, comm_size, OP_PART_WORLD);

    //    MPI_Request request_send[exp->ranks_size];
    MPI_Request* request_send = (MPI_Request *)xmalloc(exp->ranks_size*sizeof(MPI_Request));

    int* rbuf;
    cap = 0; count = 0;

    for(int i=0; i<exp->ranks_size; i++) {
      //printf("export from %d to %d set %10s, list of size %d \n",
      //my_rank,exp->ranks[i],set->name,exp->sizes[i]);
      int* sbuf = &exp->list[exp->disps[i]];
      MPI_Isend( sbuf,  exp->sizes[i],  MPI_INT, exp->ranks[i], 1,
          OP_PART_WORLD, &request_send[i] );
    }

    for(int i=0; i< ranks_size; i++) cap = cap + sizes[i];
    temp_list = (int *)xmalloc(cap*sizeof(int));

    for(int i=0; i<ranks_size; i++) {
      //printf("import from %d to %d set %10s, list of size %d\n",
      //neighbors[i], my_rank, set->name, sizes[i]);
      rbuf = (int *)xmalloc(sizes[i]*sizeof(int));

      MPI_Recv(rbuf, sizes[i], MPI_INT, neighbors[i], 1, OP_PART_WORLD,
          MPI_STATUS_IGNORE );
      memcpy(&temp_list[count],(void *)&rbuf[0],sizes[i]*sizeof(int));
      count = count + sizes[i];
      op_free(rbuf);
    }

    MPI_Waitall(exp->ranks_size,request_send, MPI_STATUSES_IGNORE );
    pi_list[set->index] = (halo_list) xmalloc(sizeof(halo_list_core));
    create_import_list(set, temp_list, pi_list[set->index], count,
        neighbors, sizes, ranks_size, comm_size, my_rank);

    free(request_send);
  }


  /*--STEP 2 - Perform Partitioning Data migration -----------------------------*/

  //data migration first ......
  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];

    halo_list imp = pi_list[set->index];
    halo_list exp = pe_list[set->index];

    //    MPI_Request request_send[exp->ranks_size];
    MPI_Request* request_send = (MPI_Request *)xmalloc(exp->ranks_size*sizeof(MPI_Request));

    //migrate data defined on this set
    op_dat_entry *item; int d = -1; //d is just simply the tag for mpi comms
    TAILQ_FOREACH(item, &OP_dat_list, entries) {
      d++; //increase tag to do mpi comm for the next op_dat
      op_dat dat = item->dat;

      if(compare_sets(dat->set,set)==1) //this data array is defines on this set
      {

        //prepare bits of the data array to be exported
        char** sbuf = (char **)xmalloc(exp->ranks_size*sizeof(char *));

        for(int i=0; i < exp->ranks_size; i++) {
          sbuf[i] = (char *)xmalloc(exp->sizes[i]*dat->size);
          for(int j = 0; j<exp->sizes[i]; j++)
          {
            int index = exp->list[exp->disps[i]+j];
            memcpy(&sbuf[i][j*dat->size],
                (void *)&dat->data[dat->size*(index)],dat->size);
          }
          //printf("export from %d to %d data %10s, number of elements of size %d | sending:\n ",
          //    my_rank,exp->ranks[i],dat->name,exp->sizes[i]);
          MPI_Isend(sbuf[i], dat->size*exp->sizes[i],
              MPI_CHAR, exp->ranks[i],
              d, OP_PART_WORLD, &request_send[i]);
        }

        char *rbuf = (char *)xmalloc(dat->size*imp->size);
        for(int i=0; i<imp->ranks_size; i++) {
          //printf("imported on to %d data %10s, number of elements of size %d | recieving:\n ",
          //    my_rank, dat->name, imp->size);
          MPI_Recv(&rbuf[imp->disps[i]*dat->size],dat->size*imp->sizes[i],
              MPI_CHAR, imp->ranks[i], d, OP_PART_WORLD, MPI_STATUS_IGNORE);
        }

        MPI_Waitall(exp->ranks_size,request_send, MPI_STATUSES_IGNORE );
        for(int i=0; i < exp->ranks_size; i++) op_free(sbuf[i]); op_free(sbuf);

        //delete the data entirs that has been sent and create a
        //modified data array
        char* new_dat = (char *)xmalloc(dat->size*(set->size+imp->size));

        count = 0;
        for(int i = 0; i < dat->set->size;i++)//iterate over old set size
        {
          if(OP_part_list[set->index]->elem_part[i] == my_rank)
          {
            memcpy(&new_dat[count*dat->size],
                (void *)&dat->data[dat->size*i],dat->size);
            count++;
          }
        }

        memcpy(&new_dat[count*dat->size],(void *)rbuf,
            dat->size*imp->size);
        count = count+imp->size;
        new_dat = (char *)xrealloc(new_dat,dat->size*count);
        op_free(rbuf);

        op_free(dat->data);
        dat->data = new_dat;
      }
    }

    free(request_send);
  }

  //mapping tables second ......
  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];

    halo_list imp = pi_list[set->index];
    halo_list exp = pe_list[set->index];

    //    MPI_Request request_send[exp->ranks_size];
    MPI_Request* request_send = (MPI_Request *)xmalloc(exp->ranks_size*sizeof(MPI_Request));

    //migrate mapping tables from this set
    for(int m=0; m<OP_map_index; m++) { //for each maping table
      op_map map=OP_map_list[m];

      if(compare_sets(map->from,set)==1) //need to select mappings FROM this set
      {
        //prepare bits of the mapping tables to be exported
        int** sbuf = (int **)xmalloc(exp->ranks_size*sizeof(int *));

        //send mapping table entirs to relevant mpi processes
        for(int i=0; i<exp->ranks_size; i++) {
          sbuf[i] = (int *)xmalloc(exp->sizes[i]*map->dim*sizeof(int));
          for(int j = 0; j<exp->sizes[i]; j++)
          {
            for(int p = 0; p< map->dim; p++)
            {
              sbuf[i][j*map->dim+p] =
                map->map[map->dim*(exp->list[exp->disps[i]+j])+p];
            }
          }
          //printf("\n export from %d to %d map %10s, number of elements of size %d | sending:\n ",
          //    my_rank,exp->ranks[i],map->name,exp->sizes[i]);
          MPI_Isend(sbuf[i],  map->dim*exp->sizes[i],
              MPI_INT, exp->ranks[i],
              m, OP_PART_WORLD, &request_send[i]);
        }

        int *rbuf = (int *)xmalloc(map->dim*sizeof(int)*imp->size);

        //receive mapping table entirs from relevant mpi processes
        for(int i=0; i < imp->ranks_size; i++) {
          //printf("\n imported on to %d map %10s, number of elements of size %d | recieving: ",
          //    my_rank, map->name, imp->size);
          MPI_Recv(&rbuf[imp->disps[i]*map->dim],
              map->dim*imp->sizes[i],
              MPI_INT, imp->ranks[i], m,
              OP_PART_WORLD, MPI_STATUS_IGNORE);
        }

        MPI_Waitall(exp->ranks_size,request_send, MPI_STATUSES_IGNORE );
        for(int i=0; i < exp->ranks_size; i++) op_free(sbuf[i]); op_free(sbuf);

        //delete the mapping table entirs that has been sent and create a
        //modified mapping table
        int* new_map = (int *)xmalloc(sizeof(int)*(set->size+imp->size)*map->dim);

        count = 0;
        for(int i = 0; i < map->from->size;i++)//iterate over old size of the maping table
        {
          if(OP_part_list[map->from->index]->elem_part[i] == my_rank)
          {
            memcpy(&new_map[count*map->dim],
                (void *)&OP_map_list[map->index]-> map[map->dim*i],
                map->dim*sizeof(int));
            count++;
          }
        }
        memcpy(&new_map[count*map->dim],(void *)rbuf,
            map->dim*sizeof(int)*imp->size);
        count = count+imp->size;
        new_map = (int *)xrealloc(new_map,sizeof(int)*count*map->dim);

        op_free(rbuf);
        op_free(OP_map_list[map->index]->map);
        OP_map_list[map->index]->map = new_map;
      }
    }

    free(request_send);
  }

  /*--STEP 3 - Update Partitioning Information and Sort Set Elements------------*/

  //need to exchange the original g_index
  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];

    halo_list imp = pi_list[set->index];
    halo_list exp = pe_list[set->index];

    //    MPI_Request request_send[exp->ranks_size];
    MPI_Request* request_send = (MPI_Request *)xmalloc(exp->ranks_size*sizeof(MPI_Request));

    //prepare bits of the original g_index array to be exported
    int** sbuf = (int **)xmalloc(exp->ranks_size*sizeof(int *));

    //send original g_index values to relevant mpi processes
    for(int i=0; i<exp->ranks_size; i++) {
      sbuf[i] = (int *)xmalloc(exp->sizes[i]*sizeof(int));
      for(int j = 0; j<exp->sizes[i]; j++)
      {
        sbuf[i][j] = OP_part_list[set->index]->
          g_index[exp->list[exp->disps[i]+j]];
      }
      MPI_Isend(sbuf[i],  exp->sizes[i],
          MPI_INT, exp->ranks[i],
          s, OP_PART_WORLD, &request_send[i]);
    }


    int *rbuf = (int *)xmalloc(sizeof(int)*imp->size);

    //receive original g_index values from relevant mpi processes
    for(int i=0; i < imp->ranks_size; i++) {

      MPI_Recv(&rbuf[imp->disps[i]],imp->sizes[i],
          MPI_INT, imp->ranks[i], s,
          OP_PART_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Waitall(exp->ranks_size,request_send, MPI_STATUSES_IGNORE );
    for(int i=0; i < exp->ranks_size; i++) op_free(sbuf[i]); op_free(sbuf);

    //delete the g_index entirs that has been sent and create a
    //modified g_index
    int* new_g_index = (int *)xmalloc(sizeof(int)*(set->size+imp->size));

    count = 0;
    for(int i = 0; i < set->size;i++)//iterate over old size of the g_index array
    {
      if(OP_part_list[set->index]->elem_part[i] == my_rank)
      {
        new_g_index[count] = OP_part_list[set->index]->g_index[i];
        count++;
      }
    }

    memcpy(&new_g_index[count],(void *)rbuf,sizeof(int)*imp->size);
    count = count+imp->size;
    new_g_index = (int *)xrealloc(new_g_index,sizeof(int)*count);
    int* new_part = (int *)xmalloc(sizeof(int)*count);
    for(int i = 0; i< count; i++)new_part[i] = my_rank;

    op_free(rbuf);
    op_free(OP_part_list[set->index]->g_index);
    op_free(OP_part_list[set->index]->elem_part);

    OP_part_list[set->index]->elem_part = new_part;
    OP_part_list[set->index]->g_index = new_g_index;

    OP_set_list[set->index]->size = count;
    OP_part_list[set->index]->set= OP_set_list[set->index];

    free(request_send);
  }

  //re-set values in mapping tables
  for(int m=0; m<OP_map_index; m++) { //for each maping table
    op_map map=OP_map_list[m];

    OP_map_list[map->index]->from = OP_set_list[map->from->index];
    OP_map_list[map->index]->to = OP_set_list[map->to->index];
  }

  //re-set values in data arrays
  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    dat->set = OP_set_list[dat->set->index];
  }

  //finally .... need to sort for each set, data on the set and mapping tables
  //from this set accordiing to the OP_part_list[set.index]->g_index array values.
  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];

    //first ... data on this set
    op_dat_entry *item;
    TAILQ_FOREACH(item, &OP_dat_list, entries) {
      op_dat dat = item->dat;

      if(compare_sets(dat->set,set) == 1)
      {
        if(set->size > 0)
        {
          int* temp = (int *)xmalloc(sizeof(int)*set->size);
          memcpy(temp, (void *)OP_part_list[set->index]->g_index,
              sizeof(int)*set->size);
          quickSort_dat(temp,dat->data, 0, set->size-1, dat->size);
          op_free(temp);
        }
      }
    }

    //second ... mapping tables
    for(int m=0; m<OP_map_index; m++) { //for each maping table
      op_map map=OP_map_list[m];

      if(compare_sets(map->from,set) == 1)
      {
        if(set->size > 0)
        {
          int* temp = (int *)xmalloc(sizeof(int)*set->size);
          memcpy(temp, (void *)OP_part_list[set->index]->g_index,
              sizeof(int)*set->size);
          quickSort_map(temp,OP_map_list[map->index]->map, 0,
              set->size-1, map->dim);
          op_free(temp);
        }
      }
    }
    if(set->size > 0)
      quickSort(OP_part_list[set->index]->g_index, 0, set->size-1);
  }

  //cleanup
  //destroy pe_list, pi_list
  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];
    op_free(pe_list[set->index]->ranks);op_free(pe_list[set->index]->disps);
    op_free(pe_list[set->index]->sizes);op_free(pe_list[set->index]->list);

    op_free(pi_list[set->index]->ranks);op_free(pi_list[set->index]->disps);
    op_free(pi_list[set->index]->sizes);op_free(pi_list[set->index]->list);
    op_free(pe_list[set->index]);op_free(pi_list[set->index]);
  }

}

/*****************************************************************************************************************************************
 * This routine partitions based on information contained in an op_dat called partvecXXXX (number of total partitions, padded with 0s)
 *****************************************************************************************************************************************/

void op_partition_external(op_set primary_set, op_dat partvec)
{
  //declare timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  double time;
  double max_time;

  op_timers(&cpu_t1, &wall_t1); //timer start for partitioning

  //create new communicator for partitioning
  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_PART_WORLD);
  MPI_Comm_rank(OP_PART_WORLD, &my_rank);
  MPI_Comm_size(OP_PART_WORLD, &comm_size);

  /*--STEP 0 - initialise partitioning data stauctures with the current (block)
    partitioning information */

  // Compute global partition range information for each set
  int** part_range = (int **)xmalloc(OP_set_index*sizeof(int*));
  get_part_range(part_range,my_rank,comm_size, OP_PART_WORLD);

  //save the original part_range for future partition reversing
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

  //allocate memory for list
  OP_part_list = (part *)xmalloc(OP_set_index*sizeof(part));

  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];
    int *g_index = (int *)xmalloc(sizeof(int)*set->size);
    for(int i = 0; i< set->size; i++)
      g_index[i] = get_global_index(i,my_rank, part_range[set->index],comm_size);
    decl_partition(set, g_index, NULL);
  }

  int *partition = (int *)xmalloc(sizeof(int)*primary_set->size);
  memcpy(partition, partvec->data, sizeof(int)*primary_set->size);

  //initialise primary set as partitioned
  OP_part_list[primary_set->index]->elem_part= partition;
  OP_part_list[primary_set->index]->is_partitioned = 1;

  //free part range
  for(int i = 0; i<OP_set_index; i++)op_free(part_range[i]);op_free(part_range);

  /*-STEP 2 - Partition all other sets,migrate data and renumber mapping tables-*/

  //partition all other sets
  partition_all(primary_set, my_rank, comm_size);

  //migrate data, sort elements
  migrate_all(my_rank, comm_size);

  //renumber mapping tables
  renumber_maps(my_rank, comm_size);

  op_timers(&cpu_t2, &wall_t2);  //timer stop for partitioning

  //print time for partitioning
  time = wall_t2-wall_t1;
  MPI_Reduce(&time,&max_time,1,MPI_DOUBLE, MPI_MAX,MPI_ROOT, OP_PART_WORLD);
  MPI_Comm_free(&OP_PART_WORLD);
  if(my_rank==MPI_ROOT)printf("Max total random partitioning time = %lf\n",max_time);
}

/*******************************************************************************
 * This routine partitions a given set randomly
 *******************************************************************************/

void op_partition_random(op_set primary_set)
{
  //declare timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  double time;
  double max_time;

  op_timers(&cpu_t1, &wall_t1); //timer start for partitioning

  //create new communicator for partitioning
  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_PART_WORLD);
  MPI_Comm_rank(OP_PART_WORLD, &my_rank);
  MPI_Comm_size(OP_PART_WORLD, &comm_size);

  /*--STEP 0 - initialise partitioning data stauctures with the current (block)
    partitioning information */

  // Compute global partition range information for each set
  int** part_range = (int **)xmalloc(OP_set_index*sizeof(int*));
  get_part_range(part_range,my_rank,comm_size, OP_PART_WORLD);

  //save the original part_range for future partition reversing
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

  //allocate memory for list
  OP_part_list = (part *)xmalloc(OP_set_index*sizeof(part));

  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];
    //printf("set %s size = %d\n", set.name, set.size);
    int *g_index = (int *)xmalloc(sizeof(int)*set->size);
    for(int i = 0; i< set->size; i++)
      g_index[i] = get_global_index(i,my_rank, part_range[set->index],comm_size);
    decl_partition(set, g_index, NULL);
  }

  /*-----STEP 1 - Partition Primary set using a random number generator --------*/

  int *partition = (int *)xmalloc(sizeof(int)*primary_set->size);
  //printf("RAND_MAX = %d",RAND_MAX);
  for(int i = 0; i < primary_set->size; i++)
  {
    //not sure if this is the best way to generate the required random number
    partition[i] = //rand()%comm_size;
    (int)((double)rand()/((double)RAND_MAX + 1)*comm_size);
  }

  //initialise primary set as partitioned
  OP_part_list[primary_set->index]->elem_part= partition;
  OP_part_list[primary_set->index]->is_partitioned = 1;

  //free part range
  for(int i = 0; i<OP_set_index; i++)op_free(part_range[i]);op_free(part_range);

  /*-STEP 2 - Partition all other sets,migrate data and renumber mapping tables-*/

  //partition all other sets
  partition_all(primary_set, my_rank, comm_size);

  //migrate data, sort elements
  migrate_all(my_rank, comm_size);

  //renumber mapping tables
  renumber_maps(my_rank, comm_size);

  op_timers(&cpu_t2, &wall_t2);  //timer stop for partitioning
  //printf time for partitioning
  time = wall_t2-wall_t1;
  MPI_Reduce(&time,&max_time,1,MPI_DOUBLE, MPI_MAX,MPI_ROOT, OP_PART_WORLD);
  MPI_Comm_free(&OP_PART_WORLD);
  if(my_rank==MPI_ROOT)printf("Max total random partitioning time = %lf\n",max_time);
}

/*******************************************************************************
 * Routine to revert back to the original partitioning
 *******************************************************************************/

void op_partition_destroy()
{
  //destroy OP_part_list[]
  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];
    op_free(OP_part_list[set->index]->g_index);
    op_free(OP_part_list[set->index]->elem_part);
    op_free(OP_part_list[set->index]);
  }
  op_free(OP_part_list);
  for(int i = 0; i<OP_set_index; i++)op_free(orig_part_range[i]);
  op_free(orig_part_range);
}

#ifdef HAVE_PARMETIS

/*******************************************************************************
 * Wrapper routine to use ParMETIS_V3_PartGeom() which partitions a set
 * Using its XYZ Geometry Data
 *******************************************************************************/

void op_partition_geom(op_dat coords)
{
  //declare timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  double time;
  double max_time;

  op_timers(&cpu_t1, &wall_t1); //timer start for partitioning

  //create new communicator for partitioning
  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_PART_WORLD);
  MPI_Comm_rank(OP_PART_WORLD, &my_rank);
  MPI_Comm_size(OP_PART_WORLD, &comm_size);

  /*--STEP 0 - initialise partitioning data stauctures with the current (block)
    partitioning information */

  // Compute global partition range information for each set
  int** part_range = (int **)xmalloc(OP_set_index*sizeof(int*));
  get_part_range(part_range,my_rank,comm_size, OP_PART_WORLD);

  //save the original part_range for future partition reversing
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

  //allocate memory for list
  OP_part_list = (part *)xmalloc(OP_set_index*sizeof(part));

  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];
    //printf("set %s size = %d\n", set.name, set.size);
    int *g_index = (int *)xmalloc(sizeof(int)*set->size);
    for(int i = 0; i< set->size; i++)
      g_index[i] = get_global_index(i,my_rank, part_range[set->index],comm_size);
    decl_partition(set, g_index, NULL);
  }

  /*--- STEP 1 - Partition primary set using its coordinates (1D,2D or 3D) -----*/

  //Setup data structures for ParMetis PartGeom
  idx_t *vtxdist = (idx_t *)xmalloc(sizeof(idx_t)*(comm_size+1));
  idx_t *partition = (idx_t *)xmalloc(sizeof(idx_t)*coords->set->size);

  idx_t ndims = coords->dim;
  float* xyz = 0;

  // Create ParMetis compatible coordinates array
  //- i.e. coordinates should be floats
  if(ndims == 3 || ndims == 2 || ndims == 1)
  {
    xyz = (float* )xmalloc(coords->set->size*coords->dim*sizeof(float));
    size_t mult = coords->size/coords->dim;
    for(int i = 0;i < coords->set->size;i++)
    {
      double temp;
      for(int e = 0; e < coords->dim;e++)
      {
        memcpy(&temp, (void *)&(coords->data[(i*coords->dim+e)*mult]), mult);
        xyz[i*coords->dim + e] = (float)temp;
      }
    }
  }
  else
  {
    printf("Dimensions of Coordinate array not one of 3D,2D or 1D\n");
    printf("Not supported by ParMetis - Indicate correct coordinates array\n");
    MPI_Abort(OP_PART_WORLD, 1);
  }

  for(int i=0; i<comm_size; i++)
  {
    vtxdist[i] = part_range[coords->set->index][2*i];
  }
  vtxdist[comm_size] = part_range[coords->set->index][2*(comm_size-1)+1]+1;


  //use xyz coordinates to feed into ParMETIS_V3_PartGeom
  ParMETIS_V3_PartGeom(vtxdist, &ndims, xyz, partition, &OP_PART_WORLD);
  op_free(xyz);op_free(vtxdist);

  //free part range
  for(int i = 0; i<OP_set_index; i++)op_free(part_range[i]);op_free(part_range);

  //saniti check to see if all elements were partitioned
  for(int i = 0; i<coords->size; i++)
  {
    if(partition[i]<0)
    {
      printf("Partitioning problem: on rank %d, set %s element %d not assigned a partition\n",
          my_rank,coords->name, i);
      MPI_Abort(OP_PART_WORLD, 2);
    }
  }

  //initialise primary set as partitioned
  OP_part_list[coords->set->index]->elem_part= (int *) partition;
  OP_part_list[coords->set->index]->is_partitioned = 1;

  /*-STEP 2 - Partition all other sets,migrate data and renumber mapping tables-*/

  //partition all other sets
  partition_all(coords->set, my_rank, comm_size);

  //migrate data, sort elements
  migrate_all(my_rank, comm_size);

  //renumber mapping tables
  renumber_maps(my_rank, comm_size);

  op_timers(&cpu_t2, &wall_t2);  //timer stop for partitioning
  //printf time for partitioning
  time = wall_t2-wall_t1;
  MPI_Reduce(&time,&max_time,1,MPI_DOUBLE, MPI_MAX,MPI_ROOT, OP_PART_WORLD);
  MPI_Comm_free(&OP_PART_WORLD);
  if(my_rank==MPI_ROOT)printf("Max total geometric partitioning time = %lf\n",max_time);
}

/*******************************************************************************
 * Wrapper routine to partition a given set Using ParMETIS PartKway()
 *******************************************************************************/

void op_partition_kway(op_map primary_map)
{
  //declare timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  double time;
  double max_time;

  op_timers(&cpu_t1, &wall_t1); //timer start for partitioning

  //create new communicator for partitioning
  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_PART_WORLD);
  MPI_Comm_rank(OP_PART_WORLD, &my_rank);
  MPI_Comm_size(OP_PART_WORLD, &comm_size);

#ifdef DEBUG
  //check if the  primary_map is an on to map from the from-set to the to-set
  if(is_onto_map(primary_map) != 1)
  {
    printf("Map %s is an not an onto map from set %s to set %s \n",
        primary_map->name, primary_map->from->name, primary_map->to->name);
    MPI_Abort(OP_PART_WORLD, 2);
  }
#endif

  /*--STEP 0 - initialise partitioning data stauctures with the current (block)
    partitioning information */

  // Compute global partition range information for each set
  int** part_range = (int **)xmalloc(OP_set_index*sizeof(int*));
  get_part_range(part_range,my_rank,comm_size, OP_PART_WORLD);

  //save the original part_range for future partition reversing
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

  //allocate memory for list
  OP_part_list = (part *)xmalloc(OP_set_index*sizeof(part));

  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];
    //printf("set %s size = %d\n", set.name, set.size);
    int *g_index = (int *)xmalloc(sizeof(int)*set->size);
    for(int i = 0; i< set->size; i++)
      g_index[i] = get_global_index(i,my_rank, part_range[set->index],comm_size);
    decl_partition(set, g_index, NULL);
  }

  /*--STEP 1 - Construct adjacency list of the to-set of the primary_map -------*/

  //
  //create export list
  //
  int c = 0; int cap = 1000;
  int* list = (int *)xmalloc(cap*sizeof(int));//temp list

  for(int e=0; e < primary_map->from->size; e++) { //for each maping table entry
    int part, local_index;
    for(int j=0; j < primary_map->dim; j++) { //for each element pointed at by this entry
      part = get_partition(primary_map->map[e*primary_map->dim+j],
          part_range[primary_map->to->index],&local_index,comm_size);
      if(c>=cap)
      {
        cap = cap*2;
        list = (int *)xrealloc(list,cap*sizeof(int));
      }

      if(part != my_rank){
        list[c++] = part; //add to export list
        list[c++] = e;
      }
    }
  }
  halo_list exp_list= (halo_list)xmalloc(sizeof(halo_list_core));
  create_export_list(primary_map->from,list, exp_list, c, comm_size, my_rank);
  op_free(list);//free temp list

  //
  //create import list
  //
  int *neighbors, *sizes;
  int ranks_size;

  //-----Discover neighbors-----
  ranks_size = 0;
  neighbors = (int *)xmalloc(comm_size*sizeof(int));
  sizes = (int *)xmalloc(comm_size*sizeof(int));

  //halo_list list = OP_export_exec_list[set->index];
  find_neighbors_set(exp_list, neighbors, sizes, &ranks_size, my_rank,
      comm_size, OP_PART_WORLD);
  //  MPI_Request request_send[exp_list->ranks_size];
  MPI_Request* request_send = (MPI_Request *)xmalloc(exp_list->ranks_size*sizeof(MPI_Request));

  int* rbuf, index = 0;
  cap = 0;

  for(int i=0; i<exp_list->ranks_size; i++) {
    int* sbuf = &exp_list->list[exp_list->disps[i]];
    MPI_Isend( sbuf,  exp_list->sizes[i],  MPI_INT, exp_list->ranks[i],
        primary_map->index, OP_PART_WORLD, &request_send[i] );
  }

  for(int i=0; i< ranks_size; i++) cap = cap + sizes[i];
  int* temp = (int *)xmalloc(cap*sizeof(int));

  //import this list from those neighbors
  for(int i=0; i<ranks_size; i++) {
    rbuf = (int *)xmalloc(sizes[i]*sizeof(int));
    MPI_Recv(rbuf, sizes[i], MPI_INT, neighbors[i],primary_map->index,
        OP_PART_WORLD, MPI_STATUS_IGNORE );
    memcpy(&temp[index],(void *)&rbuf[0],sizes[i]*sizeof(int));
    index = index + sizes[i];
    op_free(rbuf);
  }
  MPI_Waitall(exp_list->ranks_size,request_send, MPI_STATUSES_IGNORE );

  halo_list imp_list= (halo_list)xmalloc(sizeof(halo_list_core));
  create_import_list(primary_map->from, temp, imp_list, index,neighbors, sizes,
      ranks_size, comm_size, my_rank);


  //
  //Exchange mapping table entries using the import/export lists
  //

  //prepare bits of the mapping tables to be exported
  int** sbuf = (int **)xmalloc(exp_list->ranks_size*sizeof(int *));

  for(int i=0; i < exp_list->ranks_size; i++) {
    sbuf[i] = (int *)xmalloc(exp_list->sizes[i]*primary_map->dim*sizeof(int));
    for(int j = 0; j < exp_list->sizes[i]; j++)
    {
      for(int p = 0; p < primary_map->dim; p++)
      {
        sbuf[i][j*primary_map->dim+p] =
          primary_map->map[primary_map->dim*
          (exp_list->list[exp_list->disps[i]+j])+p];
      }
    }
    MPI_Isend(sbuf[i],  primary_map->dim*exp_list->sizes[i],  MPI_INT,
        exp_list->ranks[i], primary_map->index, OP_PART_WORLD, &request_send[i]);
  }

  //prepare space for the incomming mapping tables
  int* foreign_maps = (int *)xmalloc(primary_map->dim*(imp_list->size)*sizeof(int));

  for(int i=0; i<imp_list->ranks_size; i++) {
    MPI_Recv(&foreign_maps[imp_list->disps[i]*primary_map->dim],
        primary_map->dim*imp_list->sizes[i], MPI_INT, imp_list->ranks[i],
        primary_map->index, OP_PART_WORLD, MPI_STATUS_IGNORE);
  }

  MPI_Waitall(exp_list->ranks_size,request_send, MPI_STATUSES_IGNORE );
  for(int i=0; i < exp_list->ranks_size; i++) op_free(sbuf[i]); op_free(sbuf);

  int** adj = (int **)xmalloc(primary_map->to->size*sizeof(int *));
  int* adj_i = (int *)xmalloc(primary_map->to->size*sizeof(int ));
  int* adj_cap = (int *)xmalloc(primary_map->to->size*sizeof(int ));

  for(int i = 0; i<primary_map->to->size; i++)adj_i[i] = 0;
  for(int i = 0; i<primary_map->to->size; i++)adj_cap[i] = primary_map->dim;
  for(int i = 0; i<primary_map->to->size; i++)adj[i] = (int *)xmalloc(adj_cap[i]*sizeof(int));


  //go through each from-element of local primary_map and construct adjacency list
  for(int i = 0; i<primary_map->from->size; i++)
  {
    int part, local_index;
    for(int j=0; j < primary_map->dim; j++) { //for each element pointed at by this entry
      part = get_partition(primary_map->map[i*primary_map->dim+j],
          part_range[primary_map->to->index],&local_index,comm_size);

      if(part == my_rank)
      {
        for(int k = 0; k<primary_map->dim; k++)
        {
          if(adj_i[local_index] >= adj_cap[local_index])
          {
            adj_cap[local_index] = adj_cap[local_index]*2;
            adj[local_index] = (int *)xrealloc(adj[local_index],
                adj_cap[local_index]*sizeof(int));
          }
          adj[local_index][adj_i[local_index]++] =
            primary_map->map[i*primary_map->dim+k];
        }
      }
    }
  }
  //go through each from-element of foreign primary_map and add to adjacency list
  for(int i = 0; i<imp_list->size; i++)
  {
    int part, local_index;
    for(int j=0; j < primary_map->dim; j++) { //for each element pointed at by this entry
      part = get_partition(foreign_maps[i*primary_map->dim+j],
          part_range[primary_map->to->index],&local_index,comm_size);

      if(part == my_rank)
      {
        for(int k = 0; k<primary_map->dim; k++)
        {
          if(adj_i[local_index] >= adj_cap[local_index])
          {
            adj_cap[local_index] = adj_cap[local_index]*2;
            adj[local_index] = (int *)xrealloc(adj[local_index],
                adj_cap[local_index]*sizeof(int));
          }
          adj[local_index][adj_i[local_index]++] =
            foreign_maps[i*primary_map->dim+k];
        }
      }
    }
  }
  op_free(foreign_maps);

  //
  //Setup data structures for ParMetis PartKway
  //
  idx_t comm_size_pm = comm_size;

  idx_t *vtxdist = (idx_t *) xmalloc(sizeof(idx_t)*(comm_size+1));
  for(int i=0; i<comm_size; i++)
  {
    vtxdist[i] = part_range[primary_map->to->index][2*i];
  }
  vtxdist[comm_size] = part_range[primary_map->to->index][2*(comm_size-1)+1]+1;


  idx_t *xadj = (idx_t *)xmalloc(sizeof(idx_t)*(primary_map->to->size+1));
  cap = (primary_map->to->size)*primary_map->dim;

  idx_t *adjncy = (idx_t *)xmalloc(sizeof(idx_t)*cap);
  int count = 0;int prev_count = 0;
  for(int i = 0; i<primary_map->to->size; i++)
  {
    int g_index = get_global_index(i, my_rank,
        part_range[primary_map->to->index], comm_size);
    quickSort(adj[i], 0, adj_i[i]-1);
    adj_i[i] = removeDups(adj[i], adj_i[i]);


    if(adj_i[i] < 2)
    {
      printf("The from set: %s of primary map: %s is not an on to set of to-set: %s\n",
          primary_map->from->name, primary_map->name, primary_map->to->name);
      printf("Need to select a different primary map\n");
      MPI_Abort(OP_PART_WORLD, 2);
    }

    adj[i] = (int *)xrealloc(adj[i],adj_i[i]*sizeof(int));
    for(int j = 0; j<adj_i[i]; j++)
    {
      if(adj[i][j] != g_index)
      {
        if(count >= cap)
        {
          cap = cap*2;
          adjncy = (idx_t *)xrealloc(adjncy,sizeof(idx_t)*cap);
        }
        adjncy[count++] = (idx_t)adj[i][j];
      }
    }
    if(i != 0)
    {
      xadj[i] = prev_count;
      prev_count = count;
    }
    else
    {
      xadj[i] = 0;
      prev_count = count;
    }
  }
  xadj[primary_map->to->size] = count;

  // printf("On rank %d\n", my_rank);
  /* for(int i = 0; i<primary_map->to->size; i++)
    {
    if(xadj[i+1]-xadj[i]>8)printf("On rank %d, element %d, Size = %d\n",
    my_rank, i, xadj[i+1]-xadj[i]);
    }*/
  //printf("\n\n");

  for(int i = 0; i<primary_map->to->size; i++)op_free(adj[i]);
  op_free(adj_i);op_free(adj_cap);op_free(adj);


  idx_t *partition_pm = (idx_t *)xmalloc(sizeof(idx_t)*primary_map->to->size);
  for(int i = 0; i < primary_map->to->size; i++){ partition_pm[i] = -99; }

  idx_t edge_cut = 0;
  idx_t numflag = 0;
  idx_t wgtflag = 0;
  idx_t options[3] = {1,3,15};

  int *hybrid_flags = (int *)xmalloc(comm_size*sizeof(int));
  MPI_Allgather( &OP_hybrid_gpu, 1, MPI_INT,  hybrid_flags,
      1,MPI_INT,OP_PART_WORLD);
  double total = 0;
  for (int i = 0; i < comm_size; i++)
    total += hybrid_flags[i] == 1 ? OP_hybrid_balance : 1.0;

  idx_t ncon = 1;
  float *tpwgts = (float *)xmalloc(comm_size*sizeof(float)*ncon);
  for(int i = 0; i<comm_size*ncon; i++) tpwgts[i] = hybrid_flags[i] == 1 ? OP_hybrid_balance/total : 1.0/total;

  float *ubvec = (float *)xmalloc(sizeof(float)*ncon);
  *ubvec = 1.05;

  //clean up before calling ParMetis
  for(int i = 0; i<OP_set_index; i++)op_free(part_range[i]);op_free(part_range);
  op_free(imp_list->list);op_free(imp_list->disps);op_free(imp_list->ranks);op_free(imp_list->sizes);
  op_free(exp_list->list);op_free(exp_list->disps);op_free(exp_list->ranks);op_free(exp_list->sizes);
  op_free(imp_list);op_free(exp_list);

  if(my_rank==MPI_ROOT)
  { printf("-----------------------------------------------------------\n");
    printf("ParMETIS_V3_PartKway Output\n");
    printf("-----------------------------------------------------------\n");
  }
  ParMETIS_V3_PartKway(vtxdist, xadj, adjncy, NULL, NULL, &wgtflag, &numflag,
		       &ncon, &comm_size_pm,tpwgts, ubvec, options, &edge_cut, partition_pm, &OP_PART_WORLD);
  if(my_rank==MPI_ROOT)
    printf("-----------------------------------------------------------\n");
  op_free(vtxdist); op_free(xadj); op_free(adjncy);
  op_free(ubvec);op_free(tpwgts);


  int *partition = (int *)xmalloc(sizeof(int)*primary_map->to->size);
  for(int i = 0; i<primary_map->to->size; i++)
  {
    //sanity check to see if all elements were partitioned
    if(partition_pm[i]<0 || partition_pm[i]>=comm_size)
    {
      printf("Partitioning problem: on rank %d, set %s element %d not assigned a partition\n",
          my_rank,primary_map->to->name, i);
      MPI_Abort(OP_PART_WORLD, 2);
    }
    partition[i] = partition_pm[i];
  }
  free(partition_pm);

  //initialise primary set as partitioned
  OP_part_list[primary_map->to->index]->elem_part= partition;
  OP_part_list[primary_map->to->index]->is_partitioned = 1;

  /*-STEP 2 - Partition all other sets,migrate data and renumber mapping tables-*/

  //partition all other sets
  partition_all(primary_map->to, my_rank, comm_size);

  //migrate data, sort elements
  migrate_all(my_rank, comm_size);

  //renumber mapping tables
  renumber_maps(my_rank, comm_size);

  op_timers(&cpu_t2, &wall_t2);  //timer stop for partitioning
  //printf time for partitioning
  time = wall_t2-wall_t1;
  MPI_Reduce(&time,&max_time,1,MPI_DOUBLE, MPI_MAX,MPI_ROOT, OP_PART_WORLD);
  MPI_Comm_free(&OP_PART_WORLD);
  if(my_rank==MPI_ROOT)printf("Max total Kway partitioning time = %lf\n",max_time);

  free(request_send);
}

/*******************************************************************************
 * Wrapper routine to use ParMETIS PartGeomKway() which partitions the to-set
 * of an op_map using its XYZ Geometry Data
 *******************************************************************************/

void op_partition_geomkway(op_dat coords, op_map primary_map)
{
  //declare timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  double time;
  double max_time;

  op_timers(&cpu_t1, &wall_t1); //timer start for partitioning

  //create new communicator for partitioning
  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_PART_WORLD);
  MPI_Comm_rank(OP_PART_WORLD, &my_rank);
  MPI_Comm_size(OP_PART_WORLD, &comm_size);

  //check if coords->set and primary_map's to set is the same
  if (compare_sets( coords->set, primary_map->to) == 0)
  {
    printf("primary map's to set %s mismatches the op_dat's set %s: on rank %d\n",
        primary_map->to->name, coords->set->name, my_rank);
    MPI_Abort(OP_PART_WORLD, 2);
  }


  /*--STEP 0 - initialise partitioning data stauctures with the current (block)
    partitioning information */

  // Compute global partition range information for each set
  int** part_range = (int **)xmalloc(OP_set_index*sizeof(int*));
  get_part_range(part_range,my_rank,comm_size, OP_PART_WORLD);

  //save the original part_range for future partition reversing
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

  //allocate memory for list
  OP_part_list = (part *)xmalloc(OP_set_index*sizeof(part));

  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];
    //printf("set %s size = %d\n", set.name, set.size);
    int *g_index = (int *)xmalloc(sizeof(int)*set->size);
    for(int i = 0; i< set->size; i++)
      g_index[i] = get_global_index(i,my_rank, part_range[set->index],comm_size);
    decl_partition(set, g_index, NULL);
  }

  /*--- STEP 1 - Set up coordinates (1D,2D or 3D) data structures   ------------*/

  idx_t ndims = coords->dim;
  float* xyz = 0;

  // Create ParMetis compatible coordinates array
  //- i.e. coordinates should be floats
  if(ndims == 3 || ndims == 2 || ndims == 1)
  {
    xyz = (float* )xmalloc(coords->set->size*coords->dim*sizeof(float));
    size_t mult = coords->size/coords->dim;
    for(int i = 0;i < coords->set->size;i++)
    {
      double temp;
      for(int e = 0; e < coords->dim;e++)
      {
        memcpy(&temp, (void *)&(coords->data[(i*coords->dim+e)*mult]), mult);
        xyz[i*coords->dim + e] = (float)temp;
      }
    }
  }
  else
  {
    printf("Dimensions of Coordinate array not one of 3D,2D or 1D\n");
    printf("Not supported by ParMetis - Indicate correct coordinates array\n");
    MPI_Abort(OP_PART_WORLD, 1);
  }


  /*--STEP 1 - Construct adjacency list of the to-set of the primary_map -------*/

  //
  //create export list
  //
  int c = 0; int cap = 1000;
  int* list = (int *)xmalloc(cap*sizeof(int));//temp list

  for(int e=0; e < primary_map->from->size; e++) { //for each maping table entry
    int part, local_index;
    for(int j=0; j < primary_map->dim; j++) { //for each element pointed at by this entry
      part = get_partition(primary_map->map[e*primary_map->dim+j],
          part_range[primary_map->to->index],&local_index,comm_size);
      if(c>=cap)
      {
        cap = cap*2;
        list = (int *)xrealloc(list,cap*sizeof(int));
      }

      if(part != my_rank){
        list[c++] = part; //add to export list
        list[c++] = e;
      }
    }
  }
  halo_list exp_list= (halo_list)xmalloc(sizeof(halo_list_core));
  create_export_list(primary_map->from,list, exp_list, c, comm_size, my_rank);
  op_free(list);//free temp list

  //
  //create import list
  //
  int *neighbors, *sizes;
  int ranks_size;

  //-----Discover neighbors-----
  ranks_size = 0;
  neighbors = (int *)xmalloc(comm_size*sizeof(int));
  sizes = (int *)xmalloc(comm_size*sizeof(int));

  //halo_list list = OP_export_exec_list[set->index];
  find_neighbors_set(exp_list, neighbors, sizes, &ranks_size, my_rank,
      comm_size, OP_PART_WORLD);
  //  MPI_Request request_send[exp_list->ranks_size];
  MPI_Request* request_send = (MPI_Request *)xmalloc(exp_list->ranks_size*sizeof(MPI_Request));

  int* rbuf, index = 0;
  cap = 0;

  for(int i=0; i<exp_list->ranks_size; i++) {
    int* sbuf = &exp_list->list[exp_list->disps[i]];
    MPI_Isend( sbuf,  exp_list->sizes[i],  MPI_INT, exp_list->ranks[i],
        primary_map->index, OP_PART_WORLD, &request_send[i] );
  }

  for(int i=0; i< ranks_size; i++) cap = cap + sizes[i];
  int* temp = (int *)xmalloc(cap*sizeof(int));

  //import this list from those neighbors
  for(int i=0; i<ranks_size; i++) {
    rbuf = (int *)xmalloc(sizes[i]*sizeof(int));
    MPI_Recv(rbuf, sizes[i], MPI_INT, neighbors[i],primary_map->index,
        OP_PART_WORLD, MPI_STATUS_IGNORE );
    memcpy(&temp[index],(void *)&rbuf[0],sizes[i]*sizeof(int));
    index = index + sizes[i];
    op_free(rbuf);
  }
  MPI_Waitall(exp_list->ranks_size,request_send, MPI_STATUSES_IGNORE );

  halo_list imp_list= (halo_list)xmalloc(sizeof(halo_list_core));
  create_import_list(primary_map->from, temp, imp_list, index,neighbors, sizes,
      ranks_size, comm_size, my_rank);


  //
  //Exchange mapping table entries using the import/export lists
  //

  //prepare bits of the mapping tables to be exported
  int** sbuf = (int **)xmalloc(exp_list->ranks_size*sizeof(int *));

  for(int i=0; i < exp_list->ranks_size; i++) {
    sbuf[i] = (int *)xmalloc(exp_list->sizes[i]*primary_map->dim*sizeof(int));
    for(int j = 0; j < exp_list->sizes[i]; j++)
    {
      for(int p = 0; p < primary_map->dim; p++)
      {
        sbuf[i][j*primary_map->dim+p] =
          primary_map->map[primary_map->dim*
          (exp_list->list[exp_list->disps[i]+j])+p];
      }
    }
    MPI_Isend(sbuf[i],  primary_map->dim*exp_list->sizes[i],  MPI_INT,
        exp_list->ranks[i], primary_map->index, OP_PART_WORLD, &request_send[i]);
  }

  //prepare space for the incomming mapping tables
  int* foreign_maps = (int *)xmalloc(primary_map->dim*(imp_list->size)*sizeof(int));

  for(int i=0; i<imp_list->ranks_size; i++) {
    MPI_Recv(&foreign_maps[imp_list->disps[i]*primary_map->dim],
        primary_map->dim*imp_list->sizes[i], MPI_INT, imp_list->ranks[i],
        primary_map->index, OP_PART_WORLD, MPI_STATUS_IGNORE);
  }

  MPI_Waitall(exp_list->ranks_size,request_send, MPI_STATUSES_IGNORE );
  for(int i=0; i < exp_list->ranks_size; i++) op_free(sbuf[i]); op_free(sbuf);

  int** adj = (int **)xmalloc(primary_map->to->size*sizeof(int *));
  int* adj_i = (int *)xmalloc(primary_map->to->size*sizeof(int ));
  int* adj_cap = (int *)xmalloc(primary_map->to->size*sizeof(int ));

  for(int i = 0; i<primary_map->to->size; i++)adj_i[i] = 0;
  for(int i = 0; i<primary_map->to->size; i++)adj_cap[i] = primary_map->dim;
  for(int i = 0; i<primary_map->to->size; i++)adj[i] = (int *)xmalloc(adj_cap[i]*sizeof(int));


  //go through each from-element of local primary_map and construct adjacency list
  for(int i = 0; i<primary_map->from->size; i++)
  {
    int part, local_index;
    for(int j=0; j < primary_map->dim; j++) { //for each element pointed at by this entry
      part = get_partition(primary_map->map[i*primary_map->dim+j],
          part_range[primary_map->to->index],&local_index,comm_size);

      if(part == my_rank)
      {
        for(int k = 0; k<primary_map->dim; k++)
        {
          if(adj_i[local_index] >= adj_cap[local_index])
          {
            adj_cap[local_index] = adj_cap[local_index]*2;
            adj[local_index] = (int *)xrealloc(adj[local_index],
                adj_cap[local_index]*sizeof(int));
          }
          adj[local_index][adj_i[local_index]++] =
            primary_map->map[i*primary_map->dim+k];
        }
      }
    }
  }
  //go through each from-element of foreign primary_map and add to adjacency list
  for(int i = 0; i<imp_list->size; i++)
  {
    int part, local_index;
    for(int j=0; j < primary_map->dim; j++) { //for each element pointed at by this entry
      part = get_partition(foreign_maps[i*primary_map->dim+j],
          part_range[primary_map->to->index],&local_index,comm_size);

      if(part == my_rank)
      {
        for(int k = 0; k<primary_map->dim; k++)
        {
          if(adj_i[local_index] >= adj_cap[local_index])
          {
            adj_cap[local_index] = adj_cap[local_index]*2;
            adj[local_index] = (int *)xrealloc(adj[local_index],
                adj_cap[local_index]*sizeof(int));
          }
          adj[local_index][adj_i[local_index]++] =
            foreign_maps[i*primary_map->dim+k];
        }
      }
    }
  }
  op_free(foreign_maps);

  //
  //Setup data structures for ParMetis PartGeomKway
  //
  idx_t comm_size_pm = comm_size;

  idx_t *vtxdist = (idx_t *)xmalloc(sizeof(idx_t)*(comm_size+1));
  for(int i=0; i<comm_size; i++)
  {
    vtxdist[i] = part_range[primary_map->to->index][2*i];
  }
  vtxdist[comm_size] = part_range[primary_map->to->index][2*(comm_size-1)+1]+1;


  idx_t *xadj = (idx_t *)xmalloc(sizeof(idx_t)*(primary_map->to->size+1));
  cap = (primary_map->to->size)*primary_map->dim;

  idx_t *adjncy = (idx_t *)xmalloc(sizeof(idx_t)*cap);
  int count = 0;int prev_count = 0;
  for(int i = 0; i<primary_map->to->size; i++)
  {
    int g_index = get_global_index(i, my_rank,
        part_range[primary_map->to->index], comm_size);
    quickSort(adj[i], 0, adj_i[i]-1);
    adj_i[i] = removeDups(adj[i], adj_i[i]);


    if(adj_i[i] < 2)
    {
      printf("The from set: %s of primary map: %s is not an on to set of to-set: %s\n",
          primary_map->from->name, primary_map->name, primary_map->to->name);
      printf("Need to select a different primary map\n");
      MPI_Abort(OP_PART_WORLD, 2);
    }

    adj[i] = (int *)xrealloc(adj[i],adj_i[i]*sizeof(int));
    for(int j = 0; j<adj_i[i]; j++)
    {
      if(adj[i][j] != g_index)
      {
        if(count >= cap)
        {
          cap = cap*2;
          adjncy = (idx_t *)xrealloc(adjncy,sizeof(idx_t)*cap);
        }
        adjncy[count++] = (idx_t)adj[i][j];
      }
    }
    if(i != 0)
    {
      xadj[i] = prev_count;
      prev_count = count;
    }
    else
    {
      xadj[i] = 0;
      prev_count = count;
    }
  }
  xadj[primary_map->to->size] = count;

  for(int i = 0; i<primary_map->to->size; i++)op_free(adj[i]);
  op_free(adj_i);op_free(adj_cap);op_free(adj);

  idx_t *partition = (idx_t *)xmalloc(sizeof(idx_t)*primary_map->to->size);
  for(int i = 0; i < primary_map->to->size; i++){ partition[i] = -99; }

  idx_t edge_cut = 0;
  idx_t numflag = 0;
  idx_t wgtflag = 0;
  idx_t options[3] = {1,3,15};

  idx_t ncon = 1;
  float *tpwgts = (float *)xmalloc(comm_size*sizeof(float)*ncon);
  for(int i = 0; i<comm_size*ncon; i++)tpwgts[i] = (float)1.0/(float)comm_size;

  float *ubvec = (float *)xmalloc(sizeof(float)*ncon);
  *ubvec = 1.05;

  //clean up before calling ParMetis
  for(int i = 0; i<OP_set_index; i++)op_free(part_range[i]);op_free(part_range);
  op_free(imp_list->list);op_free(imp_list->disps);op_free(imp_list->ranks);op_free(imp_list->sizes);
  op_free(exp_list->list);op_free(exp_list->disps);op_free(exp_list->ranks);op_free(exp_list->sizes);
  op_free(imp_list);op_free(exp_list);

  if(my_rank==MPI_ROOT)
  { printf("-----------------------------------------------------------\n");
    printf("ParMETIS_V3_PartGeomKway Output\n");
    printf("-----------------------------------------------------------\n");
  }
  ParMETIS_V3_PartGeomKway(vtxdist, xadj, adjncy, NULL, NULL, &wgtflag, &numflag,
			   &ndims, xyz, &ncon, &comm_size_pm,tpwgts, ubvec, options, &edge_cut, partition,
      &OP_PART_WORLD);

  if(my_rank==MPI_ROOT)
    printf("-----------------------------------------------------------\n");

  op_free(vtxdist); op_free(xadj); op_free(adjncy);
  op_free(ubvec);op_free(tpwgts);op_free(xyz);


  //saniti check to see if all elements were partitioned
  for(int i = 0; i<primary_map->to->size; i++)
  {
    if(partition[i]<0)
    {
      printf("Partitioning problem: on rank %d, set %s element %d not assigned a partition\n",
          my_rank,primary_map->to->name, i);
      MPI_Abort(OP_PART_WORLD, 2);
    }
  }

  //initialise primary set as partitioned
  OP_part_list[coords->set->index]->elem_part= (int *) partition;
  OP_part_list[coords->set->index]->is_partitioned = 1;

  /*-STEP 2 - Partition all other sets,migrate data and renumber mapping tables-*/

  //partition all other sets
  partition_all(primary_map->to, my_rank, comm_size);

  //migrate data, sort elements
  migrate_all(my_rank, comm_size);

  //renumber mapping tables
  renumber_maps(my_rank, comm_size);

  op_timers(&cpu_t2, &wall_t2);  //timer stop for partitioning
  //printf time for partitioning
  time = wall_t2-wall_t1;
  MPI_Reduce(&time,&max_time,1,MPI_DOUBLE, MPI_MAX,MPI_ROOT, OP_PART_WORLD);
  MPI_Comm_free(&OP_PART_WORLD);
  if(my_rank==MPI_ROOT)printf("Max total geometric k-way partitioning time = %lf\n",max_time);

  free(request_send);
}

/*******************************************************************************
 * Wrapper routine to use ParMETIS PartMeshKway() which partitions a the to-set
 * of an op_map using the from-set of the op_map
 *******************************************************************************/

void op_partition_meshkway(op_map primary_map) //not working !!
{
  ///The to-set elements of the primary_map is considerd to be
  ///the mesh elements used in the partitioning

  //declare timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  double time;
  double max_time;

  op_timers(&cpu_t1, &wall_t1); //timer start for partitioning

  //create new communicator for partitioning
  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_PART_WORLD);
  MPI_Comm_rank(OP_PART_WORLD, &my_rank);
  MPI_Comm_size(OP_PART_WORLD, &comm_size);

#ifdef DEBUG
  //check if the  primary_map is an on to map from the from-set to the to-set
  if(is_onto_map(primary_map) != 1)
  {
    printf("Map %s is an not an onto map from set %s to set %s \n",
        primary_map->name, primary_map->from->name, primary_map->to->name);
    MPI_Abort(OP_PART_WORLD, 2);
  }
#endif

  ///
  ///WE NEED TO DO SOME CHECKS ON THE PRIMARY MAP TO SEE IF IT IS
  ///A SUITABLE ELEMENT ? - not sure if the following is all we need as a check
  ///
  if(primary_map->dim < 3)
  {
    printf("Unsuitable primary map for using ParMETIS_V3_PartMeshKway\n");
    MPI_Abort(OP_PART_WORLD, 2);
  }

  /*--STEP 0 - initialise partitioning data stauctures with the current (block)
    partitioning information */

  // Compute global partition range information for each set
  int** part_range = (int **)xmalloc(OP_set_index*sizeof(int*));
  get_part_range(part_range,my_rank,comm_size, OP_PART_WORLD);

  //save the original part_range for future partition reversing
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

  //allocate memory for list
  OP_part_list = (part *)xmalloc(OP_set_index*sizeof(part));

  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];
    //printf("set %s size = %d\n", set.name, set.size);
    int *g_index = (int *)xmalloc(sizeof(int)*set->size);
    for(int i = 0; i< set->size; i++)
      g_index[i] = get_global_index(i,my_rank, part_range[set->index],comm_size);
    decl_partition(set, g_index, NULL);
  }

  /*--STEP 1 - Construct adjacency list of the to-set of the primary_map
    including eptr and eind arrays  ------------------------------------------*/


  //
  //Setup data structures for ParMetis PartMeshKway
  //
  idx_t comm_size_pm = comm_size;

  idx_t *elemdist = (idx_t *)xmalloc(sizeof(idx_t)*(comm_size+1));
  for(int i=0; i<comm_size; i++)
  {
    elemdist[i] = part_range[primary_map->from->index][2*i];
  }
  elemdist[comm_size] = part_range[primary_map->from->index][2*(comm_size-1) + 1] + 1;

  idx_t *eind = (idx_t *)xmalloc(sizeof(idx_t)*(primary_map->from->size)*primary_map->dim);
  idx_t *eptr = (idx_t *)xmalloc(sizeof(idx_t)*(primary_map->from->size + 1));

  //setup the eind
  for(int i=0; i<primary_map->from->size; i++)
  {
    eptr[i] = primary_map->dim*i;
  }
  eptr[primary_map->from->size] = primary_map->dim*primary_map->from->size;

  //make a copy of the primary_map into the eptr
  for(int i=0; i<primary_map->from->size; i++)
  {
    for(int j = 0; j<primary_map->dim; j++)
    {
      //eind[eptr[i]+j] = (idx_t) OP_map_list[primary_map->index]->map[eptr[i]+j];
      eind[primary_map->dim*i+j] =
        (idx_t) OP_map_list[primary_map->index]->map[primary_map->dim*i+j];
    }
  }
  //memcpy(&eind[0], primary_map->map,
  //  (primary_map->from->size)*primary_map->dim*sizeof(int));

  idx_t *partition = (idx_t *)xmalloc(sizeof(idx_t)*primary_map->to->size);
  for(int i = 0; i < primary_map->to->size; i++){ partition[i] = -99; }

  idx_t edge_cut = 0;
  idx_t numflag = 0;
  idx_t wgtflag = 0;
  idx_t options[3] = {1,3,15};

  idx_t ncon = 1;
  float *tpwgts = (float *)xmalloc(comm_size*sizeof(float)*ncon);
  for(int i = 0; i<comm_size*ncon; i++)tpwgts[i] = (float)1.0/(float)comm_size;

  float *ubvec = (float *)xmalloc(sizeof(float)*ncon);
  *ubvec = 1.05;

  idx_t ncommonnodes = 1;

  if(my_rank==MPI_ROOT)
  { printf("-----------------------------------------------------------\n");
    printf("ParMETIS_V3_PartMeshKway Output\n");
    printf("-----------------------------------------------------------\n");
  }
  ParMETIS_V3_PartMeshKway(elemdist, eptr, eind, NULL, &wgtflag, &numflag,
      &ncon, &ncommonnodes, &comm_size_pm,tpwgts, ubvec, options, &edge_cut,
      partition, &OP_PART_WORLD);
  if(my_rank==MPI_ROOT)
    printf("-----------------------------------------------------------\n");
  op_free(elemdist); op_free(eptr); op_free(eind);
  op_free(ubvec);op_free(tpwgts);

  //check if all to-set elements have been partitioned
  for(int i = 0; i<primary_map->to->size; i++)
  {
    if(partition[i]<0)
    {
      printf("Partitioning problem on rank %d, set %s element %d not found\n",
          my_rank,primary_map->to->name, i);
      MPI_Abort(OP_PART_WORLD, 2);
    }
  }

  //initialise primary set as partitioned
  OP_part_list[primary_map->to->index]->elem_part= (int *) partition;
  OP_part_list[primary_map->to->index]->is_partitioned = 1;

  /*-STEP 2 - Partition all other sets,migrate data and renumber mapping tables-*/
  /*
  //partition all other sets
  partition_all(primary_map->to, my_rank, comm_size);

  //migrate data, sort elements
  migrate_all(my_rank, comm_size);

  //renumber mapping tables
  renumber_maps(my_rank, comm_size);
  */
  op_timers(&cpu_t2, &wall_t2);  //timer stop for partitioning
  //printf time for partitioning
  time = wall_t2-wall_t1;
  MPI_Reduce(&time,&max_time,1,MPI_DOUBLE, MPI_MAX,MPI_ROOT, OP_PART_WORLD);
  MPI_Comm_free(&OP_PART_WORLD);
  if(my_rank==MPI_ROOT)printf("Max total MeshKway partitioning time = %lf\n",max_time);

}

#endif

#ifdef HAVE_PTSCOTCH

void op_partition_ptscotch(op_map primary_map)
{
  //declare timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  double time;
  double max_time;

  op_timers(&cpu_t1, &wall_t1); //timer start for partitioning

  //create new communicator for partitioning
  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_PART_WORLD);
  MPI_Comm_rank(OP_PART_WORLD, &my_rank);
  MPI_Comm_size(OP_PART_WORLD, &comm_size);

#ifdef DEBUG
  //check if the  primary_map is an on to map from the from-set to the to-set
  if(is_onto_map(primary_map) != 1)
  {
    printf("Map %s is an not an onto map from set %s to set %s \n",
        primary_map->name, primary_map->from->name, primary_map->to->name);
    MPI_Abort(OP_PART_WORLD, 2);
  }
#endif

  /*--STEP 0 - initialise partitioning data stauctures with the current (block)
    partitioning information */

  // Compute global partition range information for each set
  int** part_range = (int **)xmalloc(OP_set_index*sizeof(int*));
  get_part_range(part_range,my_rank,comm_size, OP_PART_WORLD);

  //save the original part_range for future partition reversing
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

  //allocate memory for list
  OP_part_list = (part *)xmalloc(OP_set_index*sizeof(part));

  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];
    //printf("set %s size = %d\n", set.name, set.size);
    int *g_index = (int *)xmalloc(sizeof(int)*set->size);
    for(int i = 0; i< set->size; i++)
      g_index[i] = get_global_index(i,my_rank, part_range[set->index],comm_size);
    decl_partition(set, g_index, NULL);
  }

  /*--STEP 1 - Construct adjacency list of the to-set of the primary_map -------*/

  //
  //create export list
  //
  int c = 0; int cap = 1000;
  int* list = (int *)xmalloc(cap*sizeof(int));//temp list

  for(int e=0; e < primary_map->from->size; e++) { //for each maping table entry
    int part, local_index;
    for(int j=0; j < primary_map->dim; j++) { //for each element pointed at by this entry
      part = get_partition(primary_map->map[e*primary_map->dim+j],
          part_range[primary_map->to->index],&local_index,comm_size);
      if(c>=cap)
      {
        cap = cap*2;
        list = (int *)xrealloc(list,cap*sizeof(int));
      }

      if(part != my_rank){
        list[c++] = part; //add to export list
        list[c++] = e;
      }
    }
  }
  halo_list exp_list= (halo_list)xmalloc(sizeof(halo_list_core));
  create_export_list(primary_map->from,list, exp_list, c, comm_size, my_rank);
  op_free(list);//free temp list

  //
  //create import list
  //
  int *neighbors, *sizes;
  int ranks_size;

  //-----Discover neighbors-----
  ranks_size = 0;
  neighbors = (int *)xmalloc(comm_size*sizeof(int));
  sizes = (int *)xmalloc(comm_size*sizeof(int));

  //halo_list list = OP_export_exec_list[set->index];
  find_neighbors_set(exp_list, neighbors, sizes, &ranks_size, my_rank,
      comm_size, OP_PART_WORLD);
  //  MPI_Request request_send[exp_list->ranks_size];
  MPI_Request* request_send = (MPI_Request *)xmalloc(exp_list->ranks_size*sizeof(MPI_Request));

  int* rbuf, index = 0;
  cap = 0;

  for(int i=0; i<exp_list->ranks_size; i++) {
    int* sbuf = &exp_list->list[exp_list->disps[i]];
    MPI_Isend( sbuf,  exp_list->sizes[i],  MPI_INT, exp_list->ranks[i],
        primary_map->index, OP_PART_WORLD, &request_send[i] );
  }

  for(int i=0; i< ranks_size; i++) cap = cap + sizes[i];
  int* temp = (int *)xmalloc(cap*sizeof(int));

  //import this list from those neighbors
  for(int i=0; i<ranks_size; i++) {
    rbuf = (int *)xmalloc(sizes[i]*sizeof(int));
    MPI_Recv(rbuf, sizes[i], MPI_INT, neighbors[i],primary_map->index,
        OP_PART_WORLD, MPI_STATUS_IGNORE );
    memcpy(&temp[index],(void *)&rbuf[0],sizes[i]*sizeof(int));
    index = index + sizes[i];
    op_free(rbuf);
  }
  MPI_Waitall(exp_list->ranks_size,request_send, MPI_STATUSES_IGNORE );

  halo_list imp_list= (halo_list)xmalloc(sizeof(halo_list_core));
  create_import_list(primary_map->from, temp, imp_list, index,neighbors, sizes,
      ranks_size, comm_size, my_rank);


  //
  //Exchange mapping table entries using the import/export lists
  //

  //prepare bits of the mapping tables to be exported
  int** sbuf = (int **)xmalloc(exp_list->ranks_size*sizeof(int *));

  for(int i=0; i < exp_list->ranks_size; i++) {
    sbuf[i] = (int *)xmalloc(exp_list->sizes[i]*primary_map->dim*sizeof(int));
    for(int j = 0; j < exp_list->sizes[i]; j++)
    {
      for(int p = 0; p < primary_map->dim; p++)
      {
        sbuf[i][j*primary_map->dim+p] =
          primary_map->map[primary_map->dim*
          (exp_list->list[exp_list->disps[i]+j])+p];
      }
    }
    MPI_Isend(sbuf[i],  primary_map->dim*exp_list->sizes[i],  MPI_INT,
        exp_list->ranks[i], primary_map->index, OP_PART_WORLD, &request_send[i]);
  }

  //prepare space for the incomming mapping tables
  int* foreign_maps = (int *)xmalloc(primary_map->dim*(imp_list->size)*sizeof(int));

  for(int i=0; i<imp_list->ranks_size; i++) {
    MPI_Recv(&foreign_maps[imp_list->disps[i]*primary_map->dim],
        primary_map->dim*imp_list->sizes[i], MPI_INT, imp_list->ranks[i],
        primary_map->index, OP_PART_WORLD, MPI_STATUS_IGNORE);
  }

  MPI_Waitall(exp_list->ranks_size,request_send, MPI_STATUSES_IGNORE );
  for(int i=0; i < exp_list->ranks_size; i++) op_free(sbuf[i]); op_free(sbuf);

  int** adj = (int **)xmalloc(primary_map->to->size*sizeof(int *));
  int* adj_i = (int *)xmalloc(primary_map->to->size*sizeof(int ));
  int* adj_cap = (int *)xmalloc(primary_map->to->size*sizeof(int ));

  for(int i = 0; i<primary_map->to->size; i++)adj_i[i] = 0;
  for(int i = 0; i<primary_map->to->size; i++)adj_cap[i] = primary_map->dim;
  for(int i = 0; i<primary_map->to->size; i++)adj[i] = NULL;
  for(int i = 0; i<primary_map->to->size; i++)adj[i] = (int *)xmalloc(adj_cap[i]*sizeof(int));
  //for(int i = 0; i<primary_map->to->size; i++)adj[i] = (int *)xcalloc(adj_cap[i], sizeof(int));


  //go through each from-element of local primary_map and construct adjacency list
  for(int i = 0; i<primary_map->from->size; i++)
  {
    int part, local_index;
    for(int j=0; j < primary_map->dim; j++) { //for each element pointed at by this entry
      part = get_partition(primary_map->map[i*primary_map->dim+j],
          part_range[primary_map->to->index],&local_index,comm_size);

      if(part == my_rank)
      {
        for(int k = 0; k<primary_map->dim; k++)
        {
          if(adj_i[local_index] >= adj_cap[local_index])
          {
            adj_cap[local_index] = adj_cap[local_index]*2;
            adj[local_index] = (int *)xrealloc(adj[local_index],
                adj_cap[local_index]*sizeof(int));
          }
          adj[local_index][adj_i[local_index]++] =
            primary_map->map[i*primary_map->dim+k];
        }
      }
    }
  }
  //go through each from-element of foreign primary_map and add to adjacency list
  for(int i = 0; i<imp_list->size; i++)
  {
    int part, local_index;
    for(int j=0; j < primary_map->dim; j++) { //for each element pointed at by this entry
      part = get_partition(foreign_maps[i*primary_map->dim+j],
          part_range[primary_map->to->index],&local_index,comm_size);

      if(part == my_rank)
      {
        for(int k = 0; k<primary_map->dim; k++)
        {
          if(adj_i[local_index] >= adj_cap[local_index])
          {
            adj_cap[local_index] = adj_cap[local_index]*2;
            adj[local_index] = (int *)xrealloc(adj[local_index],
                adj_cap[local_index]*sizeof(int));
          }
          adj[local_index][adj_i[local_index]++] =
            foreign_maps[i*primary_map->dim+k];
        }
      }
    }
  }
  op_free(foreign_maps);


  //
  //Setup data structures for PT-Scotch
  //

  SCOTCH_Dgraph *grafptr = SCOTCH_dgraphAlloc();
  SCOTCH_dgraphInit(grafptr, OP_PART_WORLD);

  SCOTCH_Num baseval = 0;

  //vertex local number - number of vertexes on local mpi rank
  SCOTCH_Num vertlocnbr =   primary_map->to->size;

  //vertex local max - put same value as vertlocnbr
  SCOTCH_Num vertlocmax =  vertlocnbr;

  //local vertex adjacency index array, of size (vertlocnbr+1)
  SCOTCH_Num *vertloctab = (SCOTCH_Num *)xmalloc(sizeof(SCOTCH_Num)*(vertlocnbr+1));
  cap = (primary_map->to->size)*primary_map->dim;

  SCOTCH_Num *vendloctab = NULL;//not needed
  SCOTCH_Num *veloloctab = NULL;//not needed
  SCOTCH_Num *vlblocltab = NULL;//not needed

  //the local adjacency array, of size at least edgelocsiz,
  //which stores the global indices of end vertices
  SCOTCH_Num *edgeloctab = (SCOTCH_Num *)xmalloc(sizeof(SCOTCH_Num)*cap);
  int count = 0;int prev_count = 0;

  for(int i = 0; i<primary_map->to->size; i++)
  {
    int g_index = get_global_index(i, my_rank,
        part_range[primary_map->to->index], comm_size);
    quickSort(adj[i], 0, adj_i[i]-1);
    adj_i[i] = removeDups(adj[i], adj_i[i]);


    if(adj_i[i] < 2)
    {
      printf("The from set: %s of primary map: %s is not an on to set of to-set: %s\n",
          primary_map->from->name, primary_map->name, primary_map->to->name);
      printf("Need to select a different primary map\n");
      MPI_Abort(OP_PART_WORLD, 2);
    }

    adj[i] = (int *)xrealloc(adj[i],adj_i[i]*sizeof(int));
    for(int j = 0; j<adj_i[i]; j++)
    {
      if(adj[i][j] != g_index)
      {
        if(count >= cap)
        {
          cap = cap*2;
          edgeloctab = (SCOTCH_Num *)xrealloc(edgeloctab,sizeof(SCOTCH_Num)*cap);
        }
        edgeloctab[count++] = (SCOTCH_Num)adj[i][j];
      }
    }
    if(i != 0)
    {
      vertloctab[i] = prev_count;
      prev_count = count;
    }
    else
    {
      vertloctab[i] = 0;
      prev_count = count;
    }
  }
  vertloctab[primary_map->to->size] = count;

  //local number of arcs (that is, twice the number of edges)
  SCOTCH_Num edgelocnbr = count;
  SCOTCH_Num edgelocsiz = edgelocnbr; // equal to edgelocnbr

  for(int i = 0; i<primary_map->to->size; i++)op_free(adj[i]);
  op_free(adj_i);op_free(adj_cap);op_free(adj);

  SCOTCH_Num *edgegsttab = NULL; //not needed
  SCOTCH_Num *edloloctab = NULL;//not needed

  //clean up before calling Partitioner
  for(int i = 0; i<OP_set_index; i++)op_free(part_range[i]);op_free(part_range);
  op_free(imp_list->list);op_free(imp_list->disps);op_free(imp_list->ranks);op_free(imp_list->sizes);
  op_free(exp_list->list);op_free(exp_list->disps);op_free(exp_list->ranks);op_free(exp_list->sizes);
  op_free(imp_list);op_free(exp_list);

  //build a PT-Scotch graph
  SCOTCH_dgraphBuild(grafptr, baseval, vertlocnbr, vertlocmax, vertloctab,
      vendloctab, veloloctab, vlblocltab, edgelocnbr, edgelocsiz,
      edgeloctab, edgegsttab, edloloctab);
  int test = SCOTCH_dgraphCheck(grafptr);
  if(test == 1)
  {
    printf("PT-Scotch Graph Inconsistant - Aborting\n");
    MPI_Abort(OP_PART_WORLD, 2);
  }

  SCOTCH_Num *partloctab = (SCOTCH_Num *)xmalloc(sizeof(SCOTCH_Num)*primary_map->to->size);
  for(int i = 0; i < primary_map->to->size; i++){ partloctab[i] = -99; }

  //initialise partition strategy struct
  SCOTCH_Strat straptr;
  SCOTCH_stratInit(&straptr);

  //SCOTCH_stratDgraphMapBuild(&straptr, SCOTCH_STRATQUALITY/*SCOTCH_STRATSCALABILITY*/,
  //comm_size, comm_size, 1.05);

  //partition the graph
  SCOTCH_dgraphPart(grafptr, comm_size, &straptr, partloctab);
  op_free(edgeloctab);op_free(vertloctab);

  //saniti check to see if all elements were partitioned
  for(int i = 0; i<primary_map->to->size; i++)
  {
    if(partloctab[i]<0)
    {
      printf("Partitioning problem: on rank %d, set %s element %d not assigned a partition\n",
          my_rank,primary_map->to->name, i);
      MPI_Abort(OP_PART_WORLD, 2);
    }
  }

  //free strat struct
  SCOTCH_stratExit(&straptr);

  //free PT-Scotch allocated memory space
  op_free(grafptr);

  //initialise primary set as partitioned
  OP_part_list[primary_map->to->index]->elem_part= partloctab;
  OP_part_list[primary_map->to->index]->is_partitioned = 1;

  /*-STEP 2 - Partition all other sets,migrate data and renumber mapping tables-*/

  //partition all other sets
  partition_all(primary_map->to, my_rank, comm_size);

  //migrate data, sort elements
  migrate_all(my_rank, comm_size);

  //renumber mapping tables
  renumber_maps(my_rank, comm_size);

  op_timers(&cpu_t2, &wall_t2);  //timer stop for partitioning
  //printf time for partitioning
  time = wall_t2-wall_t1;
  MPI_Reduce(&time,&max_time,1,MPI_DOUBLE, MPI_MAX,MPI_ROOT, OP_PART_WORLD);
  MPI_Comm_free(&OP_PART_WORLD);
  if(my_rank==MPI_ROOT)printf("Max total PT-Scotch partitioning time = %lf\n",max_time);

  free(request_send);
}

#endif

/*******************************************************************************
 * Use OPlus style recursive bisection in the inertial directions
 *******************************************************************************/

void op_partition_inertial(op_dat x_dat)
{
  //declare timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  double time;
  double max_time;
  double *x = (double *)xmalloc(x_dat->set->size*x_dat->dim*sizeof(double));
  memcpy(x, x_dat->data, x_dat->set->size*x_dat->dim*sizeof(double));

  op_timers(&cpu_t1, &wall_t1); //timer start for partitioning

  //create new communicator for partitioning
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
  int** part_range = (int **)xmalloc(OP_set_index*sizeof(int*));
  get_part_range(part_range,my_rank,comm_size, OP_PART_WORLD);

  //save the original part_range for future partition reversing
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

  //allocate memory for list
  OP_part_list = (part *)xmalloc(OP_set_index*sizeof(part));

  for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];
    int *g_index = (int *)xmalloc(sizeof(int)*set->size);
    for(int i = 0; i< set->size; i++)
      g_index[i] = get_global_index(i,my_rank, part_range[set->index],comm_size);
    decl_partition(set, g_index, NULL);
  }

  /* - STEP 1 figure out partitioning - */
  int global_size = part_range[x_dat->set->index][2*(comm_size-1)+1]+1; //losg
  int block_lower = part_range[x_dat->set->index][2*my_rank]; //losg1
  int block_upper = part_range[x_dat->set->index][2*my_rank+1]; //losg2
  int block_size = block_upper-block_lower+1; //losgd

  int *global_indices = (int *)xmalloc(block_size*sizeof(int));
  for (int i = 0; i < block_size; i++) global_indices[i] = block_lower + i;
  int nlevel = 0;
  while ((1<<nlevel) < comm_size) nlevel++;

  int current_part_size = block_size; //losl
  int current_group_size = global_size; //lopl

  MPI_Request s_request,s_request2;
  MPI_Status s_status,s_status2;
  MPI_Status r_status;

  double *dist;
  double mtx[9];
  double mtx_global[9];
  double q[3], p[3], mue;
  for (int level = 0; level < nlevel; level++) {
    //op_inert begin
    if (comm_size != 1) {
      dist = (double *)xmalloc(current_part_size*sizeof(double));
      double x_local = 0.0;
      double y_local = 0.0;
      double z_local = 0.0;
      for (int i = 0; i < current_part_size; i++) {
        x_local += x[3*i];
        y_local += x[3*i+1];
        z_local += x[3*i+2];
      }
      double x_global = 0.0;
      double y_global = 0.0;
      double z_global = 0.0;
      MPI_Allreduce(&x_local, &x_global, 1, MPI_DOUBLE,
        MPI_SUM, mpi_comm);
      MPI_Allreduce(&y_local, &y_global, 1, MPI_DOUBLE,
        MPI_SUM, mpi_comm);
      MPI_Allreduce(&z_local, &z_global, 1, MPI_DOUBLE,
        MPI_SUM, mpi_comm);
      x_global /= (double)current_group_size;
      y_global /= (double)current_group_size;
      z_global /= (double)current_group_size;
      // printf("Centre of gravity: %5.14f %5.14f %5.14f\n", x_global, y_global, z_global);
      for (int i = 0; i < 9; i++) mtx[i]= 0.0;
      for (int i = 0; i < 9; i++) mtx_global[i]= 0.0;
      for (int i = 0; i < current_part_size; i++) {
        mtx[0] += (x[3*i]-x_global)*(x[3*i]-x_global);
        mtx[4] += (x[3*i+1]-y_global)*(x[3*i+1]-y_global);
        mtx[8] += (x[3*i+2]-z_global)*(x[3*i+2]-z_global);
        mtx[1] += (x[3*i]-x_global)*(x[3*i+1]-y_global);
        mtx[2] += (x[3*i]-x_global)*(x[3*i+2]-z_global);
        mtx[5] += (x[3*i+1]-y_global)*(x[3*i+2]-z_global);
      }
      mtx[3] = mtx[1];
      mtx[6] = mtx[2];
      mtx[7] = mtx[5];
      MPI_Allreduce(mtx, mtx_global, 9, MPI_DOUBLE,
        MPI_SUM, mpi_comm);

      q[0] = 0.0; q[1] = 0.0; q[2] = 0.0;
      if (mtx_global[0] > mtx_global[4] && mtx_global[0] > mtx_global[8]) q[0] = 1.0;
      if (mtx_global[4] > mtx_global[0] && mtx_global[4] > mtx_global[8]) q[1] = 1.0;
      if (mtx_global[8] > mtx_global[0] && mtx_global[8] > mtx_global[4]) q[2] = 1.0;

      int iter = 0;
      double err = 1.0;
      while (iter++<20000 && err > 1e-9) {
        // p = A*x
        for (int i = 0; i < 3; i++) {
          p[i] = 0.0;
          for (int j = 0; j < 3; j++) {
            p[i] += mtx_global[i*3+j] * q[j];
          }
        }
        //mue = qT*p
        mue = 0.0;
        for (int i = 0; i < 3; i++)
          mue += q[i]*p[i];
        double sum = 0.0;
        for (int i = 0; i < 3; i++)
          sum += p[i]*p[i];
        sum = 1.0/sqrt(sum);
        for (int i = 0; i < 3; i++)
          p[i] *= sum;
        err = 0.0;
        for (int i = 0; i < 3; i++)
          err += (q[i]-p[i])*(q[i]-p[i]);
        err = sqrt(err);
        for (int i = 0; i < 3; i++)
          q[i] = p[i];
      }
      // printf("Converged %d %5.14f (%5.14f %5.14f %5.14f)\n",iter,mue,q[0],q[1],q[2]);
      for (int i = 0; i < current_part_size; i++)
        dist[i] = x[3*i]*p[0] + x[3*i+1]*p[1] + x[3*i+2]*p[2];
    //op_inert end

    //op_sort begin
      double distmin = dist[0];
      double distmax = dist[0];
      double distavg = dist[0];
      for (int i = 1; i < current_part_size; i++) {
        distmin = distmin < dist[i] ? distmin : dist[i];
        distmax = distmax > dist[i] ? distmax : dist[i];
        distavg += dist[i];
      }
      double distmin_g = distmin;
      double distmax_g = distmax;
      double distavg_g = 0.0;
      MPI_Allreduce(&distmin, &distmin_g, 1, MPI_DOUBLE,
        MPI_MIN, mpi_comm);
      MPI_Allreduce(&distmax, &distmax_g, 1, MPI_DOUBLE,
        MPI_MAX, mpi_comm);
      MPI_Allreduce(&distavg, &distavg_g, 1, MPI_DOUBLE,
        MPI_SUM, mpi_comm);
      distavg_g /= (double)current_group_size;

      double dbnd = distmax_g - distmin_g;
      double dlower = distmin_g;
      double dupper = distmax_g;
      double dsplit = distavg_g;
      int nsplit = (current_group_size * (comm_size/2))/comm_size;
      int nlower_g = 0;
      while(1) {
        int nlower = 0;
        nlower_g = 0;
        for (int i = 0; i < current_part_size; i++)
          nlower += (dist[i] <= dsplit ? 1 : 0);
        MPI_Allreduce(&nlower, &nlower_g, 1, MPI_INT,
          MPI_SUM, mpi_comm);
        if (nlower_g == nsplit) break;
        else if (nlower_g < nsplit) {dlower = dsplit; dsplit = (dsplit+dupper)/2.0;}
        else if (nlower_g > nsplit) {dupper = dsplit; dsplit = (dsplit+dlower)/2.0;}

        if (dupper-dlower < 1e-8*dbnd) break;
      }
      int current_group_lower = nlower_g;
      int current_group_upper = current_group_size - nlower_g;

      double *x_keep = (double *)xmalloc(3*current_part_size*sizeof(double));
      int *idx_gbl_keep = (int *)xmalloc(current_part_size*sizeof(int));
      double *x_send = (double *)xmalloc(3*current_part_size*sizeof(double));
      int *idx_gbl_send = (int *)xmalloc(current_part_size*sizeof(int));
      int keep_ctr = 0;
      int send_ctr = 0;
      if (my_rank <= comm_size/2-1) {
        for (int i = 0; i < current_part_size; i++)  {
          if (dist[i] <= dsplit)  {
            idx_gbl_keep[keep_ctr] = global_indices[i];
            for (int j = 0; j < 3; j++) x_keep[3*keep_ctr+j] = x[3*i+j];
            keep_ctr++;
          } else {
            idx_gbl_send[send_ctr] = global_indices[i];
            for (int j = 0; j < 3; j++) x_send[3*send_ctr+j] = x[3*i+j];
            send_ctr++;
          }
        }
      } else {
        for (int i = 0; i < current_part_size; i++)  {
          if (dist[i] > dsplit)  {
            idx_gbl_keep[keep_ctr] = global_indices[i];
            for (int j = 0; j < 3; j++) x_keep[3*keep_ctr+j] = x[3*i+j];
            keep_ctr++;
          } else {
            idx_gbl_send[send_ctr] = global_indices[i];
            for (int j = 0; j < 3; j++) x_send[3*send_ctr+j] = x[3*i+j];
            send_ctr++;
          }
        }
      }

      int target_part = comm_size - my_rank - 1;
      if (my_rank == target_part) target_part--;

    //Send send buffer size
      MPI_Isend(&send_ctr, 1, MPI_INT, target_part, 0, mpi_comm, &s_request);

    //Receive send buffer sizes
      int size_0=0;
      int size_1=0;
      if (2*my_rank != (comm_size-1))
        MPI_Recv(&size_0,1,MPI_INT,target_part,0,mpi_comm, &r_status);
      if (2*(my_rank+1) == comm_size-1)
        MPI_Recv(&size_1,1,MPI_INT,target_part-1,0,mpi_comm, &r_status);

    //Allocate for next iteration
      op_free(x); op_free(global_indices);
      x = (double *)xrealloc(x_keep, (keep_ctr+size_0+size_1)*3*sizeof(double)); //Implicitly assign x = x_keep
      global_indices = (int *)xrealloc(idx_gbl_keep, (keep_ctr+size_0+size_1)*sizeof(int)); //Implicitly assign global_indices = idx_gbl_keep
      current_part_size = keep_ctr+size_0+size_1;

      MPI_Wait(&s_request, &s_status);

      MPI_Isend(x_send, 3*send_ctr, MPI_DOUBLE, target_part, 1, mpi_comm, &s_request);
      MPI_Isend(idx_gbl_send, send_ctr, MPI_INT, target_part, 2, mpi_comm, &s_request2);

    //Pack kept - receive 0 - receive 1
      if (2*my_rank != (comm_size-1)) {
        MPI_Recv(&x[keep_ctr*3],size_0*3,MPI_DOUBLE,target_part,1,mpi_comm, &r_status);
        MPI_Recv(&global_indices[keep_ctr],size_0,MPI_INT,target_part,2,mpi_comm, &r_status);
      }
      if (2*(my_rank+1) == comm_size-1) {
        MPI_Recv(&x[(keep_ctr+size_0)*3],size_1*3,MPI_DOUBLE,target_part-1,1,mpi_comm, &r_status);
        MPI_Recv(&global_indices[keep_ctr+size_0],size_1,MPI_INT,target_part-1,2,mpi_comm, &r_status);
      }

    //Divide group in two
      int *processes_lower = (int *)xmalloc(comm_size/2*sizeof(int));
      int *processes_upper = (int *)xmalloc((comm_size-comm_size/2)*sizeof(int));
      for (int i = 0; i < comm_size/2; i++)
        processes_lower[i] = i;
      for (int i = 0; i < comm_size - comm_size/2; i++)
        processes_upper[i] = comm_size/2+i;

      MPI_Group lower_group, upper_group;
      MPI_Group_incl(current_group, comm_size/2, processes_lower, &lower_group);
      MPI_Group_incl(current_group, comm_size - comm_size/2, processes_upper, &upper_group);
      MPI_Comm lower_comm, upper_comm;
      MPI_Comm_create(mpi_comm, lower_group, &lower_comm);
      MPI_Comm_create(mpi_comm, upper_group, &upper_comm);

    //Join one of the groups
      if (my_rank <= comm_size/2-1) {
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
    //free stuff
      op_free(dist);
      op_free(processes_lower); op_free(processes_upper);
      MPI_Wait(&s_request, &s_status);
      MPI_Wait(&s_request2, &s_status2);
      op_free(idx_gbl_send); op_free(x_send);
    }
  }
  op_free(x);
  quickSort(global_indices, 0, current_part_size-1);
  MPI_Comm_dup(OP_MPI_WORLD, &OP_PART_WORLD);
  MPI_Comm_rank(OP_PART_WORLD, &my_rank);
  MPI_Comm_size(OP_PART_WORLD, &comm_size);
  //start binning (global indices -> processes)
  int *sizes = (int*)xcalloc(comm_size,sizeof(int));
  int target = 0;
  for (int i = 0; i < current_part_size; i++) {
    while (!(orig_part_range[x_dat->set->index][2*target]   <= global_indices[i] &&
             orig_part_range[x_dat->set->index][2*target+1] >= global_indices[i])) target++;
    sizes[target]++;
  }
  int *sizes_recv = (int*)xcalloc(comm_size,sizeof(int));
  MPI_Alltoall(sizes, 1, MPI_INT, sizes_recv, 1, MPI_INT, OP_PART_WORLD);

  //Sanity check
  int total_size = 0;
  for (int i = 0; i < comm_size; i++) total_size += sizes_recv[i];
  if (total_size != block_size) {
    printf("Error at rank %d: original(%d) vs. collected(%d) size mismatch! Aborting...\n", my_rank, block_size, total_size);
    MPI_Abort(OP_PART_WORLD,2);
  }

  //How many partitions we are sending to/receiving from
  int send_count = 0;
  int recv_count = 0;
  for (int i = 0; i < comm_size; i++) {
    if (sizes[i]) send_count++;
    if (sizes_recv[i]) recv_count++;
  }

  //Send
  MPI_Request *send_requests = (MPI_Request *)xmalloc(send_count * sizeof(MPI_Request));
  MPI_Request *recv_requests = (MPI_Request *)xmalloc(recv_count * sizeof(MPI_Request));
  MPI_Status *send_statuses = (MPI_Status *)xmalloc(send_count * sizeof(MPI_Status));
  MPI_Status *recv_statuses = (MPI_Status *)xmalloc(recv_count * sizeof(MPI_Status));
  int *global_indices_recv = (int *)xmalloc(total_size*sizeof(int));
  send_count = 0;
  recv_count = 0;
  int send_offset = 0;
  int recv_offset = 0;
  for (int i = 0; i < comm_size; i++) {
    if (sizes[i]) {
      MPI_Isend(&global_indices[send_offset], sizes[i], MPI_INT, i, 0, OP_PART_WORLD, &send_requests[send_count]);
      send_offset += sizes[i];
      send_count++;
    }
    if (sizes_recv[i]) {
      MPI_Irecv(&global_indices_recv[recv_offset], sizes_recv[i], MPI_INT, i, 0, OP_PART_WORLD, &recv_requests[recv_count]);
      recv_offset += sizes_recv[i];
      recv_count++;
    }
  }
  MPI_Waitall(recv_count, recv_requests, recv_statuses);
  int *partition = (int *)xmalloc(sizeof(int)*x_dat->set->size);
  recv_count = 0;
  recv_offset = 0;
  for (int i = 0; i < comm_size; i++) {
    if (sizes_recv[i]) {
      for (int j = recv_offset; j < recv_offset+sizes_recv[i]; j++)
        partition[global_indices_recv[j]-block_lower] = i;
      recv_offset += sizes_recv[i];
      recv_count++;
    }
  }
  MPI_Waitall(send_count, send_requests, send_statuses);
  //Debugging: print partvecs to file
  /*FILE *file;
  char fname[64];
  sprintf(fname,"partvec_op2_%04d",my_rank);
  file = fopen(fname,"w");
  for (int i = 0; i < x_dat->set->size; i++)
    fprintf(file,"%06d\n",partition[i]+1);
  fclose(file);*/

  //initialise primary set as partitioned
  OP_part_list[x_dat->set->index]->elem_part= (int *) partition;
  OP_part_list[x_dat->set->index]->is_partitioned = 1;

  //free part range
  for(int i = 0; i<OP_set_index; i++)op_free(part_range[i]);op_free(part_range);

  /*-STEP 2 - Partition all other sets,migrate data and renumber mapping tables-*/

  //partition all other sets
  partition_all(x_dat->set, my_rank, comm_size);

  //migrate data, sort elements
  migrate_all(my_rank, comm_size);

  //renumber mapping tables
  renumber_maps(my_rank, comm_size);

  op_timers(&cpu_t2, &wall_t2);  //timer stop for partitioning
  //printf time for partitioning
  time = wall_t2-wall_t1;
  MPI_Reduce(&time,&max_time,1,MPI_DOUBLE, MPI_MAX,MPI_ROOT, OP_PART_WORLD);
  MPI_Comm_free(&OP_PART_WORLD);
  if(my_rank==MPI_ROOT)printf("Max total inertial partitioning time = %lf\n",max_time);
}


/*******************************************************************************
* Toplevel partitioning selection function - also triggers halo creation
*******************************************************************************/
void partition(const char* lib_name, const char* lib_routine,
  op_set prime_set, op_map prime_map, op_dat data )
{
#if !defined(HAVE_PTSCOTCH) && !defined(HAVE_PARMETIS)
  /* Suppress warning */
  (void)lib_routine;
  (void)prime_map;
#endif

  int partial_halo_flag = 1; //flag to indicate that partial halos should be created
                             // default is 1, but if a proper partitioning is not done
                             // i.e. with ParMetis or PTScotch then we will have orphen
                             // set elements which will result in a runtime error
                             //when using partial halos

  /*initial error checks for NULL variables*/
  if(lib_name == NULL) lib_name = "NULL";
  if(lib_routine == NULL) lib_routine = "NULL";

  if(strcmp(lib_name,"PTSCOTCH")==0) {
    #ifdef HAVE_PTSCOTCH
    op_printf("Selected Partitioning Library : %s\n",lib_name);
    if(strcmp(lib_routine,"KWAY")==0) {
      op_printf("Selected Partitioning Routine : %s\n",lib_routine);
      if(prime_map != NULL)
        op_partition_ptscotch(prime_map); //use ptscotch kaway partitioning
      else {
        op_printf("Partitioning prime_map : NULL UNSUPPORTED\n");
        op_printf("Reverting to trivial block partitioning\n");
        partial_halo_flag = 0;
      }
    }
    else {
      op_printf("Partitioning Routine : %s UNSUPPORTED\n", lib_routine);
      op_printf("Reverting to trivial block partitioning\n");
      partial_halo_flag = 0;
    }
    #else
    op_printf("OP2 Library Not built with Partitioning Library : %s\n",lib_name);
    op_printf("Ignoring input routine : %s\n",lib_routine);
    if(prime_set != NULL)op_printf("Ignoring input mapping : %s\n",prime_set->name);
    if(prime_map != NULL)op_printf("Ignoring input mapping : %s\n",prime_map->name);
    if(data != NULL)op_printf("Ignoring input data : %s\n",data->name);
    op_printf("Reverting to trivial block partitioning\n");
    partial_halo_flag = 0;
    #endif
  }
  else
  if (strcmp(lib_name,"PARMETIS")==0) {
    #ifdef HAVE_PARMETIS
    op_printf("Selected Partitioning Library : %s\n",lib_name);
    if(strcmp(lib_routine,"KWAY")==0) {
      op_printf("Selected Partitioning Routine : %s\n",lib_routine);
      if(prime_map != NULL)
        op_partition_kway(prime_map); //use parmetis kaway partitioning
      else {
        op_printf("Partitioning prime_map : NULL - UNSUPPORTED Partitioner Specification\n");
        op_printf("Reverting to trivial block partitioning\n");
        partial_halo_flag = 0;
      }
    }
    else if(strcmp(lib_routine,"GEOMKWAY")==0) {
      op_printf("Selected Partitioning Routine : %s\n",lib_routine);
      if(prime_map != NULL)
        op_partition_geomkway(data, prime_map); //use parmetis kawaygeom partitioning
      else {
        op_printf("Partitioning prime_map or coordinates : NULL - UNSUPPORTED Partitioner Specification\n");
        op_printf("Reverting to trivial block partitioning\n");
        partial_halo_flag = 0;
      }
    }
    else if(strcmp(lib_routine,"GEOM")==0) {
      op_printf("Selected Partitioning Routine : %s\n",lib_routine);
      if(data != NULL)
        op_partition_geom(data); //use parmetis geometric partitioning
      else {
        op_printf("Partitioning coordinates: NULL - UNSUPPORTED Partitioner Specification\n");
        op_printf("Reverting to trivial block partitioning\n");
        partial_halo_flag = 0;
      }
    }
    else {
      op_printf("Partitioning Routine : %s UNSUPPORTED\n",lib_routine);
      op_printf("Reverting to trivial block partitioning\n");
      partial_halo_flag = 0;
    }
    #else
    /*  Suppress warning */
    (void)data;
    op_printf("OP2 Library Not built with Partitioning Library : %s\n",lib_name);
    op_printf("Ignoring input routine : %s\n",lib_routine);
    if(prime_set != NULL)op_printf("Ignoring input set : %s\n",prime_set->name);
    if(prime_map != NULL)op_printf("Ignoring input mapping : %s\n",prime_map->name);
    if(data != NULL)op_printf("Ignoring input coordinates : %s\n",data->name);
    op_printf("Reverting to trivial block partitioning\n");
    partial_halo_flag = 0;
    #endif
  }
  else if (strcmp(lib_name,"RANDOM")==0) {
    op_printf("Selected Partitioning Routine : %s\n",lib_name);
    if(prime_set != NULL )
      op_partition_random(prime_set); //use a random partitioning - used for debugging
    else {
      op_printf("Partitioning prime_set : NULL - UNSUPPORTED Partitioner Specification\n");
      op_printf("Reverting to trivial block partitioning\n");
      partial_halo_flag = 0;
    }
  }
  else if (strcmp(lib_name,"EXTERNAL")==0) {
    op_printf("Selected Partitioning Routine : %s\n",lib_name);
    if(prime_set != NULL) {
      if(data->data != NULL) {
        if (data->dim == 1) {
          op_partition_external(prime_set, data); //use an external partitioning read in from hdf5 file
          partial_halo_flag = 0;
        }
        else {
          op_printf("External Partition vector should be an integer array with dimension 1\n");
          op_printf("Reverting to trivial block partitioning\n");
          partial_halo_flag = 0;
        }
      }
      else {
        op_printf("External Partition vector : NULL - UNSUPPORTED Partitioner Specification\n");
        partial_halo_flag = 0;
      }
    }
    else {
      op_printf("Partitioning prime_set : NULL - UNSUPPORTED Partitioner Specification\n");
      op_printf("Reverting to trivial block partitioning\n");
      partial_halo_flag = 0;
    }
  }
  else if (strcmp(lib_name,"INERTIAL")==0) {
    op_printf("Selected Partitioning Routine : %s\n",lib_name);
      if(data->data != NULL) {
        if (data->dim == 3)
          op_partition_inertial(data); //use Oplus style Inertial partitioning
        else {
          op_printf("Onlt supports 3D Inertial Bisection Partitioning - Need 3D coordinates - dim should be 3\n");
          op_printf("Reverting to trivial block partitioning\n");
          partial_halo_flag = 0;
        }
    }
    else {
      op_printf("Partitioning based on dataset : NULL - UNSUPPORTED Partitioner Specification\n");
      op_printf("Reverting to trivial block partitioning\n");
      partial_halo_flag = 0;
    }
  }
  else {
    op_printf("Partitioning Library : %s UNSUPPORTED\n",lib_name);
    op_printf("Ignoring input routine : %s\n",lib_routine);
    if(prime_set != NULL)op_printf("Ignoring input set : %s\n",prime_set->name);
    if(prime_map != NULL)op_printf("Ignoring input mapping : %s\n",prime_map->name);
    if(data != NULL)op_printf("Ignoring input coordinates : %s\n",data->name);
    op_printf("Reverting to trivial block partitioning\n");
    partial_halo_flag = 0;
  }

  //trigger halo creation routines
  op_halo_create();

  if (partial_halo_flag == 1) //only do partial halo
    op_halo_permap_create();  //creation if a valid partitioning is done
  else {
    set_import_buffer_size = (int *)xmalloc(OP_set_index*sizeof(int));
    for (int i = 0; i < OP_set_index; i++)
      set_import_buffer_size[i] = 0;
    OP_map_partial_exchange = (int *)xmalloc(OP_map_index*sizeof(int));
    for (int i = 0; i < OP_map_index; i++)
      OP_map_partial_exchange[i] = 0;
  }

#ifdef DEBUG //sanity check to identify if the partitioning results in ophan elements
  int ctr = 0;
  for (int i = 0; i < prime_map->from->size; i++) {
    if (prime_map->map[2*i]>=prime_map->to->size &&
      prime_map->map[2*i+1]>=prime_map->to->size) ctr++;
  }
  printf("Orphan edges: %d\n", ctr);
#endif

}
