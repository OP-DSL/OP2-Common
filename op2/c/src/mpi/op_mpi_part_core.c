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
 * written by: Gihan R. Mudalige, 07-06-2011
 */
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_rt_support.h>

//parmetis header
#include <parmetis.h>

#include <op_mpi_core.h>
#include <op_mpi_part_core.h>

#include <op_mpi_util.h>

//
//MPI Communicator for partitioning
//
MPI_Comm OP_PART_WORLD;


int frequencyof(int value, int* array, int size)
{
   int frequency = 0; 
   for(int i = 0; i<size; i++)
   {
    	if(array[i] == value) frequency++;   
   }
   return frequency;
}

int find_mode(int* array, int size)
{
   int count = 0, mode = array[0], current;
   for(int i=0; i<size; i++)
   {
        current = frequencyof(array[i], array, size);
    	if(count< current)
    	{   	    	
    	    count = current;
    	    mode = array[i];
    	}
   }
   return mode;
}

int compare_all_sets(op_set target_set, op_set other_sets[], int size)
{
    for(int i = 0; i < size; i++)
    {
    	if(compare_sets(target_set, other_sets[i])==1)return i;
    }    
    return -1;
}

/**special routine to create export list during partitioning map->to set 
from map_>from set in partition_to_set()**/
int* create_exp_list_2(op_set set, int* temp_list, halo_list h_list, 
    int* part_list, int size, int comm_size, int my_rank)
{
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
	free(temp_to);
	free(temp_part);
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

/**special routine to create import list during partitioning map->to set 
from map_>from set in partition_to_set()**/
void create_imp_list_2(op_set set, int* temp_list, halo_list h_list, 
    int total_size, int* ranks, int* sizes, int ranks_size, int comm_size, 
    int my_rank)
{
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



/** ----use the partitioned map->to set to partition the map->from set ------**/
int partition_from_set(op_map map, int my_rank, int comm_size, int** part_range)
{
    
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
    free(temp_list);
    
    //printf("on rank %d: good up to here\n",my_rank);
    
    //now, discover neighbors and create export list of "to" elements
    int ranks_size = 0;

    int* neighbors = (int *)xmalloc(comm_size*sizeof(int));
    int* sizes = (int *)xmalloc(comm_size*sizeof(int));
    
    halo_list pe_list = (halo_list) xmalloc(sizeof(halo_list_core));
    find_neighbors_set(pi_list, neighbors, sizes, &ranks_size, my_rank, 
    	comm_size, OP_PART_WORLD);
    
    MPI_Request request_send[pi_list->ranks_size];
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
    	    MPI_STATUSES_IGNORE );
    	memcpy(&temp_list[count],(void *)&rbuf[0],sizes[i]*sizeof(int));
    	count = count + sizes[i];
    	free(rbuf);
    }
    MPI_Waitall(pi_list->ranks_size,request_send, MPI_STATUSES_IGNORE );
    create_import_list(map->to,temp_list, pe_list, count, 
    	neighbors, sizes, ranks_size, comm_size, my_rank);    
    
    
    //printf("on rank %d; good up to this\n",my_rank);
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
    	    OP_PART_WORLD, MPI_STATUSES_IGNORE);
    	
    }
    MPI_Waitall(pe_list->ranks_size,request_send_p, MPI_STATUSES_IGNORE );
    for(int i = 0; i<pe_list->ranks_size; i++) free(sbuf[i]);free(sbuf);
    
    
    
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
    	    	     	 found_parts[j] = imp_part[elem];
    	    	     else
    	    	     {
    	    	     	 printf("Element %d not found in partition import list\n",
    	    	     	     local_index);
    	    	     	 exit(1);
    	    	     }
    	    	 }
    	    	 else 
    	    	 {
    	    	     printf("Rank %d not found in partition import list\n", part);
    	    	     exit(1);
    	    	 }
    	    }    	 	   
    	}
	partition[i] = find_mode(found_parts, map->dim);     	
    }
               
    OP_part_list[map->from->index]->elem_part = partition;
    OP_part_list[map->from->index]->is_partitioned = 1;
    
    //cleanup
    free(imp_part);
    free(pi_list->list);free(pi_list->ranks);free(pi_list->sizes);
    free(pi_list->disps);free(pi_list);
    free(pe_list->list);free(pe_list->ranks);free(pe_list->sizes);
    free(pe_list->disps);free(pe_list);
    
    return 1;    
}


/** ---- use the partitioned map->from set to partition the map->to set -------**/
int partition_to_set(op_map map, int my_rank, int comm_size, int** part_range)
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
    free(temp_list);
    
  
    int ranks_size = 0;
    int* neighbors = (int *)xmalloc(comm_size*sizeof(int));
    int* sizes = (int *)xmalloc(comm_size*sizeof(int));
    
    //to_part_list tpi_list;
    halo_list pi_list = (halo_list) xmalloc(sizeof(halo_list_core));
    int* part_list_i = NULL; //corresponding "to" element's partition infomation 
                             //imported from an mpi rank
    
    find_neighbors_set(pe_list,neighbors,sizes,&ranks_size,my_rank, 
    	comm_size, OP_PART_WORLD);
    
    
    MPI_Request request_send_t[pe_list->ranks_size];
    MPI_Request request_send_p[pe_list->ranks_size];
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
    	    MPI_STATUSES_IGNORE );
    	MPI_Recv(rbuf_p, sizes[i], MPI_INT, neighbors[i], 2, OP_PART_WORLD,
    	    MPI_STATUSES_IGNORE );
    	memcpy(&temp_list_t[count],(void *)&rbuf_t[0],sizes[i]*sizeof(int));    	
    	memcpy(&part_list_i[count],(void *)&rbuf_p[0],sizes[i]*sizeof(int));
    	count = count + sizes[i];
    	free(rbuf_t);
    	free(rbuf_p);
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
    to_elems = (int *)xrealloc(to_elems, sizeof(int)*(count+pi_list->size));
    parts = (int *)xrealloc(parts, sizeof(int)*(count+pi_list->size));
    
    memcpy(&to_elems[count],(void *)&pi_list->list[0],pi_list->size*sizeof(int));
    memcpy(&parts[count],(void *)&part_list_i[0],pi_list->size*sizeof(int));
    
    int *partition = (int *)xmalloc(sizeof(int)*map->to->size);
    for(int i = 0; i < map->to->size; i++){partition[i] = -99;}
    
    count = count+pi_list->size;
    
    //sort both to_elems[] and correspondingly parts[] arrays
    quickSort_2(to_elems, parts, 0, count-1);
    
    int* found_parts;
    for(int i = 0; i<count;)
    {
    	int curr = to_elems[i];
    	int c = 0; int cap = map->dim;
    	found_parts = (int *)xmalloc(sizeof(int)*cap);
    
    	while(curr == to_elems[i])
    	{
    	  if(c>=cap) 
    	  {
    	      cap = cap*2;
    	      found_parts = (int *)xrealloc(found_parts, sizeof(int)*cap);   
    	  }
    	  found_parts[c++] =  parts[i];
    	  i++;
    	}    	
	partition[curr] = find_mode(found_parts, c);
	free(found_parts);
    }  

    free(to_elems);free(parts);
    
    //check if this "from" set is an "on to" set 
    //need to check this globally on all processors
    int ok = 1;
    for(int i = 0; i < map->to->size; i++) 
    {	
    	if(partition[i]<0)  
    	{
    	    printf("on rank %d: Map %s is not an an on-to mapping from set %s to set %s\n",
    	    	my_rank, map->name, map->from->name,map->to->name);
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
    for(int r = 0;r<comm_size;r++)
    {
    	if(global_ok_array[r]<0)
    	{
    	    printf("Rank %d reported problem partitioning\n",r);
    	    return -1;
    	}
    }
    free(global_ok_array);
    
    OP_part_list[map->to->index]->elem_part = partition;
    OP_part_list[map->to->index]->is_partitioned = 1;
    
    //cleanup 
    free(pi_list->list);free(pi_list->ranks);free(pi_list->sizes);
    free(pi_list->disps);
    free(pe_list->list);free(pe_list->ranks);free(pe_list->sizes);
    free(pe_list->disps);    
    free(part_list_i);free(part_list_e);
    
    return 1;
}



/**-------- Partition Secondary sets using primary set partition -------------*/
void partition_all(op_set primary_set, int my_rank, int comm_size)
{   
    // Compute global partition range information for each set
    int** part_range = (int **)xmalloc(OP_set_index*sizeof(int*));
    get_part_range(part_range,my_rank,comm_size, OP_PART_WORLD);

    int sets_partitioned = 1;
    int maps_used = 0;
    
    op_set all_partitioned_sets[OP_set_index]; 
    int all_used_maps[OP_map_index];
    for(int i = 0; i<OP_map_index; i++) { all_used_maps[i] = -1;}
    
    //begin with the partitioned primary set - e.g nodes
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
    	    	    cost[map->index] = 0;
    	    }
       	}
        
       	
       	while(1)
       	{
       	    int selected = min(cost, OP_map_index);
       	    
       	    for(int i = 0; i<OP_map_index;i++)
       	    	printf(" %d",cost[i]);
       	    printf(": selected %d",selected);
       	    printf("\n");
       	    if(selected >= 0)
       	    {
       	    	op_map map=OP_map_list[selected];
       	       
       	    	//partition using this map       	              	   
       	    	part to_set = OP_part_list[map->to->index];
       	    	part from_set = OP_part_list[map->from->index];
       	       
       	    	if(to_set->is_partitioned == 1) 
       	    	{
       	    	    printf("Attempting to partition %s using %s\n",map->from->name,map->to->name);
       	    	    if( partition_from_set(map, my_rank, comm_size, part_range) > 0)
       	    	    {
       	    	    	//if(my_rank==0)
       	    	    	printf("On rank %d: Using map %s to partitioned from set %s using set %s\n",
       	    	    		   my_rank, map->name,map->from->name,map->to->name);
       	    	    	all_partitioned_sets[sets_partitioned++] = map->from;
       	    	    	all_used_maps[maps_used++] = map->index;
       	    	    	break;       	       	       
       	    	    }
       	    	    else //partitioning unsuccessful with this map- find another map
       	    	    	cost[selected] = 99;
       	    	} 
       	    	else if(from_set->is_partitioned == 1) 
       	    	{
       	    	    printf("Attempting to partition %s using %s\n",map->to->name,map->from->name);
       	    	    if( partition_to_set(map, my_rank, comm_size, part_range) > 0)
       	    	    {
       	    	    	//if(my_rank==0)
       	    	    	printf("On rank %d: Using map %s to partitioned to set %s using set %s\n",
       	    	    		   my_rank, map->name,map->to->name,map->from->name);
       	    	    	all_partitioned_sets[sets_partitioned++] = map->to;
       	    	    	all_used_maps[maps_used++] = map->index;
       	    	    	break;
       	    	    }
       	    	    else //partitioning unsuccessful with this map - find another map
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
       
    if(my_rank==0)
    {
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
    	    	}
    	    }
    	    printf("Partitioning aborted !\n");
    	    exit(1);
    	}
    }
    
    for(int i = 0; i<OP_set_index; i++)free(part_range[i]);free(part_range);
}


/**------- Renumber mapping table entries with new partition's indexes -------*/
void renumber_maps(int my_rank, int comm_size)
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
      	free(req_list);
      	  
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
      	free(g_index);
      	
      	//realloc exp_index, exp_g_index
      	exp_index = (int *)xrealloc(exp_index,sizeof(int)*exp_count);
      	exp_g_index = (int *)xrealloc(exp_g_index,sizeof(int)*exp_count);
      	
      	//now export to every MPI rank, these partition info with an all-to-all
      	MPI_Allgather(&exp_count, 1, MPI_INT, recv_count, 1, MPI_INT, OP_PART_WORLD);
      	disp = 0; free(displs);
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
      	
      	free(exp_index);
      	free(exp_g_index);
      	
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
      	
      	free(g_index);
      	free(displs);
      	free(all_imp_index);
    }         
    for(int i = 0; i<OP_set_index; i++)free(part_range[i]);free(part_range);
}


/**--------- Reverse the renumbering of mapping tables------------------------*/
void reverse_renumber_maps(int my_rank, int comm_size)
{
    int** part_range = (int **)xmalloc(OP_set_index*sizeof(int*));
    get_part_range(part_range,my_rank,comm_size, OP_PART_WORLD);
    
    //renumber mapping tables replacing the to_set elements of each mapping table
    // with the original index of those set elements from g_index (will need all to alls)
    for(int m=0; m<OP_map_index; m++) { //for each map
    	op_map map=OP_map_list[m];
    	
    	int cap = 1000; int count = 0;
      	int* req_list = (int *)xmalloc(cap*sizeof(int));
      	
    	for(int i = 0; i< map->from->size; i++)
      	{
      	    int part, local_index;
      	    for(int j=0; j<map->dim; j++) { //for each element pointed at by this entry
      	    	part = get_partition(map->map[i*map->dim+j],
      	    	    part_range[map->to->index],&local_index,comm_size);    	    
      	    	
      	    	if(count>=cap)
      	    	{
      	    	    cap = cap*2;
      	    	    req_list = (int *)xrealloc(req_list, cap*sizeof(int));
      	    	}
      	    
      	    	if(part != my_rank)
      	    	{
      	    	    //add current global index of this to_set 
      	    	    //element to the request list 
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
      	//be requesting original global index information about
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
      	free(req_list);
      	  
      	if(g_count > 0)
      	{
      	    quickSort(g_index, 0, g_count-1);
      	    g_count = removeDups(g_index, g_count);
      	    g_index = (int *)xrealloc(g_index, g_count*sizeof(int));
      	}
      	
      	//go through the recieved global g_index array and see if any local element's
      	//original global index details are requested by some foreign process
      	int *curr_index = (int *)xmalloc(sizeof(int)*g_count);
      	int *orig_index = (int *)xmalloc(sizeof(int)*g_count);
      	
      	int exp_count = 0;
      	for(int i = 0; i<g_count; i++)
      	{
      	    int local_index, part; 
      	    
      	    part = get_partition(g_index[i], part_range[map->to->index],
      	    	&local_index,comm_size);

      	    if(part == my_rank)
      	    {
      	    	orig_index[exp_count] = 
      	    	OP_part_list[map->to->index]->g_index[local_index];
      	    	curr_index[exp_count++] = g_index[i];
      	    }
      	    
      	}
      	free(g_index);
      	
      	//realloc cur_index, org_index
      	curr_index = (int *)xrealloc(curr_index,sizeof(int)*exp_count);
      	orig_index = (int *)xrealloc(orig_index,sizeof(int)*exp_count);
      	
      	//now export to every MPI rank, these original global index info with an all-to-all
      	MPI_Allgather(&exp_count, 1, MPI_INT, recv_count, 1, MPI_INT, OP_PART_WORLD);
      	disp = 0; free(displs);
      	displs = (int *)xmalloc(comm_size*sizeof(int));
      	
      	for(int i = 0; i<comm_size; i++)
      	{
      	    displs[i] =   disp;
      	    disp = disp + recv_count[i];
      	}
      	
      	//allocate memory to hold the incomming original global index details and allgatherv  
      	g_count = 0;
      	for(int i = 0; i< comm_size; i++)g_count += recv_count[i];
      	int *all_orig_index = (int *)xmalloc(sizeof(int)*g_count);
      	int *all_curr_index = (int *)xmalloc(sizeof(int)*g_count);
      	  
      	//printf("on rank %d map %s need set %s: After g_count = %d\n", 
      	//    my_rank, map.name,map.to.name,g_count);
      	  
      	MPI_Allgatherv(curr_index,exp_count,MPI_INT, all_curr_index,recv_count,displs, 
      	    MPI_INT, OP_PART_WORLD);
      	
      	MPI_Allgatherv(orig_index,exp_count,MPI_INT, all_orig_index,recv_count,
      	    displs, MPI_INT, OP_PART_WORLD);
      	
      	free(curr_index);
      	free(orig_index);
      	
      	//sort all_orig_index according to all_curr_index array
      	if(g_count > 0)quickSort_2(all_curr_index, all_orig_index, 0, g_count-1); 
      	
      	//now we hopefully have all the informattion required to reverse the 
      	//renumbering of this map. so now, again go through each entry of this 
      	//mapping table and reverse-renumber
      	for(int i = 0; i< map->from->size; i++)
      	{
      	    int part, local_index;
      	    for(int j=0; j<map->dim; j++) { //for each element pointed at by this entry
      	    	part = get_partition(map->map[i*map->dim+j],
      	    	    part_range[map->to->index],&local_index,comm_size);    	    
      	    
      	    	if(part != my_rank)
      	    	{
      	    	    //find from all_curr_index and all_orig_index
      	    	    local_index = binary_search(all_curr_index,
      	    	    map->map[i*map->dim+j], 0, g_count-1);
      	    	    if(local_index < 0) 
      	    	    	printf("Problem in reverse-renumbering\n");
      	    	    else
      	    	    	OP_map_list[map->index]->map[i*map->dim+j] =
	      	    	    all_orig_index[local_index];      	    	    
      	    	}
      	    	else
      	    	{
      	    	    OP_map_list[map->index]->map[i*map->dim+j] = 
      	    	    OP_part_list[map->to->index]->g_index[local_index];
      	    	}
      	    }
      	}
      	
      	
      	free(all_curr_index);free(all_orig_index);
      	free(displs);      	
      	
    }
    for(int i = 0; i<OP_set_index; i++)free(part_range[i]);free(part_range);
}


/**------Data migration to new partitions (or reverse parition)---------------*/
void migrate_all(int my_rank, int comm_size)
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
    	create_export_list(set, temp_list, pe_list[set->index], count, comm_size, my_rank);
    	free(temp_list);      
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
    	MPI_Request request_send[exp->ranks_size];
      
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
    	    	MPI_STATUSES_IGNORE );
    	    memcpy(&temp_list[count],(void *)&rbuf[0],sizes[i]*sizeof(int));
    	    count = count + sizes[i];
    	    free(rbuf);
    	}
      
    	MPI_Waitall(exp->ranks_size,request_send, MPI_STATUSES_IGNORE );
    	pi_list[set->index] = (halo_list) xmalloc(sizeof(halo_list_core));
    	create_import_list(set, temp_list, pi_list[set->index], count,
    	    neighbors, sizes, ranks_size, comm_size, my_rank);
    }
    

/*--STEP 2 - Perform Partitioning Data migration -----------------------------*/

    //data migration first ......  
    for(int s=0; s<OP_set_index; s++) { //for each set
    	op_set set=OP_set_list[s];
    	
    	halo_list imp = pi_list[set->index];
    	halo_list exp = pe_list[set->index];
    	
    	MPI_Request request_send[exp->ranks_size];
    	    	
    	//migrate data defined on this set
    	for(int d=0; d<OP_dat_index; d++) { //for data array
    	    op_dat dat=OP_dat_list[d];
    	    
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
    	    	    MPI_Isend(sbuf[i], dat->size*exp->sizes[i], MPI_CHAR, exp->ranks[i],
    	    	    	d, OP_PART_WORLD, &request_send[i]);
    	    	}
      	      
    	    	char *rbuf = (char *)xmalloc(dat->size*imp->size);
    	    	for(int i=0; i<imp->ranks_size; i++) {
    	    	    //printf("imported on to %d data %10s, number of elements of size %d | recieving:\n ",
    	    	    //	  my_rank, dat->name, imp->size);
    	    	    MPI_Recv(&rbuf[imp->disps[i]*dat->size],dat->size*imp->sizes[i], 
    	    	    	MPI_CHAR, imp->ranks[i], d, OP_PART_WORLD, MPI_STATUSES_IGNORE);
    	    	}
      	      
    	    	MPI_Waitall(exp->ranks_size,request_send, MPI_STATUSES_IGNORE );
    	    	for(int i=0; i < exp->ranks_size; i++) free(sbuf[i]); free(sbuf);
      	      
    	    	//delete the data entirs that has been sent and create a 
    	    	//modified data array
    	    	char* new_dat = (char *)xmalloc(dat->size*(set->size+imp->size));
    	    	
    	    	int count = 0;
    	    	for(int i = 0; i < dat->set->size;i++)//iterate over old set size
    	    	{
    	    	    if(OP_part_list[set->index]->elem_part[i] == my_rank)
    	    	    {
    	    	    	memcpy(&new_dat[count*dat->size],
    	    	    	    (void *)&OP_dat_list[dat->index]->
    	    	    	    data[dat->size*i],dat->size);
    	    	    	count++;
    	    	    }      	      	            	      		  
    	    	}

    	    	memcpy(&new_dat[count*dat->size],(void *)rbuf,
    	    	    dat->size*imp->size);
    	    	count = count+imp->size;
    	    	new_dat = (char *)xrealloc(new_dat,dat->size*count);
    	    	free(rbuf);
      	           	      
    	    	free(OP_dat_list[dat->index]->data);   	  
    	    	OP_dat_list[dat->index]->data = new_dat;
    	    }  
    	}
    }
    
    //mapping tables second ......
    for(int s=0; s<OP_set_index; s++) { //for each set
    	op_set set=OP_set_list[s];
      
    	halo_list imp = pi_list[set->index];
    	halo_list exp = pe_list[set->index];
      
    	MPI_Request request_send[exp->ranks_size];
           
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
    	    	    //	  my_rank, map->name, imp->size);
    	    	    MPI_Recv(&rbuf[imp->disps[i]*map->dim], 
    	    	    	map->dim*imp->sizes[i], 
    	    	    	MPI_INT, imp->ranks[i], m, 
    	    	    	OP_PART_WORLD, MPI_STATUSES_IGNORE);
    	    	}
      	      
    	    	MPI_Waitall(exp->ranks_size,request_send, MPI_STATUSES_IGNORE );
    	    	for(int i=0; i < exp->ranks_size; i++) free(sbuf[i]); free(sbuf);
      	      
    	    	//delete the mapping table entirs that has been sent and create a 
    	    	//modified mapping table
    	    	int* new_map = (int *)xmalloc(sizeof(int)*(set->size+imp->size)*map->dim);

    	    	int count = 0;
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
    	    	
    	    	free(rbuf);
    	    	free(OP_map_list[map->index]->map);
    	    	OP_map_list[map->index]->map = new_map;
    	    }   
    	}
    }

/*--STEP 3 - Update Partitioning Information and Sort Set Elements------------*/

    //need to exchange the original g_index 
    for(int s=0; s<OP_set_index; s++) { //for each set
    	op_set set=OP_set_list[s];
    	
    	halo_list imp = pi_list[set->index];
    	halo_list exp = pe_list[set->index];
    	
    	MPI_Request request_send[exp->ranks_size];
    	
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
    	    	OP_PART_WORLD, MPI_STATUSES_IGNORE);
    	}
    	MPI_Waitall(exp->ranks_size,request_send, MPI_STATUSES_IGNORE );
    	for(int i=0; i < exp->ranks_size; i++) free(sbuf[i]); free(sbuf);
    	
    	//delete the g_index entirs that has been sent and create a
    	//modified g_index
    	int* new_g_index = (int *)xmalloc(sizeof(int)*(set->size+imp->size));

    	int count = 0;
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
    	
    	free(rbuf);
    	free(OP_part_list[set->index]->g_index);
    	free(OP_part_list[set->index]->elem_part);
    	    	
    	OP_part_list[set->index]->elem_part = new_part;
    	OP_part_list[set->index]->g_index = new_g_index; 
    	    	
    	OP_set_list[set->index]->size = count;   
    	OP_part_list[set->index]->set= OP_set_list[set->index];
    }
    
    //re-set values in mapping tables
    for(int m=0; m<OP_map_index; m++) { //for each maping table
   	   	  op_map map=OP_map_list[m];
      	  
   	   	  OP_map_list[map->index]->from = OP_set_list[map->from->index];
   	   	  OP_map_list[map->index]->to = OP_set_list[map->to->index];
    }
  
    //re-set values in data arrays
    for(int d=0; d<OP_dat_index; d++) { //for data array
  	    	  op_dat dat=OP_dat_list[d];      	  
  	    	  OP_dat_list[dat->index]->set = OP_set_list[dat->set->index];      	  
    }

    //finally .... need to sort for each set, data on the set and mapping tables 
    //from this set accordiing to the OP_part_list[set.index]->g_index array values.  
    for(int s=0; s<OP_set_index; s++) { //for each set
    	op_set set=OP_set_list[s];
                  
    	//first ... data on this set
    	for(int d=0; d<OP_dat_index; d++) { //for data array
    	    op_dat dat=OP_dat_list[d];
      	  
    	    if(compare_sets(dat->set,set) == 1)
    	    {
    	    	if(set->size > 0)
    	    	{
    	    	    int* temp = (int *)xmalloc(sizeof(int)*set->size);
    	    	    memcpy(temp, (void *)OP_part_list[set->index]->g_index, 
    	    	    	sizeof(int)*set->size);
    	    	    quickSort_dat(temp,OP_dat_list[dat->index]->data, 0, 
    	    	    	set->size-1, dat->size);
    	    	    free(temp);
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
    	    	    free(temp);
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
      free(pe_list[set->index]->ranks);free(pe_list[set->index]->disps);
      free(pe_list[set->index]->sizes);free(pe_list[set->index]->list);
      
      free(pi_list[set->index]->ranks);free(pi_list[set->index]->disps);
      free(pi_list[set->index]->sizes);free(pi_list[set->index]->list);
      free(pe_list[set->index]);free(pi_list[set->index]);
    }        
}


/**-------------------Partition A Primary Set randomly ----------------------**/
void op_partition_random(op_set primary_set)
{
    //declare timers
    double cpu_t1, cpu_t2, wall_t1, wall_t2;
    double time;
    double max_time;
    
    op_timers(&cpu_t1, &wall_t1); //timer start for partitioning
  
    //create new communicator for partitioning
    int my_rank, comm_size;
    MPI_Comm_dup(MPI_COMM_WORLD, &OP_PART_WORLD);
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
    for(int i = 0; i<OP_set_index; i++)free(part_range[i]);free(part_range);

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
    if(my_rank==0)printf("Max total random partitioning time = %lf\n",max_time);     
}

/**------------Partition A Primary Set Using XYZ Geometry Data---------------**/
void op_partition_geom(op_dat coords, int g_nnode) //wrapper to use ParMETIS_V3_PartGeom()
{
    //declare timers
    double cpu_t1, cpu_t2, wall_t1, wall_t2;
    double time;
    double max_time;
    
    op_timers(&cpu_t1, &wall_t1); //timer start for partitioning
  
    //create new communicator for partitioning
    int my_rank, comm_size;
    MPI_Comm_dup(MPI_COMM_WORLD, &OP_PART_WORLD);
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
  
/*--STEP 1 - Partition Nodes (primary set) using Coordinates (1D,2D or 3D)----*/

    //Setup data structures for ParMetis PartGeom
    idxtype *vtxdist = (idxtype *)xmalloc(sizeof(idxtype)*(comm_size+1));
    idxtype *partition = (idxtype *)xmalloc(sizeof(idxtype)*coords->set->size);
   
    int ndims = coords->dim;
    float* xyz;
    
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
    	    	memcpy(&temp, (void *)&(OP_dat_list[coords->index]->
    	    	    data[(i*coords->dim+e)*mult]), mult);
    	    	xyz[i*coords->dim + e] = (float)temp;
    	    }
    	}
    }
    else
    {
    	printf("Dimensions of Coordinate array not one of 3D,2D or 1D\n");
    	printf("Not supported by ParMetis - Indicate correct coordinates array\n");
    	exit(1);
    }
  
    for(int i=0; i<comm_size; i++)
    {
    	vtxdist[i] = part_range[0][2*i];//nodes have index 0
    }
    vtxdist[comm_size] = g_nnode;
  
    //use xyz coordinates to feed into ParMETIS_V3_PartGeom
    ParMETIS_V3_PartGeom(vtxdist, &ndims, xyz, partition, &OP_PART_WORLD);
    free(xyz);free(vtxdist);
    
    //free part range
    for(int i = 0; i<OP_set_index; i++)free(part_range[i]);free(part_range);

    //initialise primary set as partitioned
    OP_part_list[coords->set->index]->elem_part= partition;
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
    if(my_rank==0)printf("Max total geometric partitioning time = %lf\n",max_time);    
}


/**------------ Partition Using ParMETIS PartKway() routine --------------**/
void op_partition_kway(op_map primary_map){}


/**------------ Partition Using ParMETIS PartGeomKway() routine --------------**/
void op_partition_geomkway(op_dat coords, int g_nnode, op_map primary_map){}




//revert back to the original partitioning 
void op_partition_reverse()
{
    //declare timers
    double cpu_t1, cpu_t2, wall_t1, wall_t2;
    double time;
    double max_time;
    op_timers(&cpu_t1, &wall_t1); //timer start for partition reversing
    
    //create new communicator for reverse-partitioning
    int my_rank, comm_size;
    MPI_Comm_dup(MPI_COMM_WORLD, &OP_PART_WORLD);
    MPI_Comm_rank(OP_PART_WORLD, &my_rank);
    MPI_Comm_size(OP_PART_WORLD, &comm_size);
    
    //need original g_index with current index - already in OP_part_list
    //need original part_range - saved during partition creation
          
    //use g_index and original part range to fill in 
    //OP_part_list[set->index]->elem_part with original partitioning information
    for(int s=0; s<OP_set_index; s++) { //for each set
    	op_set set=OP_set_list[s];
    	
    	for(int i = 0; i < set->size; i++)
    	{
    	    int local_index;
    	    OP_part_list[set->index]->elem_part[i] = 
    	    get_partition(OP_part_list[set->index]->g_index[i], 
    	    	orig_part_range[set->index], &local_index, comm_size);
    	}
    }
    
    //reverse renumbering of mapping tables
    reverse_renumber_maps(my_rank, comm_size);
    
    //reverse back migration
    migrate_all(my_rank, comm_size);
    
    //destroy OP_part_list[]
    for(int s=0; s<OP_set_index; s++) { //for each set
    	op_set set=OP_set_list[s];
    	free(OP_part_list[set->index]->g_index);
    	free(OP_part_list[set->index]->elem_part);
    	free(OP_part_list[set->index]);
    }
    free(OP_part_list);
    for(int i = 0; i<OP_set_index; i++)free(orig_part_range[i]);free(orig_part_range);
    
    op_timers(&cpu_t2, &wall_t2);  //timer stop for partition reversing
    //printf time for partition reversing
    time = wall_t2-wall_t1;
    MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_ROOT, OP_PART_WORLD);
    MPI_Comm_free(&OP_PART_WORLD);  
    if(my_rank==0)printf("Max total partition reverse time = %lf\n",max_time);    
}




