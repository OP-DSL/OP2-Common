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

#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_rt_support.h>

/*
 * written by: Gihan R. Mudalige, 01-03-2011
 */

//mpi header
#include <mpi.h>

#include <op_mpi_core.h>

//utility functions
//from www.gnu.org/
void* xmalloc (size_t size)
{
  //if(size == 0) return (void *)NULL;

  register void *value = malloc (size);
  if (value == 0) printf("Virtual memory exhausted at malloc\n");
  return value;
}

void* xrealloc (void *ptr, size_t size)
{
  if(size == 0) return (void *)NULL;

  register void *value = realloc (ptr, size);
  if (value == 0) printf ("Virtual memory exhausted at realloc\n");
  return value;
}


int compare_sets(op_set set1, op_set set2)
{
  if(set1->size == set2->size &
      strcmp(set1->name,set2->name)==0 &
      set1->index == set2->index)
    return 1;
  else return 0;

}


int min(int array[], int size)
{
  int min = 99;
  int index = -1;
  for(int i=0; i<size; i++)
  {
    if(array[i]<min)
    {
      index = i;
      min = array[i];
    }
  }
  return index;
}

int binary_search(int a[], int value, int low, int high)
{
  if (high < low)
    return -1; // not found

  int mid = low + (high - low) / 2;
  if (a[mid] > value)
    return binary_search(a, value, low, mid-1);
  else if (a[mid] < value)
    return binary_search(a, value, mid+1, high);
  else
    return mid; // found
}

int linear_search(int a[], int value, int low, int high)
{
  for(int i = low; i<=high; i++)
  {
    if (a[i] == value) return i;
  }
  return -1;
}

void quickSort(int arr[], int left, int right)
{
  int i = left, j = right;
  int tmp;
  int pivot = arr[(left + right) / 2];

  // partition
  while (i <= j) {
    while (arr[i] < pivot)i++;
    while (arr[j] > pivot)j--;
    if (i <= j) {
      tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
      i++; j--;
    }
  };
  // recursion
  if (left < j)
    quickSort(arr, left, j);
  if (i < right)
    quickSort(arr, i, right);
}

//sort arr1 and organise arr2 elements according to the sorted arr1 order
void quickSort_2(int arr1[], int arr2[], int left, int right)
{
  int i = left, j = right;
  int tmp1,tmp2;
  int pivot = arr1[(left + right) / 2];

  // partition
  while (i <= j) {
    while (arr1[i] < pivot)i++;
    while (arr1[j] > pivot)j--;
    if (i <= j) {
      tmp1 = arr1[i];
      arr1[i] = arr1[j];
      arr1[j] = tmp1;

      tmp2 = arr2[i];
      arr2[i] = arr2[j];
      arr2[j] = tmp2;
      i++; j--;
    }
  };
  // recursion
  if (left < j)
    quickSort_2(arr1, arr2, left, j);
  if (i < right)
    quickSort_2(arr1, arr2, i, right);
}


void quickSort_dat(int arr[], char dat[], int left, int right, int elem_size)
{
  int i = left, j = right;
  int tmp;
  char* tmp_dat = (char *)xmalloc(sizeof(char)*elem_size);
  int pivot = arr[(left + right) / 2];

  // partition
  while (i <= j) {
    while (arr[i] < pivot)i++;
    while (arr[j] > pivot)j--;
    if (i <= j) {
      tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;

      //tmp_dat = dat[i];
      memcpy(tmp_dat,(void *)&dat[i*elem_size],elem_size);
      //dat[i] = dat[j];
      memcpy(&dat[i*elem_size],(void *)&dat[j*elem_size],elem_size);
      //dat[j] = tmp_dat;
      memcpy(&dat[j*elem_size],(void *)tmp_dat,elem_size);
      i++; j--;
    }
  };

  // recursion
  if (left < j)
    quickSort_dat(arr, dat, left, j, elem_size);
  if (i < right)
    quickSort_dat(arr, dat, i, right, elem_size);
  free(tmp_dat);
}

void quickSort_map(int arr[], int map[], int left, int right, int dim)
{
  int i = left, j = right;
  int tmp;
  int* tmp_map = (int *)xmalloc(sizeof(int)*dim);
  int pivot = arr[(left + right) / 2];

  // partition
  while (i <= j) {
    while (arr[i] < pivot)i++;
    while (arr[j] > pivot)j--;
    if (i <= j) {
      tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;

      //tmp_dat = dat[i];
      memcpy(tmp_map,(void *)&map[i*dim],dim*sizeof(int));
      //dat[i] = dat[j];
      memcpy(&map[i*dim],(void *)&map[j*dim],dim*sizeof(int));
      //dat[j] = tmp_dat;
      memcpy(&map[j*dim],(void *)tmp_map,dim*sizeof(int));
      i++; j--;
    }
  };

  // recursion
  if (left < j)
    quickSort_map(arr, map, left, j, dim);
  if (i < right)
    quickSort_map(arr, map, i, right, dim);
  free(tmp_map);
}

int removeDups(int a[], int array_size)
{
  int i, j;
  j = 0;
  // Remove the duplicates ...
  for (i = 1; i < array_size; i++)
  {
    if (a[i] != a[j])
    {
      j++;
      a[j] = a[i]; // Move it to the front
    }
  }
  // The new array size..
  array_size = (j + 1);
  return array_size;
}


//
//MPI Communicator for halo creation and exchange
//
MPI_Comm OP_MPI_WORLD;

/**---------------------MPI Halo related global variables -------------------**/

halo_list *OP_export_exec_list;//EEH list
halo_list *OP_import_exec_list;//IEH list

halo_list *OP_import_nonexec_list;//INH list
halo_list *OP_export_nonexec_list;//ENH list

//global array to hold dirty_bits for op_dats
int* dirtybit;

//halo exchange buffers for each op_dat
op_mpi_buffer *OP_mpi_buffer_list;


/*array to holding the index of the final element
that can be computed without halo exchanges for each set

0 to owned_num[set->index] - no halo exchange needed
owned_num[set->index] to n<set->size - halo exchange needed
*/
int *owned_num;

/*table holding MPI performance of each loop
(accessed via a hash of loop name) */
#define HASHSIZE 50
op_mpi_kernel op_mpi_kernel_tab[HASHSIZE];


//global variables to hold partition information on an MPI rank
int OP_part_index = 0;
part *OP_part_list;


//
//Save original partition ranges
//
int** orig_part_range = NULL;

/**-------------------------MPI halo utility functions ----------------------**/

//declare partition information for a given set
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

//get partition range on all mpi ranks for all sets
void get_part_range(int** part_range, int my_rank, int comm_size, MPI_Comm Comm)
{
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
#if DEBUG
      if(my_rank == 0)
        printf("range of %10s in rank %d: %d-%d\n",set->name,i,
            part_range[set->index][2*i], part_range[set->index][2*i+1]);
#endif
    }
    free(sizes);
  }
}

//get partition (i.e. mpi rank) where global_index is located and its local index
int get_partition(int global_index, int* part_range, int* local_index, int comm_size)
{
  for(int i = 0; i<comm_size; i++)
  {
    if (global_index >= part_range[2*i] &
        global_index <= part_range[2*i+1])
    {
      *local_index = global_index -  part_range[2*i];
      return i;
    }
  }
  return 0;
}

//convert a local index in to a global index
int get_global_index(int local_index, int partition, int* part_range, int comm_size)
{
  int g_index = part_range[2*partition]+local_index;
#if DEBUG
  if(g_index > part_range[2*(comm_size-1)+1])
    printf("Global index larger than set size\n");
#endif
  return g_index;
}



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


void create_list(int* list, int* ranks, int* disps, int* sizes, int* ranks_size,
    int* total, int* temp_list, int size, int comm_size, int my_rank)
{
  int index = 0;
  int total_size = 0;

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

void create_set_export_list(op_set set, int* temp_list, int size,
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

  halo_list h_list= (halo_list)xmalloc(sizeof(halo_list_core));
  h_list->set = set;
  h_list->size = total_size;
  h_list->ranks = ranks;
  h_list->ranks_size = ranks_size;
  h_list->disps = disps;
  h_list->sizes = sizes;
  h_list->list = list;

  OP_export_exec_list[set->index] = h_list;

}

void create_nonexec_set_import_list(op_set set, int* temp_list, int size,
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

  halo_list h_list= (halo_list)xmalloc(sizeof(halo_list_core));
  h_list->set = set;
  h_list->size = total_size;
  h_list->ranks = ranks;
  h_list->ranks_size = ranks_size;
  h_list->disps = disps;
  h_list->sizes = sizes;
  h_list->list = list;

  OP_import_nonexec_list[set->index] = h_list;

}

void create_set_import_list(op_set set, int* temp_list, int total_size,
    int* ranks, int* sizes, int ranks_size, int comm_size, int my_rank)
{
  int* disps = (int *)xmalloc(comm_size*sizeof(int));
  disps[0] = 0;
  for(int i=0; i<ranks_size; i++)
  {
    if(i>0)disps[i] = disps[i-1]+sizes[i-1];
  }

  halo_list h_list= (halo_list)xmalloc(sizeof(halo_list_core));
  h_list->set = set;
  h_list->size = total_size;
  h_list->ranks = ranks;
  h_list->ranks_size = ranks_size;
  h_list->disps = disps;
  h_list->sizes = sizes;
  h_list->list = temp_list;

  OP_import_exec_list[set->index] = h_list;
}

void create_nonexec_set_export_list(op_set set, int* temp_list, int total_size,
    int* ranks, int* sizes, int ranks_size, int comm_size, int my_rank)
{
  int* disps = (int *)xmalloc(comm_size*sizeof(int));
  disps[0] = 0;
  for(int i=0; i<ranks_size; i++)
  {
    if(i>0)disps[i] = disps[i-1]+sizes[i-1];
  }

  halo_list h_list= (halo_list)xmalloc(sizeof(halo_list_core));
  h_list->set = set;
  h_list->size = total_size;
  h_list->ranks = ranks;
  h_list->ranks_size = ranks_size;
  h_list->disps = disps;
  h_list->sizes = sizes;
  h_list->list = temp_list;

  OP_export_nonexec_list[set->index] = h_list;
}



/**--------------------------- Halo List Creation ---------------------------**/
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

  int* map_list[OP_map_index];
  int cap_s = 1000; //keep track of the temp array capacities


  for(int s=0; s<OP_set_index; s++){ //for each set
    op_set set=OP_set_list[s];

    //create a temporaty scratch space to hold export list for this set
    s_i = 0;cap_s = 1000;
    set_list = (int *)xmalloc(cap_s*sizeof(int));

    for(int e=0; e<set->size;e++){//for each elment of this set
      for(int m=0; m<OP_map_index; m++) { //for each maping table
        op_map map=OP_map_list[m];

        if(compare_sets(map->from,set)==1) //need to select mappings FROM this set
        {
          int part, local_index;
          for(int j=0; j<map->dim; j++) { //for each element pointed at by this entry
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
    //printf("creating set export list for set %10s of size %d\n",set->name,s_i);
    create_set_export_list(set,set_list,s_i, comm_size, my_rank);
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
    create_set_import_list(set, temp, index,neighbors, sizes,
        ranks_size, comm_size, my_rank);
  }


  /*-- STEP 3 - Exchange mapping table entries using the import/export lists--*/

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
          for(int j=0; j < map->dim; j++) { //for each element pointed at by this entry
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
                // not in this partition and not found in exec list
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
    create_nonexec_set_import_list(set,set_list, s_i, comm_size, my_rank);
    free(set_list);//free temp list
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
      if(sizes == NULL) {
        printf(" op_list_create -- error allocating memory: rbuf\n");
        exit(-1);
      }
      MPI_Recv(rbuf, sizes[i], MPI_INT, neighbors[i],s, OP_MPI_WORLD,
          MPI_STATUSES_IGNORE );
      memcpy(&temp[index],(void *)&rbuf[0],sizes[i]*sizeof(int));
      index = index + sizes[i];
      free(rbuf);
    }

    MPI_Waitall(list->ranks_size,request_send, MPI_STATUSES_IGNORE );

    //create import lists
    //printf("creating nonexec set export list with number of neighbors %d\n",ranks_size);
    create_nonexec_set_export_list(set, temp, index, neighbors, sizes,
        ranks_size, comm_size, my_rank);
  }


  /*-STEP 6 - Exchange execute set elements/data using the import/export lists--*/

  for(int s=0; s<OP_set_index; s++){ //for each set
    op_set set=OP_set_list[s];
    halo_list i_list = OP_import_exec_list[set->index];
    halo_list e_list = OP_export_exec_list[set->index];

    //for each data array
    for(int d=0; d<OP_dat_index; d++){
      op_dat dat=OP_dat_list[d];

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
        OP_dat_list[dat->index]->data = (char *)xrealloc(OP_dat_list[dat->index]->data,
            (set->size+i_list->size)*dat->size);

        int init = set->size*dat->size;
        for(int i=0; i<i_list->ranks_size; i++) {
          MPI_Recv(&(OP_dat_list[dat->index]->
                data[init+i_list->disps[i]*dat->size]),
              dat->size*i_list->sizes[i],
              MPI_CHAR, i_list->ranks[i], d,
              OP_MPI_WORLD, MPI_STATUSES_IGNORE);
        }

        MPI_Waitall(e_list->ranks_size,request_send, MPI_STATUSES_IGNORE );
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
    for(int d=0; d<OP_dat_index; d++){
      op_dat dat=OP_dat_list[d];

      if(compare_sets(set,dat->set)==1)//if this data array is defined on this set
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

        OP_dat_list[dat->index]->data = (char *)xrealloc(OP_dat_list[dat->index]->data,
            (set->size+exec_i_list->size+i_list->size)*dat->size);

        int init = (set->size+exec_i_list->size)*dat->size;
        for(int i=0; i < i_list->ranks_size; i++) {
          MPI_Recv(&(OP_dat_list[dat->index]->
                data[init+i_list->disps[i]*dat->size]),
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
          for(int j=0; j < map->dim; j++) { //for each element pointed at by this entry

            int part;
            int local_index = 0;
            part = get_partition(map->map[e*map->dim+j],
                part_range[map->to->index],&local_index,comm_size);

            if(part == my_rank)
            {
              OP_map_list[map->index]->map[e*map->dim+j] = local_index;
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

              if(rank2 >=0 & found <0)
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
                    from partition %d\n",set->name, local_index, my_rank,part );
            }
          }
        }
      }
    }
  }



  /*-STEP 9 ---------------- Create MPI send Buffers-----------------------*/

  OP_mpi_buffer_list = (op_mpi_buffer *)xmalloc(OP_dat_index*sizeof(op_mpi_buffer));

  for(int d=0; d<OP_dat_index; d++){//for each data array
    op_dat dat=OP_dat_list[d];

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
    mpi_buf->dat_index = dat->index;
    OP_mpi_buffer_list[dat->index] = mpi_buf;
  }


  //set dirty bits of all data arrays to 0
  //for each data array
  dirtybit = (int *)xmalloc(OP_dat_index*sizeof(int));

  for(int d=0; d<OP_dat_index; d++){
    op_dat dat=OP_dat_list[d];
    dirtybit[dat->index] = 0;
  }


  /*-STEP 10 -------------------- Separate owned elements------------------------*/

  owned_num = (int *)xmalloc(OP_dat_index*sizeof(int ));

  int** owned_elems = (int **)xmalloc(OP_set_index*sizeof(int *));
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
      owned_elems[set->index] = (int *)xmalloc(set->size*sizeof(int ));
      int count = 0;
      for(int e=0; e < set->size;e++){//for each elment of this set

        if((binary_search(exp_elems[set->index], e, 0, num_exp-1) < 0))
        {
          owned_elems[set->index][count++] = e;
        }
      }
      quickSort(owned_elems[set->index], 0, count-1);

      if(count+num_exp != set->size) printf("sizes not equal\n");
      owned_num[set->index] = count;

      //for each data array defined on this set seperate its elements
      for(int d=0; d<OP_dat_index; d++) { //for each set
        op_dat dat=OP_dat_list[d];

        if(compare_sets(set,dat->set)==1)//if this data array is defined on this set
        {
          char* new_dat = (char* )xmalloc(set->size*dat->size);
          for(int i = 0; i<count; i++)
          {
            memcpy(&new_dat[i*dat->size],
                &dat->data[owned_elems[set->index][i]*dat->size],
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

        if(compare_sets(map->from,set)==1)//if this mapping is defined from this set
        {
          int* new_map = (int *)xmalloc(set->size*map->dim*sizeof(int));
          for(int i = 0; i<count; i++)
          {
            memcpy(&new_map[i*map->dim],
                &map->map[owned_elems[set->index][i]*map->dim],
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
        if(index < 0) printf("Problem in seperating owned elements - exec list\n");
        else exec->list[i] = count + index;
      }

      for(int i  = 0; i< nonexec->size;i++)
      {
        int index = binary_search(owned_elems[set->index],
            nonexec->list[i], 0, count-1);
        if (index < 0)
        {
          index = binary_search(exp_elems[set->index],
              nonexec->list[i], 0, num_exp-1);
          if(index < 0) printf("Problem in seperating owned elements - nonexec list\n");
          else nonexec->list[i] = count + index;
        }
        else nonexec->list[i] = index;
      }
    }
    else
    {
      owned_elems[set->index] = (int *)xmalloc(set->size*sizeof(int ));
      exp_elems[set->index] = (int *)xmalloc(0*sizeof(int ));
      for(int e=0; e < set->size;e++){//for each elment of this set
        owned_elems[set->index][e] = e;
      }
      owned_num[set->index] = set->size;
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
      for(int j=0; j < map->dim; j++) { //for each element pointed at by this entry
        if(map->map[e*map->dim+j] < map->to->size)
        {
          int index = binary_search(owned_elems[map->to->index],
              map->map[e*map->dim+j],
              0, owned_num[map->to->index]-1);
          if(index < 0)
          {
            index = binary_search(exp_elems[map->to->index],
                map->map[e*map->dim+j],
                0, (map->to->size) - (owned_num[map->to->index]) -1);
            if(index < 0) printf("Problem in seperating owned elements - \
                renumbering map\n");
            else OP_map_list[map->index]->map[e*map->dim+j] =
              owned_num[map->to->index] + index;
          }
          else OP_map_list[map->index]->map[e*map->dim+j] = index;
        }
      }
    }
  }


  /*-STEP 11 ----------- Save the original set element indexes------------------*/

  if(OP_part_index != OP_set_index) //OP_part_list empty, (i.e. no previous partitioning done)
    //create it and store the seperation of elements using owned_elems and exp_elems
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
        g_index[i] = get_global_index(i,my_rank, part_range[set->index],comm_size);
        partition[i] = my_rank;
      }
      decl_partition(set, g_index, partition);

      //combine owned_elems and exp_elems to one memory block
      int* temp = (int *)xmalloc(sizeof(int)*set->size);
      memcpy(&temp[0], owned_elems[set->index],
          owned_num[set->index]*sizeof(int));
      memcpy(&temp[owned_num[set->index]], exp_elems[set->index],
          (set->size-owned_num[set->index])*sizeof(int));

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

      //combine owned_elems and exp_elems to one memory block
      int* temp = (int *)xmalloc(sizeof(int)*set->size);
      memcpy(&temp[0], owned_elems[set->index],
          owned_num[set->index]*sizeof(int));
      memcpy(&temp[owned_num[set->index]], exp_elems[set->index],
          (set->size-owned_num[set->index])*sizeof(int));

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

  /*-STEP 12 ---------- Clean up and Compute rough halo size numbers------------*/

  for(int i = 0; i<OP_set_index; i++)
  { free(part_range[i]);
    free(owned_elems[i]); free(exp_elems[i]);
  }
  free(part_range);
  free(exp_elems); free(owned_elems);

  op_timers(&cpu_t2, &wall_t2);  //timer stop for list create
  //compute import/export lists creation time
  time = wall_t2-wall_t1;
  MPI_Reduce(&time,&max_time,1,MPI_DOUBLE, MPI_MAX,0, OP_MPI_WORLD);

  //compute average halo size in Bytes
  int tot_halo_size = 0;
  for(int s = 0; s< OP_set_index; s++){
    op_set set=OP_set_list[s];

    for(int d=0; d<OP_dat_index; d++){
      op_dat dat=OP_dat_list[d];

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
  MPI_Reduce(&tot_halo_size,&avg_halo_size,1,MPI_INT, MPI_SUM,0, OP_MPI_WORLD);

  //print performance results
  if(my_rank==0)
  {
    printf("Max total halo creation time = %lf\n",max_time);
    printf("Average (worst case) Halo size = %d Bytes\n",avg_halo_size/comm_size);
  }
}



/**--------------------------- Clean-up Halo Lists --------------------------**/

void op_halo_destroy()
{
  //remove halos from op_dats
  for(int d=0; d<OP_dat_index; d++){
    op_dat dat=OP_dat_list[d];
    OP_dat_list[dat->index]->data = (char *)xrealloc(OP_dat_list[dat->index]->data,
        dat->set->size*dat->size);
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

  for(int d=0; d<OP_dat_index; d++){
    op_dat dat=OP_dat_list[d];

    free(OP_mpi_buffer_list[dat->index]->buf_exec);
    free(OP_mpi_buffer_list[dat->index]->buf_nonexec);
    free(OP_mpi_buffer_list[dat->index]->s_req);
    free(OP_mpi_buffer_list[dat->index]->r_req);
    free(OP_mpi_buffer_list[dat->index]);
  }
  free(OP_mpi_buffer_list);

  //cleanup performance data
#if COMM_PERF
  for (int n=0; n<HASHSIZE; n++) {
    free(op_mpi_kernel_tab[n].op_dat_indices);
    free(op_mpi_kernel_tab[n].tot_count);
    free(op_mpi_kernel_tab[n].tot_bytes);
  }
#endif
  MPI_Comm_free(&OP_MPI_WORLD);
}


/**-------------------------MPI Halo Exchange Functions----------------------**/

int exchange_halo(op_arg arg)
{
  op_dat dat = arg.dat;

  if((arg.idx != -1) && (arg.acc == OP_READ || arg.acc == OP_RW ) &&
      (dirtybit[dat->index] == 1))
  {
    //printf("Exchanging Halo of data array %10s\n",dat->name);
    halo_list imp_exec_list = OP_import_exec_list[dat->set->index];
    halo_list imp_nonexec_list = OP_import_nonexec_list[dat->set->index];

    halo_list exp_exec_list = OP_export_exec_list[dat->set->index];
    halo_list exp_nonexec_list = OP_export_nonexec_list[dat->set->index];

    //-------first exchange exec elements related to this data array--------

    //sanity checks
    if(compare_sets(imp_exec_list->set,dat->set)==0)
    { printf("Error: Import list and set mismatch\n"); exit(2);}
    if(compare_sets(exp_exec_list->set,dat->set)==0)
    {printf("Error: Export list and set mismatch\n"); exit(2);}

    int set_elem_index;
    for(int i=0; i<exp_exec_list->ranks_size; i++) {
      for(int j = 0; j < exp_exec_list->sizes[i]; j++)
      {
        set_elem_index = exp_exec_list->list[exp_exec_list->disps[i]+j];
        memcpy(&OP_mpi_buffer_list[dat->index]->
            buf_exec[exp_exec_list->disps[i]*dat->size+j*dat->size],
            (void *)&dat->data[dat->size*(set_elem_index)],dat->size);
      }
      //printf("export from %d to %d data %10s, number of elements of size %d | sending:\n ",
      //          my_rank, exp_exec_list->ranks[i], dat->name,exp_exec_list->sizes[i]);
      MPI_Isend(&OP_mpi_buffer_list[dat->index]->
          buf_exec[exp_exec_list->disps[i]*dat->size],
          dat->size*exp_exec_list->sizes[i],
          MPI_CHAR, exp_exec_list->ranks[i],
          dat->index, OP_MPI_WORLD,
          &OP_mpi_buffer_list[dat->index]->
          s_req[OP_mpi_buffer_list[dat->index]->s_num_req++]);
    }


    int init = dat->set->size*dat->size;
    for(int i=0; i < imp_exec_list->ranks_size; i++) {
      //printf("import on to %d from %d data %10s, number of elements of size %d | recieving:\n ",
      //      my_rank, imp_exec_list.ranks[i], dat.name, imp_exec_list.sizes[i]);
      MPI_Irecv(&(OP_dat_list[dat->index]->
            data[init+imp_exec_list->disps[i]*dat->size]),
          dat->size*imp_exec_list->sizes[i],
          MPI_CHAR, imp_exec_list->ranks[i],
          dat->index, OP_MPI_WORLD,
          &OP_mpi_buffer_list[dat->index]->
          r_req[OP_mpi_buffer_list[dat->index]->r_num_req++]);
    }

    //-----second exchange nonexec elements related to this data array------
    //sanity checks
    if(compare_sets(imp_nonexec_list->set,dat->set)==0)
    { printf("Error: Non-Import list and set mismatch"); exit(2);}
    if(compare_sets(exp_nonexec_list->set,dat->set)==0)
    {printf("Error: Non-Export list and set mismatch"); exit(2);}


    for(int i=0; i<exp_nonexec_list->ranks_size; i++) {
      for(int j = 0; j < exp_nonexec_list->sizes[i]; j++)
      {
        set_elem_index = exp_nonexec_list->list[exp_nonexec_list->disps[i]+j];
        memcpy(&OP_mpi_buffer_list[dat->index]->
            buf_nonexec[exp_nonexec_list->disps[i]*dat->size+j*dat->size],
            (void *)&dat->data[dat->size*(set_elem_index)],dat->size);
      }
      MPI_Isend(&OP_mpi_buffer_list[dat->index]->
          buf_nonexec[exp_nonexec_list->disps[i]*dat->size],
          dat->size*exp_nonexec_list->sizes[i],
          MPI_CHAR, exp_nonexec_list->ranks[i],
          dat->index, OP_MPI_WORLD,
          &OP_mpi_buffer_list[dat->index]->
          s_req[OP_mpi_buffer_list[dat->index]->s_num_req++]);
    }

    int nonexec_init = (dat->set->size+imp_exec_list->size)*dat->size;
    for(int i=0; i<imp_nonexec_list->ranks_size; i++) {
      MPI_Irecv(&(OP_dat_list[dat->index]->
            data[nonexec_init+imp_nonexec_list->disps[i]*dat->size]),
          dat->size*imp_nonexec_list->sizes[i],
          MPI_CHAR, imp_nonexec_list->ranks[i],
          dat->index, OP_MPI_WORLD,
          &OP_mpi_buffer_list[dat->index]->
          r_req[OP_mpi_buffer_list[dat->index]->r_num_req++]);
    }
    //clear dirty bit
    dirtybit[dat->index] = 0;
    return 1;
  }
  return 0;

}

void wait_all(op_arg arg)
{
  op_dat dat = arg.dat;
  MPI_Waitall(OP_mpi_buffer_list[dat->index]->s_num_req,
      OP_mpi_buffer_list[dat->index]->s_req,
      MPI_STATUSES_IGNORE );
  MPI_Waitall(OP_mpi_buffer_list[dat->index]->r_num_req,
      OP_mpi_buffer_list[dat->index]->r_req,
      MPI_STATUSES_IGNORE );
  OP_mpi_buffer_list[dat->index]->s_num_req = 0;
  OP_mpi_buffer_list[dat->index]->r_num_req = 0;
}


void set_dirtybit(op_arg arg)
{
  op_dat dat = arg.dat;

  if(arg.acc == OP_INC || arg.acc == OP_WRITE || arg.acc == OP_RW)
    dirtybit[dat->index] = 1;
}


void global_reduce(op_arg *arg)
{
  if(strcmp("double",arg->type)==0)
  {
    double result;
    if(arg->acc == OP_INC)//global reduction
    {
      MPI_Reduce((double *)arg->data,&result,1,MPI_DOUBLE, MPI_SUM,0, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(double));
    }
    else if(arg->acc == OP_MAX)//global maximum
    {
      MPI_Reduce((double *)arg->data,&result,1,MPI_DOUBLE, MPI_MAX,0, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(double));;
    }
    else if(arg->acc == OP_MIN)//global minimum
    {
      MPI_Reduce((double *)arg->data,&result,1,MPI_DOUBLE, MPI_MIN,0, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(double));
    }
  }
  else if(strcmp("float",arg->type)==0)
  {
    float result;
    if(arg->acc == OP_INC)//global reduction
    {
      MPI_Reduce((float *)arg->data,&result,1,MPI_FLOAT, MPI_SUM,0, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(float));
    }
    else if(arg->acc == OP_MAX)//global maximum
    {
      MPI_Reduce((float *)arg->data,&result,1,MPI_FLOAT, MPI_MAX,0, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(float));;
    }
    else if(arg->acc == OP_MIN)//global minimum
    {
      MPI_Reduce((float *)arg->data,&result,1,MPI_FLOAT, MPI_MIN,0, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(float));
    }
  }
  else if(strcmp("int",arg->type)==0)
  {
    int result;
    if(arg->acc == OP_INC)//global reduction
    {
      MPI_Reduce((int *)arg->data,&result,1,MPI_INT, MPI_SUM,0, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(int));
    }
    else if(arg->acc == OP_MAX)//global maximum
    {
      MPI_Reduce((int *)arg->data,&result,1,MPI_INT, MPI_MAX,0, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(int));;
    }
    else if(arg->acc == OP_MIN)//global minimum
    {
      MPI_Reduce((int *)arg->data,&result,1,MPI_INT, MPI_MIN,0, OP_MPI_WORLD);
      memcpy(arg->data, &result, sizeof(int));
    }
  }
}

void op_mpi_fetch_data(op_dat dat)
{
  //need the orig_part_range and OP_part_list

  //1. make a copy of the distributed op_dat on to a distributed temp op_dat
  //2. use orig_part_range to fill in OP_part_list[set->index]->elem_part with
  //   original partitioning information
  //3. migrate the temp op_dat to the correct MPI ranks
  //4. make a copy of the original g_index and migrate that also to the correct MPi process
  //4. sort elements according to g_index on the temp op_dat
  //5. return the temp op_dat to user application
}

void op_mpi_put_data(op_dat dat)
{
  //the op_dat in parameter list is modified
  //need the orig_part_range and OP_part_list

  //need to do some checks to see if the input op_dat has the same dimensions
  //and other values as the internal op_dat

  //1.
}

// initialise import halo data to NaN - for diagnostics pourposes
void reset_halo(op_arg arg)
{
  op_dat dat = arg.dat;

  if((arg.idx != -1) && (arg.acc == OP_READ || arg.acc == OP_RW ) &&
      (dirtybit[dat->index] == 1))
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
    memcpy(&(OP_dat_list[dat->index]->data[init]), NaN,
        dat->size*imp_exec_list->size + dat->size*imp_nonexec_list->size);
    free(NaN);
  }
}


/**-------------------Performance measurement and reporting------------------**/

void op_mpi_timing_output()
{
  int my_rank, comm_size;
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_WORLD, &comm_size);

  int count;
  double tot_time;
  double avg_time;
  int tot_bytes_rank,tot_bytes;

#if COMM_PERF

  printf("\n\n___________________________________\n");
  printf("Performance information on rank %d\n", my_rank);

  for (int n=0; n<HASHSIZE; n++) {
    if (op_mpi_kernel_tab[n].count>0) {
      printf("-----------------------------------\n");
      printf("Kernel        :  %10s\n",op_mpi_kernel_tab[n].name);
      printf("Count         :  %10.4d  \n", op_mpi_kernel_tab[n].count);
      printf("tot_time(sec) :  %10.4f  \n", op_mpi_kernel_tab[n].time);
      printf("avg_time(sec) :  %10.4f  \n",
          op_mpi_kernel_tab[n].time/op_mpi_kernel_tab[n].count );


      if(op_mpi_kernel_tab[n].num_indices>0)
      {
        printf("halo exchanges:  ");
        for(int i = 0; i<op_mpi_kernel_tab[n].num_indices; i++)
          printf("%10s ",OP_dat_list[op_mpi_kernel_tab[n].op_dat_indices[i]]->name);
        printf("\n");
        printf("       count  :  ");
        for(int i = 0; i<op_mpi_kernel_tab[n].num_indices; i++)
          printf("%10d ",op_mpi_kernel_tab[n].tot_count[i]);printf("\n");
        printf("total(Kbytes) :  ");
        for(int i = 0; i<op_mpi_kernel_tab[n].num_indices; i++)
          printf("%10d ",op_mpi_kernel_tab[n].tot_bytes[i]/1024);printf("\n");
        printf("average(bytes):  ");
        for(int i = 0; i<op_mpi_kernel_tab[n].num_indices; i++)
          printf("%10d ",op_mpi_kernel_tab[n].tot_bytes[i]/
              op_mpi_kernel_tab[n].tot_count[i] );printf("\n");
      }else
      {
        printf("halo exchanges:  %10s\n","NONE");
      }
    }
  }
  printf("___________________________________\n");
#endif


  if(my_rank == 0)
  {
    printf("Kernel        Count   Max time(sec)   Avg time(sec)  \n");
  }
  for (int n=0; n<HASHSIZE; n++) {
    MPI_Reduce(&op_mpi_kernel_tab[n].count,&count,1,MPI_INT, MPI_MAX,0, OP_MPI_WORLD);
    MPI_Reduce(&op_mpi_kernel_tab[n].time,&avg_time,1,MPI_DOUBLE, MPI_SUM,0, OP_MPI_WORLD);
    MPI_Reduce(&op_mpi_kernel_tab[n].time,&tot_time,1,MPI_DOUBLE, MPI_MAX,0, OP_MPI_WORLD);

    if(my_rank == 0 && count > 0)
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


unsigned hash(const char *s)
{
  unsigned hashval;
  for (hashval = 0; *s != '\0'; s++)
    hashval = *s + 31 * hashval;
  return hashval % HASHSIZE;
}


int op_mpi_perf_time(const char* name, double time)
{
  int kernel_index = hash(name);
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

void op_mpi_perf_comm(int kernel_index, op_arg arg)
{
  op_dat dat = arg.dat;

  halo_list exp_exec_list = OP_export_exec_list[dat->set->index];
  halo_list exp_nonexec_list = OP_export_nonexec_list[dat->set->index];


  int tot_halo_size = (exp_exec_list->size + exp_nonexec_list->size) * dat->size;

  int num_indices = op_mpi_kernel_tab[kernel_index].num_indices;

  if(num_indices == 0)
  {
    op_mpi_kernel_tab[kernel_index].op_dat_indices = (int *)xmalloc(1*sizeof(int));
    op_mpi_kernel_tab[kernel_index].tot_count = (int *)xmalloc(1*sizeof(int));
    op_mpi_kernel_tab[kernel_index].tot_bytes = (int *)xmalloc(1*sizeof(int));

    //clear first
    op_mpi_kernel_tab[kernel_index].tot_count[num_indices] = 0;
    op_mpi_kernel_tab[kernel_index].tot_bytes[num_indices] = 0;

    op_mpi_kernel_tab[kernel_index].op_dat_indices[num_indices] = dat->index;
    op_mpi_kernel_tab[kernel_index].tot_count[num_indices] += 1;
    op_mpi_kernel_tab[kernel_index].tot_bytes[num_indices] += tot_halo_size;

    op_mpi_kernel_tab[kernel_index].num_indices++;
  }
  else
  {
    int index = linear_search(op_mpi_kernel_tab[kernel_index].op_dat_indices,
        dat->index, 0, num_indices-1);

    if(index < 0)
    {
      op_mpi_kernel_tab[kernel_index].op_dat_indices =
        (int *)xrealloc(op_mpi_kernel_tab[kernel_index].op_dat_indices,
            (num_indices+1)*sizeof(int));

      op_mpi_kernel_tab[kernel_index].tot_count =
        (int *)xrealloc(op_mpi_kernel_tab[kernel_index].tot_count,
            (num_indices+1)*sizeof(int));

      op_mpi_kernel_tab[kernel_index].tot_bytes =
        (int *)xrealloc(op_mpi_kernel_tab[kernel_index].tot_bytes,
            (num_indices+1)*sizeof(int));

      //clear first
      op_mpi_kernel_tab[kernel_index].tot_count[num_indices] = 0;
      op_mpi_kernel_tab[kernel_index].tot_bytes[num_indices] = 0;

      op_mpi_kernel_tab[kernel_index].op_dat_indices[num_indices] = dat->index;
      op_mpi_kernel_tab[kernel_index].tot_count[num_indices] += 1;
      op_mpi_kernel_tab[kernel_index].tot_bytes[num_indices] += tot_halo_size;

      op_mpi_kernel_tab[kernel_index].num_indices++;
    }
    else
    {
      op_mpi_kernel_tab[kernel_index].tot_count[index] += 1;
      op_mpi_kernel_tab[kernel_index].tot_bytes[index] += tot_halo_size;
    }
  }
}


/**-------------------------------Output functions----------------------------**/


//mpi_gather a data array (of type double) and print its values on proc 0
//(i.e. proper element data values)
void gatherprint_tofile(op_dat dat, const char *file_name)
{
  //create new communicator for output
  int rank, comm_size;
  MPI_Comm OP_MPI_IO_WORLD;
  MPI_Comm_dup(MPI_COMM_WORLD, &OP_MPI_IO_WORLD);
  MPI_Comm_rank(OP_MPI_IO_WORLD, &rank);
  MPI_Comm_size(OP_MPI_IO_WORLD, &comm_size);

  op_dat data=OP_dat_list[dat->index];
  size_t mult = dat->size/dat->dim;

  double *l_array  = (double *) xmalloc(dat->dim*(dat->set->size)*sizeof(double));
  memcpy(l_array, (void *)&(OP_dat_list[dat->index]->data[0]),
      dat->size*dat->set->size);

  int l_size = dat->set->size;
  int elem_size = dat->dim;
  int* recevcnts = (int *) xmalloc(comm_size*sizeof(int));
  int* displs = (int *) xmalloc(comm_size*sizeof(int));
  int disp = 0;
  double *g_array;

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

  if(rank==0) g_array  = (double *) xmalloc(elem_size*g_size*sizeof(double));
  MPI_Gatherv(l_array, l_size*elem_size, MPI_DOUBLE, g_array, recevcnts,
      displs, MPI_DOUBLE, 0, OP_MPI_IO_WORLD);


  if(rank==0)
  {
    FILE *fp;
    if ( (fp = fopen("out_grid.dat","w")) == NULL) {
      printf("can't open file out_grid.dat\n"); exit(-1);
    }

    if (fprintf(fp,"%d %d\n",g_size, elem_size)<0)
    {
      printf("error writing to out_grid.dat\n"); exit(-1);
    }

    for(int i = 0; i< g_size; i++)
    {
      for(int j = 0; j < elem_size; j++ )
      {
        if (fprintf(fp,"%lf ",g_array[i*elem_size+j])<0)
        {
          printf("error writing to out_grid.dat\n"); exit(-1);
        }
      }
      fprintf(fp,"\n");
    }
    fclose(fp);
    free(g_array);
  }
  free(l_array);free(recevcnts);free(displs);
}

//mpi_gather a data array (of type double) and print its values on proc 0
//in binary form
void gatherprint_bin_tofile(op_dat dat, const char *file_name)
{
  //create new communicator for output
  int rank, comm_size;
  MPI_Comm OP_MPI_IO_WORLD;
  MPI_Comm_dup(MPI_COMM_WORLD, &OP_MPI_IO_WORLD);
  MPI_Comm_rank(OP_MPI_IO_WORLD, &rank);
  MPI_Comm_size(OP_MPI_IO_WORLD, &comm_size);

  op_dat data=OP_dat_list[dat->index];
  size_t mult = dat->size/dat->dim;

  double *l_array  = (double *) xmalloc(dat->dim*(dat->set->size)*sizeof(double));
  memcpy(l_array, (void *)&(OP_dat_list[dat->index]->data[0]),
      dat->size*dat->set->size);

  int l_size = dat->set->size;
  int elem_size = dat->dim;
  int* recevcnts = (int *) xmalloc(comm_size*sizeof(int));
  int* displs = (int *) xmalloc(comm_size*sizeof(int));
  int disp = 0;
  double *g_array;

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
  if(rank==0) g_array  = (double *) xmalloc(elem_size*g_size*sizeof(double));
  MPI_Gatherv(l_array, l_size*elem_size, MPI_DOUBLE, g_array, recevcnts,
      displs, MPI_DOUBLE, 0, OP_MPI_IO_WORLD);


  if(rank==0)
  {
    FILE *fp;
    if ( (fp = fopen(file_name,"wb")) == NULL) {
      printf("can't open file %s\n",file_name); exit(-1);
    }

    if (fwrite(&g_size, sizeof(int),1, fp)<1)
    {
      printf("error writing to %s",file_name); exit(-1);
    }
    if (fwrite(&elem_size, sizeof(int),1, fp)<1)
    {
      printf("error writing to %s\n",file_name); exit(-1);
    }

    for(int i = 0; i< g_size; i++)
    {
      if (fwrite( &g_array[i*elem_size], sizeof(double), elem_size, fp ) != 4)
      {
        printf("error writing to %s\n",file_name); exit(-1);
      }
    }
    fclose(fp);
    free(g_array);
  }

  free(l_array);free(recevcnts);free(displs);
}
