
// mpi header
#include <mpi.h>

#include <op_lib_c.h>
#include <op_lib_mpi.h>
#include <op_util.h>

#include <op_mpi_core.h>
#include <limits.h>

int** aug_part_range;
int* aug_part_range_size;
int* aug_part_range_cap;

int*** foreign_aug_part_range;
int** foreign_aug_part_range_size;

halo_list* OP_export_part_range_list;
halo_list* OP_import_part_range_list;

halo_list* OP_aug_export_exec_lists[10];  // 10 levels for now
halo_list* OP_aug_import_exec_lists[10];  // 10 levels for now

halo_list* OP_aug_export_nonexec_lists[10];  // 10 levels for now
halo_list* OP_aug_import_nonexec_lists[10];  // 10 levels for now

halo_list *OP_merged_import_exec_list;
halo_list *OP_merged_export_exec_list;

halo_list *OP_merged_import_nonexec_list;
halo_list *OP_merged_export_nonexec_list;


void print_array(int* arr, int size, const char* name, int my_rank){
  for(int i = 0; i < size; i++){
    printf("array my_rank=%d name=%s size=%d value[%d]=%d\n", my_rank, name, size, i, arr[i]);
  }
}

void print_halo(halo_list h_list, const char* name, int my_rank){
  if(h_list == NULL)
    return;
    
  printf("print_halo my_rank=%d, name=%s\n", my_rank, name);
  printf("print_halo my_rank=%d, name=%s, set=%s, size=%d\n", my_rank, name, h_list->set->name, h_list->size);

  for(int i = 0; i < h_list->ranks_size; i++){
    printf("print_halo my_rank=%d, name=%s, to_rank=%d, size=%d, disp=%d start>>>>>>>>>\n", my_rank, name, h_list->ranks[i], h_list->sizes[i], h_list->disps[i]);

    for(int j = h_list->disps[i]; j < h_list->disps[i] + h_list->sizes[i]; j++){
      printf("print_halo my_rank=%d, name=%s, to_rank=%d, size=%d, disp=%d value[%d]=%d\n", 
      my_rank, name, h_list->ranks[i], h_list->sizes[i], h_list->disps[i], j, h_list->list[j]);
    }

    printf("print_halo my_rank=%d, name=%s, to_rank=%d, size=%d, disp=%d end<<<<<<<<<<\n", my_rank, name, h_list->ranks[i], h_list->sizes[i], h_list->disps[i]);
  }
}

void print_aug_part_list(int** aug_range, int* aug_size, int my_rank){
  for(int i = 0; i < OP_set_index; i++){
    for(int j = 0; j < aug_size[i]; j++){
      printf("aug_range my_rank=%d set=%s size=%d value[%d]=%d\n", my_rank, OP_set_list[i]->name, aug_size[i], j, aug_range[i][j]);
    }
  }
}

void print_foreign_aug_part_list(int*** aug_range, int** aug_size, int comm_size, int my_rank){
  printf("foreign_aug_range start my_rank=%d\n", my_rank);
  for(int r = 0; r < comm_size; r++){
    for(int s = 0; s < OP_set_index; s++){
      for(int i = 0; i < aug_size[r][s]; i++){
        printf("foreign_aug_range my_rank=%d foreign=%d set=%s size=%d value[%d]=%d\n", my_rank, r, OP_set_list[s]->name, aug_size[r][s], i, aug_range[r][s][i]);
      }
    }
  }
  printf("foreign_aug_range end my_rank=%d\n", my_rank);
}

/*******************************************************************************
 * Routine to create an nonexec-import extended list (only a wrapper)
 *******************************************************************************/

static void create_nonexec_ex_import_list(op_set set, int *temp_list,
                                       halo_list h_list, int size,
                                       int comm_size, int my_rank) {
  create_export_list(set, temp_list, h_list, size, comm_size, my_rank);
}

/*******************************************************************************
 * Routine to create an nonexec-export extended list (only a wrapper)
 *******************************************************************************/

static void create_nonexec_ex_export_list(op_set set, int *temp_list,
                                       halo_list h_list, int total_size,
                                       int *ranks, int *sizes, int ranks_size,
                                       int comm_size, int my_rank) {
  create_import_list(set, temp_list, h_list, total_size, ranks, sizes,
                     ranks_size, comm_size, my_rank);
}

int get_max_value(int* arr, int from, int to){
  int max = 0;  // assumption: max >= 0
  for(int i = from; i < to; i++){
    if(max < arr[i]){
      max = arr[i];
    }  
  }
  return max;
}

void check_augmented_part_range(int* parts, int set_index, int value, int my_rank, int comm_size){
  for(int r = 0; r < comm_size; r++){
    parts[r] = -1;
    if(r != my_rank){
      for(int i = 0; i < foreign_aug_part_range_size[r][set_index]; i++){
        if(value == foreign_aug_part_range[r][set_index][i]){
          parts[r] = 1;
          break;
        }
      }
    }
  }
}

halo_list merge_halo_lists(int count, halo_list* h_lists, int my_rank, int comm_size){
  
  if(count < 1){
    printf("ERROR: No elements in the list to merge. count=%d\n", count);
    return NULL;
  }

  int start_index = 0;
  for(start_index = 0; start_index < count; start_index++){
    if(h_lists[start_index] != NULL)
      break;
  }

  op_set set = h_lists[start_index]->set;
  int init_ranks_size = 0;
  for(int i = start_index; i < count; i++){
    if(h_lists[i] != NULL && compare_sets(set, h_lists[i]->set) != 1){
      printf("ERROR: Invalid set merge");
      return NULL;
    }
    init_ranks_size += h_lists[i]->ranks_size;
  }

  int *temp_ranks = (int *)xmalloc(init_ranks_size * sizeof(int));

  int num_levels = 0;
  int new_size = 0;
  int new_ranks_size = 0;
  for(int i = 0; i < count; i++){
    halo_list h_list = h_lists[i];
    if(h_list == NULL)
      continue;

    memcpy(&(temp_ranks[new_ranks_size]), h_list->ranks, h_list->ranks_size * sizeof(int));

    num_levels += h_list->num_levels;
    new_size += h_list->size;
    new_ranks_size += h_list->ranks_size;
  }

  int tmp_rank_size = 0;
  if(new_ranks_size > 0){
    quickSort(temp_ranks, 0, (new_ranks_size - 1));
    tmp_rank_size = removeDups(temp_ranks, new_ranks_size);
  }

  new_ranks_size = tmp_rank_size * num_levels;

  int *list = (int *)xmalloc(new_size * sizeof(int));
  int *ranks = (int *)xmalloc(new_ranks_size * sizeof(int));
  int *disps = (int *)xmalloc(new_ranks_size * sizeof(int));
  int *sizes = (int *)xmalloc(new_ranks_size * sizeof(int));

  int* ranks_sizes = (int *)xmalloc(num_levels * sizeof(int));
  int* level_sizes = (int *)xmalloc(num_levels * sizeof(int));
  int* level_disps = (int *)xmalloc(num_levels * sizeof(int));
  int* rank_disps = (int *)xmalloc(num_levels * sizeof(int));

  int* sizes_by_rank = (int *)xmalloc(tmp_rank_size * sizeof(int));
  int* disps_by_rank = (int *)xmalloc(tmp_rank_size * sizeof(int));

  for(int i = 0; i < tmp_rank_size; i++){
    sizes_by_rank[i] = 0;
    disps_by_rank[i] = 0;
  }

  if(num_levels > 0){
    rank_disps[0] = 0;
    level_disps[0] = 0;
  }

  int list_start = 0;
  int rank_start = 0;
  int level_start = 0;
  for(int i = 0; i < count; i++){
    halo_list h_list = h_lists[i];
    if(h_list == NULL)
      continue;

    memcpy(&(list[list_start]), h_list->list, h_list->size * sizeof(int));

    for(int l = 0; l < h_list->num_levels; l++){
      for(int r = 0; r < tmp_rank_size; r++){
        int rank_index = binary_search(&h_list->ranks[h_list->rank_disps[l]], temp_ranks[r], 0, h_list->ranks_sizes[l] - 1);
        ranks[rank_start + r] = temp_ranks[r];
        if(rank_index >= 0){
          sizes[rank_start + r] = h_list->sizes[h_list->rank_disps[l] + rank_index];
          sizes_by_rank[r] += h_list->sizes[h_list->rank_disps[l] + rank_index];
        }else{
          sizes[rank_start + r] = 0;
        }
      }

      if(tmp_rank_size > 0){
        disps[rank_start] = 0;
      }
        
      for(int r = 1; r < tmp_rank_size; r++){
        disps[rank_start + r] = sizes[rank_start + r - 1] + disps[rank_start + r - 1];
      }

      ranks_sizes[level_start + l] = tmp_rank_size;
      level_sizes[level_start + l] = h_list->level_sizes[l];
      level_disps[level_start + l] = list_start + h_list->level_disps[l];

      if(level_start + l > 0)
        rank_disps[level_start + l] = ranks_sizes[level_start + l - 1] + rank_disps[level_start + l - 1];

      rank_start += tmp_rank_size; //h_list->ranks_size;
    }

    list_start += h_list->size;
    level_start += h_list->num_levels;
  }

  for(int r = 1; r < tmp_rank_size; r++){
    disps_by_rank[r] = sizes_by_rank[r - 1] + disps_by_rank[r - 1];
  }

  halo_list h_list = (halo_list)xmalloc(sizeof(halo_list_core));

  h_list->set = set;
  h_list->size = new_size;
  h_list->ranks = ranks;
  h_list->ranks_size = tmp_rank_size * num_levels;
  h_list->disps = disps;
  h_list->sizes = sizes;
  h_list->list = list;
  h_list->num_levels = num_levels;
  h_list->ranks_sizes = ranks_sizes;
  h_list->level_sizes = level_sizes;
  h_list->level_disps = level_disps;
  h_list->rank_disps = rank_disps;
  h_list->sizes_by_rank = sizes_by_rank;
  h_list->disps_by_rank = disps_by_rank;

  // for(int i = 0; i < count; i++){
  //   op_free(h_lists[i]->ranks);
  //   op_free(h_lists[i]->disps);
  //   op_free(h_lists[i]->sizes);
  //   op_free(h_lists[i]->list);
  //   op_free(h_lists[i]);
  // }
  return h_list;
}

halo_list merge_halo_lists_prev(halo_list h_list1, halo_list h_list2, int sort, int my_rank, int comm_size){

  if(!h_list1 && !h_list2){
    return NULL;
  }
  if(!h_list1 && h_list2){
    return h_list2;
  }
  if(h_list1 && !h_list2){
    return h_list1;
  }

  if(h_list1->ranks_size == 0 && h_list2->ranks_size == 0){
    return h_list1;
  }
  if(h_list1->ranks_size == 0 && h_list2->ranks_size != 0){
    return h_list2;
  }
  if(h_list1->ranks_size != 0 && h_list2->ranks_size == 0){
    return h_list1;
  }

  if (compare_sets(h_list1->set, h_list2->set) != 1) {
    printf("ERROR: Invalid set merge");
    return NULL;
  }

  int* h1_ranks = (int *)xmalloc(comm_size * sizeof(int));
  int* h2_ranks = (int *)xmalloc(comm_size * sizeof(int));

  int* h1_sizes = (int *)xmalloc(comm_size * sizeof(int));
  int* h2_sizes = (int *)xmalloc(comm_size * sizeof(int));

  int* h1_disps = (int *)xmalloc(comm_size * sizeof(int));
  int* h2_disps = (int *)xmalloc(comm_size * sizeof(int));

  int *ranks = (int *)xmalloc(comm_size * sizeof(int));
  int *disps = (int *)xmalloc(comm_size * sizeof(int));
  int *sizes = (int *)xmalloc(comm_size * sizeof(int));

  for(int i = 0; i < comm_size; i ++){
    h1_ranks[i] = -99;
    h2_ranks[i] = -99;
    h1_sizes[i] = -99;
    h2_sizes[i] = -99;
    h1_disps[i] = -99;
    h2_disps[i] = -99;
    ranks[i] = -99;
    disps[i] = -99;
    sizes[i] = -99;
  }

  for(int i = 0; i < h_list1->ranks_size; i++){
    h1_ranks[h_list1->ranks[i]] = 1;
    h1_sizes[h_list1->ranks[i]] = h_list1->sizes[i];
    h1_disps[h_list1->ranks[i]] = h_list1->disps[i];
  }

  for(int i = 0; i < h_list2->ranks_size; i++){
    h2_ranks[h_list2->ranks[i]] = 1;
    h2_sizes[h_list2->ranks[i]] = h_list2->sizes[i];
    h2_disps[h_list2->ranks[i]] = h_list2->disps[i];
  }

  //count cumulative size
  int merged_size = 0;
  for(int i = 0; i < comm_size; i++){
    if(h1_ranks[i] > 0){
      merged_size++;
    }
    else if(h2_ranks[i] > 0){
      merged_size++;
    }
  }
  
  int total_size = 0;
  for(int i = 0; i < comm_size; i++){
    if(h1_ranks[i] > 0){
      total_size += h1_sizes[i];
    }
    if(h2_ranks[i] > 0){
      total_size += h2_sizes[i];
    }
  }
  int* list = (int *)xmalloc(total_size * sizeof(int)); // max total size of the list

  int disp = 0;
  int start = 0;
  int index = 0;
  total_size = 0;

  
  for(int i = 0; i < comm_size; i++){
    int rank = -1;
    int size = 0;
    start = disp;
    if(h1_ranks[i] > 0){
      rank = i;
      for(int j = 0; j < h1_sizes[i]; j++){
        list[disp++] = h_list1->list[h1_disps[i] + j];
      }
    }
    if(h2_ranks[i] > 0){
      rank = i;
      for(int j = 0; j < h2_sizes[i]; j++){
        list[disp++] = h_list2->list[h2_disps[i] + j];
      }
    }

    if(disp > start){
      if(sort == 1)
        quickSort(&list[start], 0, (disp - start - 1));
      int new_size = removeDups(&list[start], (disp - start));

      if(new_size != disp - start){
        printf("ERROR duplicates new_size=%d prev_size=%d\n", new_size, disp - start);
      }else{
        printf("Merging duplicates new_size=%d prev_size=%d\n", new_size, disp - start);
      }
      disp = start + new_size;
    }

    if(h1_ranks[i] > 0 || h2_ranks[i] > 0){
      ranks[index] = rank;
      sizes[index] = disp - start;
      total_size += sizes[index];
      index++;
    }
  }

  disps[0] = 0;
  for (int i = 0; i < comm_size; i++) {
    if (i > 0)
      disps[i] = disps[i - 1] + sizes[i - 1];
  }

  halo_list h_list = (halo_list)xmalloc(sizeof(halo_list_core));

  h_list->set = h_list1->set;
  h_list->size = total_size;
  h_list->ranks = ranks;
  h_list->ranks_size = merged_size;
  h_list->disps = disps;
  h_list->sizes = sizes;
  h_list->list = list;

  // op_free(h_list1->ranks);
  // op_free(h_list1->disps);
  // op_free(h_list1->sizes);
  // op_free(h_list1->list);
  // op_free(h_list1);

  // op_free(h_list2->ranks);
  // op_free(h_list2->disps);
  // op_free(h_list2->sizes);
  // op_free(h_list2->list);
  // op_free(h_list2);


  op_free(h1_ranks);
  op_free(h2_ranks);
  op_free(h1_sizes);
  op_free(h2_sizes);
  op_free(h1_disps);
  op_free(h2_disps);

  return h_list;
}

halo_list* create_handshake_h_list(halo_list * h_lists, int **part_range, int my_rank, int comm_size){

  if(h_lists == NULL)
    return NULL;

  halo_list* handshake_h_list = (halo_list *)xmalloc(OP_set_index * sizeof(halo_list));

  int *neighbors, *sizes;
  int ranks_size;

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];

    //-----Discover neighbors-----
    halo_list list = h_lists[set->index];
    if(!list)
      continue;
    ranks_size = 0;
    neighbors = (int *)xmalloc(comm_size * sizeof(int));
    sizes = (int *)xmalloc(comm_size * sizeof(int));

    find_neighbors_set(list, neighbors, sizes, &ranks_size, my_rank, comm_size,
                       OP_MPI_WORLD);
    MPI_Request request_send[list->ranks_size];

    int *rbuf, cap = 0, index = 0;

    for (int i = 0; i < list->ranks_size; i++) {
      int *sbuf = &list->list[list->disps[i]];
      MPI_Isend(sbuf, list->sizes[i], MPI_INT, list->ranks[i], s, OP_MPI_WORLD,
                &request_send[i]);
    }

    for (int i = 0; i < ranks_size; i++)
      cap = cap + sizes[i];
    int *temp = (int *)xmalloc(cap * sizeof(int));

    // import this list from those neighbors
    for (int i = 0; i < ranks_size; i++) {
      rbuf = (int *)xmalloc(sizes[i] * sizeof(int));
      MPI_Recv(rbuf, sizes[i], MPI_INT, neighbors[i], s, OP_MPI_WORLD,
               MPI_STATUS_IGNORE);
      memcpy(&temp[index], (void *)&rbuf[0], sizes[i] * sizeof(int));
      index = index + sizes[i];
      op_free(rbuf);
    }

    MPI_Waitall(list->ranks_size, request_send, MPI_STATUSES_IGNORE);

    // create import lists
    halo_list h_list = (halo_list)xmalloc(sizeof(halo_list_core));
    create_import_list(set, temp, h_list, index, neighbors, sizes, ranks_size,
                       comm_size, my_rank);
    handshake_h_list[set->index] = h_list;
  }
  return handshake_h_list;
}

void create_aug_part_range(int halo_id, int **part_range, int my_rank, int comm_size){
  
  int exec_size = 0;

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];

    if(is_halo_required_for_set(set, halo_id) != 1){
      continue;
    }

    exec_size = 0;
    for(int l = 0; l <= halo_id; l++){
      exec_size += OP_aug_import_exec_lists[l][set->index]->size;
    }
    int start = 0;
    if(halo_id > 0){
      start = set->size + exec_size - OP_aug_import_exec_lists[halo_id][set->index]->size;
    }
    int end = set->size + exec_size;
   
    for (int e = start; e < end; e++) {      // for each elment of this set
      for (int m = 0; m < OP_map_index; m++) { // for each maping table
        op_map map = OP_map_list[m];

        if (compare_sets(map->from, set) == 1) { // need to select mappings from this set
        
          int part, local_index;
          for (int j = 0; j < map->dim; j++) { // for each element
                                               // pointed at by this entry
            part = get_partition(map->map[e * map->dim + j],
                                 part_range[map->to->index], &local_index,
                                 comm_size);
            if (aug_part_range_size[map->to->index] >= aug_part_range_cap[map->to->index]) {
              aug_part_range_cap[map->to->index] *= 2;
              aug_part_range[map->to->index] = (int *)xrealloc(aug_part_range[map->to->index], aug_part_range_cap[map->to->index] * sizeof(int));
            }

            if (part != my_rank) {
              aug_part_range[map->to->index][aug_part_range_size[map->to->index]++] = map->map[e * map->dim + j];
            }
          }
        }
      }
    }
  }

  for (int s = 0; s < OP_set_index; s++) {
    if(aug_part_range_size[s] > 0){
      quickSort(aug_part_range[s], 0, aug_part_range_size[s] - 1);
      aug_part_range_size[s] = removeDups(aug_part_range[s], aug_part_range_size[s]);
    }
    
  }
  // print_aug_part_list(aug_part_range, aug_part_range_size, my_rank);
}

void exchange_aug_part_ranges(int halo_id, int **part_range, int my_rank, int comm_size){

  OP_export_part_range_list = (halo_list*) xmalloc(OP_set_index * sizeof(halo_list));

  int* set_list;
  int s_i = 0;
  int cap_s = 1000;

  for (int s = 0; s < OP_set_index; s++) {

    op_set set = OP_set_list[s];
    if(is_halo_required_for_set(set, halo_id) != 1){
      continue;
    }
    s_i = 0;
    cap_s = 1000;
    set_list = (int*)xmalloc(aug_part_range_size[s] * comm_size * 2 * sizeof(int));

    for(int r = 0; r < comm_size; r++){
      if(r != my_rank){
        for(int i = 0; i < aug_part_range_size[s]; i++){
          set_list[s_i++] = r;
          set_list[s_i++] = aug_part_range[s][i];
        }
      }
    }

    halo_list h_list = (halo_list)xmalloc(sizeof(halo_list_core));
    create_export_list(set, set_list, h_list, s_i, comm_size, my_rank);
    OP_export_part_range_list[set->index] = h_list;
    op_free(set_list);
  }

  OP_import_part_range_list = create_handshake_h_list(OP_export_part_range_list, part_range, my_rank, comm_size);

  for (int s = 0; s < OP_set_index; s++) {
    halo_list h_list = OP_import_part_range_list[s];

    for(int r = 0; r < h_list->ranks_size; r++){
      if(foreign_aug_part_range_size[h_list->ranks[r]][s] < h_list->sizes[r]){
        foreign_aug_part_range[h_list->ranks[r]][s] = (int*)xrealloc(foreign_aug_part_range[h_list->ranks[r]][s], h_list->sizes[r] * sizeof(int));
      }
      
      foreign_aug_part_range_size[h_list->ranks[r]][s] = h_list->sizes[r];
      for(int i = 0; i < h_list->sizes[r]; i++){
        foreign_aug_part_range[h_list->ranks[r]][s][i] = h_list->list[h_list->disps[r] + i];
      }
    }
  }
  // print_foreign_aug_part_list(foreign_aug_part_range, foreign_aug_part_range_size, comm_size, my_rank);
}

bool is_in_prev_export_exec_halos(int halo_id, int set_index, int export_rank, int export_value, int my_rank){

  for(int i = 0; i < halo_id; i++){
    halo_list h_list = OP_aug_export_exec_lists[i][set_index];

    int rank_index = binary_search(h_list->ranks, export_rank, 0, h_list->ranks_size - 1);

    if(rank_index >= 0){
      for(int j = h_list->disps[rank_index]; j < h_list->disps[rank_index] + h_list->sizes[rank_index]; j++){
        if(h_list->list[j] == export_value){
          return true;
        }
      }
    }
  }
  return false;
}

halo_list* step1_create_aug_export_exec_list(int halo_id, int **part_range, int my_rank, int comm_size){

  halo_list* aug_export_exec_list = (halo_list *)xmalloc(OP_set_index * sizeof(halo_list));
  // declare temporaty scratch variables to hold set export lists and mapping
  // table export lists
  int s_i;
  int *set_list;

  int cap_s = 1000; // keep track of the temp array capacities
  int* parts;

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];

    if(is_halo_required_for_set(set, halo_id) != 1){
      aug_export_exec_list[set->index] = NULL;
      continue;
    }

    // create a temporaty scratch space to hold export list for this set
    s_i = 0;
    cap_s = 1000;
    set_list = (int *)xmalloc(cap_s * sizeof(int));
    parts = (int*)xmalloc(comm_size * sizeof(int));

    for (int e = 0; e < set->size; e++) {      // iterating upto set->size rather than upto nonexec size will eliminate duplicated being exported
      for (int m = 0; m < OP_map_index; m++) { // for each maping table
        op_map map = OP_map_list[m];

        if (compare_sets(map->from, set) == 1) { // need to select mappings
                                                 // FROM this set
          int part, local_index;
          for (int j = 0; j < map->dim; j++) { // for each element
                                               // pointed at by this entry
            part = get_partition(map->map[e * map->dim + j],
                                 part_range[map->to->index], &local_index,
                                 comm_size);

            check_augmented_part_range(parts, map->to->index, map->map[e * map->dim + j],
                                    my_rank, comm_size);
            if (s_i + comm_size * 2 >= cap_s) {
              cap_s = cap_s * 2;
              set_list = (int *)xrealloc(set_list, cap_s * sizeof(int));
            }

            if (part != my_rank && !is_in_prev_export_exec_halos(halo_id, set->index, part, e, my_rank)) {
              set_list[s_i++] = part; // add to set export list
              set_list[s_i++] = e;
            }

            for(int r = 0; r < comm_size; r++){
              if(r != part && parts[r] == 1 && !is_in_prev_export_exec_halos(halo_id, set->index, r, e, my_rank)){
                set_list[s_i++] = r; // add to set export list
                set_list[s_i++] = e;
              }
            }
          }
        }
      }
    }

    // create set export list
    halo_list h_list = (halo_list)xmalloc(sizeof(halo_list_core));
    create_export_list(set, set_list, h_list, s_i, comm_size, my_rank);
    aug_export_exec_list[set->index] = h_list;
    op_free(set_list);
    op_free(parts);
  }
  return aug_export_exec_list;
}

void step3_exchange_exec_mappings(int exec_level, int **part_range, int my_rank, int comm_size){

  for (int m = 0; m < OP_map_index; m++) { // for each maping table
    op_map map = OP_map_list[m];
    halo_list i_list = OP_aug_import_exec_lists[exec_level][map->from->index];
    halo_list e_list = OP_aug_export_exec_lists[exec_level][map->from->index];

    if(!i_list || !e_list)
      continue;

    MPI_Request request_send[e_list->ranks_size];

    // prepare bits of the mapping tables to be exported
    int **sbuf = (int **)xmalloc(e_list->ranks_size * sizeof(int *));

    for (int i = 0; i < e_list->ranks_size; i++) {
      sbuf[i] = (int *)xmalloc(e_list->sizes[i] * map->dim * sizeof(int));
      for (int j = 0; j < e_list->sizes[i]; j++) {
        for (int p = 0; p < map->dim; p++) {
          sbuf[i][j * map->dim + p] =
              map->map[map->dim * (e_list->list[e_list->disps[i] + j]) + p];
        }
      }
      MPI_Isend(sbuf[i], map->dim * e_list->sizes[i], MPI_INT, e_list->ranks[i],
                m, OP_MPI_WORLD, &request_send[i]);
    }

    // prepare space for the incomming mapping tables - realloc each
    // mapping tables in each mpi process

    int prev_exec_size = 0;
    for(int i = 0; i < exec_level; i++){
      halo_list prev_h_list = OP_aug_import_exec_lists[i][map->from->index];
      prev_exec_size += prev_h_list->size;
    }

    OP_map_list[map->index]->map = (int *)xrealloc(
        OP_map_list[map->index]->map,
        (map->dim * (map->from->size + prev_exec_size + i_list->size)) * sizeof(int));

    int init = map->dim * (map->from->size + prev_exec_size);
    for (int i = 0; i < i_list->ranks_size; i++) {
      MPI_Recv(
          &(OP_map_list[map->index]->map[init + i_list->disps[i] * map->dim]),
          map->dim * i_list->sizes[i], MPI_INT, i_list->ranks[i], m,
          OP_MPI_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Waitall(e_list->ranks_size, request_send, MPI_STATUSES_IGNORE);
    for (int i = 0; i < e_list->ranks_size; i++)
      op_free(sbuf[i]);
    op_free(sbuf);
  }
}

void step6_exchange_exec_data(int exec_level, int **part_range, int my_rank, int comm_size){
  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    halo_list i_list = OP_aug_import_exec_lists[exec_level][set->index];
    halo_list e_list = OP_aug_export_exec_lists[exec_level][set->index];

    if(!i_list || !e_list)
      continue;

    // for each data array
    op_dat_entry *item;
    int d = -1; // d is just simply the tag for mpi comms
    TAILQ_FOREACH(item, &OP_dat_list, entries) {
      d++; // increase tag to do mpi comm for the next op_dat
      op_dat dat = item->dat;

      if (compare_sets(set, dat->set) == 1) { // if this data array
                                              // is defined on this set
        MPI_Request request_send[e_list->ranks_size];

        // prepare execute set element data to be exported
        char **sbuf = (char **)xmalloc(e_list->ranks_size * sizeof(char *));

        for (int i = 0; i < e_list->ranks_size; i++) {
          sbuf[i] = (char *)xmalloc(e_list->sizes[i] * dat->size);
          for (int j = 0; j < e_list->sizes[i]; j++) {
            int set_elem_index = e_list->list[e_list->disps[i] + j];
            memcpy(&sbuf[i][j * dat->size],
                   (void *)&dat->data[dat->size * (set_elem_index)], dat->size);
          }
          MPI_Isend(sbuf[i], dat->size * e_list->sizes[i], MPI_CHAR,
                    e_list->ranks[i], d, OP_MPI_WORLD, &request_send[i]);
        }

        // prepare space for the incomming data - realloc each
        // data array in each mpi process

        int prev_exec_size = 0;
        for(int i = 0; i < exec_level; i++){
          halo_list prev_h_list = OP_aug_import_exec_lists[i][set->index];
          prev_exec_size += prev_h_list->size;
        }

        dat->data =
            (char *)xrealloc(dat->data, (set->size + prev_exec_size + i_list->size) * dat->size);

        int init = (set->size + prev_exec_size) * dat->size;
        for (int i = 0; i < i_list->ranks_size; i++) {
          MPI_Recv(&(dat->data[init + i_list->disps[i] * dat->size]),
                   dat->size * i_list->sizes[i], MPI_CHAR, i_list->ranks[i], d,
                   OP_MPI_WORLD, MPI_STATUS_IGNORE);
        }

        MPI_Waitall(e_list->ranks_size, request_send, MPI_STATUSES_IGNORE);
        for (int i = 0; i < e_list->ranks_size; i++)
          op_free(sbuf[i]);
        op_free(sbuf);
      }
    }
  }
}


void print_maps(int my_rank){

  printf("pmap my_rank=%d >>>>>>>>>>>>>>>start>>>>>>>\n", my_rank);
  for (int m = 0; m < OP_map_index; m++) { 
    op_map map = OP_map_list[m];
    int num_levels = map->from->halo_info->nhalos_count;
    int max_level = map->from->halo_info->max_nhalos;

    int exec_size = 0;
    for(int l = 0; l < max_level; l++){
      exec_size += OP_aug_import_exec_lists[l][map->from->index]->size;
    }

    int nonexec_size = 0;
    for(int l = 0; l < num_levels; l++){
      nonexec_size += OP_aug_import_nonexec_lists[l][map->from->index]->size;
    }
    int size = map->from->size + exec_size + nonexec_size;
    for(int i = 0; i < size; i++){
      for(int j = 0; j < map->dim; j++){
        printf("pmap my_rank=%d map=%s set=%s setsize=%d size=%d map[%d][%d]=%d\n", my_rank, map->name, map->from->name, size, map->from->size, i, j, map->map[i * map->dim + j]);
      } 
    }
  }
  printf("pmap my_rank=%d >>>>>>>>>>>>>>>end>>>>>>>\n", my_rank);  
}


void prepare_aug_maps(){

  for (int m = 0; m < OP_map_index; m++) { 
    op_map map = OP_map_list[m];
    map->aug_maps = (int **)malloc((size_t)map->from->halo_info->nhalos_count * sizeof(int *));

    int max_level = map->from->halo_info->max_nhalos;

    int exec_size = 0;
    for(int l = 0; l < max_level; l++){
      exec_size += OP_aug_import_exec_lists[l][map->from->index]->size;
    }

    map->map = (int *)xrealloc(map->map, (map->from->size + exec_size) * (size_t)map->dim * sizeof(int));        
    map->map_org = (int *)malloc((size_t)(map->from->size + exec_size) * (size_t)map->dim * sizeof(int));
    memcpy(map->map_org, map->map, (size_t)(map->from->size + exec_size) * (size_t)map->dim * sizeof(int));

    for(int el = 0; el < map->from->halo_info->nhalos_count; el++){
      int *m = (int *)malloc((size_t)(map->from->size + exec_size) * (size_t)map->dim * sizeof(int));
      if (m == NULL) {
        printf(" op_decl_map_core error -- error allocating memory to map\n");
        exit(-1);
      }
      map->aug_maps[el] = m;
    }
  }  
}

void prepare_aug_sets(){
  for(int s = 0; s < OP_set_index; s++){
    op_set set = OP_set_list[s];
    int max_level = set->halo_info->max_nhalos;

    printf("prepate_aug_sets set=%s exec_levels=%d\n", set->name, max_level);

    set->core_sizes =  (int *) malloc(max_level * sizeof(int));
    for(int i = 0; i < max_level; i++){
      set->core_sizes[i] = 0;
    }
  }
}

void step8_renumber_mappings(int dummy, int **part_range, int my_rank, int comm_size){

  prepare_aug_maps();
  // print_maps(my_rank);

  prepare_aug_sets();
  // return;
  int* parts;
  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];

    parts = (int*)xmalloc(comm_size * sizeof(int));

    for (int m = 0; m < OP_map_index; m++) { // for each maping table
      op_map map = OP_map_list[m];

      if (compare_sets(map->to, set) == 1) { // need to select
                                            // mappings TO this set

        int num_levels = map->from->halo_info->nhalos_count;
        int max_level = map->from->halo_info->max_nhalos;

        for(int el = 0; el < num_levels; el++){
          int exec_levels = map->from->halo_info->nhalos[el];

          // halo_list nonexec_set_list = OP_import_nonexec_list[set->index];
          halo_list nonexec_set_list = OP_aug_import_nonexec_lists[el][set->index];
          //get exec level size
          int exec_map_len = 0;
          for(int l = 0; l < exec_levels; l++){
            exec_map_len += OP_aug_import_exec_lists[l][map->from->index]->size;
          }

          // for each entry in this mapping table: original+execlist
          int len = map->from->size + exec_map_len;
          for (int e = 0; e < len; e++) {
            for (int j = 0; j < map->dim; j++) { // for each element
                                                // pointed at by this entry
              int part;
              int local_index = 0;
              part = get_partition(map->map_org[e * map->dim + j],
                                  part_range[map->to->index], &local_index,
                                  comm_size);

              check_augmented_part_range(parts, map->to->index, map->map_org[e * map->dim + j],
                                      my_rank, comm_size);

              if (part == my_rank) {
                if(exec_levels == 1){
                  OP_map_list[map->index]->map[e * map->dim + j] = local_index;
                  // printf("renumber00 my_rank=%d map=%s set=%s size=%d orgval[%d][%d]=%d prev=%d\n", 
                  // my_rank, map->name, map->from->name, len, e, j, map->map[e * map->dim + j], map->map_org[e * map->dim + j]);
                }
                OP_map_list[map->index]->aug_maps[el][e * map->dim + j] = local_index;
                // printf("renumber01 my_rank=%d map=%s set=%s size=%d augval[%d][%d][%d]=%d prev=%d\n", 
                //   my_rank, map->name, map->from->name, len, el, e, j, map->aug_maps[el][e * map->dim + j], map->map_org[e * map->dim + j]);
              } else {
                int found = -1;
                int rank1 = -1;
                for(int l = 0; l < exec_levels; l++){
                    found = -1;
                    rank1 = -1;

                    halo_list exec_set_list = OP_aug_import_exec_lists[l][set->index];
                    rank1 = binary_search(exec_set_list->ranks, part, 0,
                                        exec_set_list->ranks_size - 1);

                    if (rank1 >= 0) {
                      found = binary_search(exec_set_list->list, local_index,
                                          exec_set_list->disps[rank1],
                                          exec_set_list->disps[rank1] +
                                              exec_set_list->sizes[rank1] - 1);
                    }
                    //only one found should happen in this loop
                    if (found >= 0) {
                      int prev_exec_set_list_size = 0;
                      for(int l1 = 0; l1 < l; l1++){  //take the size of prev exec levels
                        prev_exec_set_list_size += OP_aug_import_exec_lists[l1][set->index]->size;
                      }
                      if(exec_levels == 1){
                        OP_map_list[map->index]->map[e * map->dim + j] =
                            found + map->to->size + prev_exec_set_list_size;
                            // printf("renumber10 my_rank=%d map=%s set=%s size=%d orgval[%d][%d]=%d prev=%d\n", 
                            //   my_rank, map->name, map->from->name, len, e, j, map->map[e * map->dim + j], map->map_org[e * map->dim + j]);
                      }
                      
                      OP_map_list[map->index]->aug_maps[el][e * map->dim + j] =
                          found + map->to->size + prev_exec_set_list_size;

                      // printf("renumber11 my_rank=%d map=%s set=%s size=%d augval[%d][%d][%d]=%d prev=%d\n", 
                      //     my_rank, map->name, map->from->name, len, el, e, j, map->aug_maps[el][e * map->dim + j], map->map_org[e * map->dim + j]);
                      break;
                    }
                }
                // check in nonexec list
                int rank2 = binary_search(nonexec_set_list->ranks, part, 0,
                                          nonexec_set_list->ranks_size - 1);

                

                if (rank2 >= 0 && found < 0) {
                  found = binary_search(nonexec_set_list->list, local_index,
                                        nonexec_set_list->disps[rank2],
                                        nonexec_set_list->disps[rank2] +
                                            nonexec_set_list->sizes[rank2] - 1);
                  if (found >= 0) {
                    int exec_set_list_size = 0;
                    for(int l = 0; l < max_level; l++){// for(int l = 0; l < exec_levels; l++){
                      exec_set_list_size += OP_aug_import_exec_lists[l][set->index]->size;
                    }

                    //todo: this is not needed since we do have duplicates in the non exec section
                    int non_exec_set_list_size = 0;
                    for(int l = 0; l < el; l++){
                      non_exec_set_list_size += OP_aug_import_nonexec_lists[l][set->index]->size;
                    }

                    // printf("step8 map=%s set=%s el=%d nonexec_size=%d\n", map->name, set->name, el, non_exec_set_list_size);
                    if(exec_levels == 1){
                      OP_map_list[map->index]->map[e * map->dim + j] =
                          found + set->size + exec_set_list_size + non_exec_set_list_size;
                      // printf("renumber20 my_rank=%d map=%s set=%s size=%d orgval[%d][%d]=%d prev=%d nonexec_size=%d\n", 
                      //     my_rank, map->name, map->from->name, len, e, j, map->map[e * map->dim + j], map->map_org[e * map->dim + j], non_exec_set_list_size);
                    }

                    OP_map_list[map->index]->aug_maps[el][e * map->dim + j] =
                        found + set->size + exec_set_list_size + non_exec_set_list_size;
                    // printf("renumber21 my_rank=%d map=%s set=%s size=%d augval[%d][%d][%d]=%d prev=%d set_size=%d exec_size=%d nonexec_size=%d\n", 
                    //       my_rank, map->name, map->from->name, len, el, e, j, map->aug_maps[el][e * map->dim + j], OP_map_list[map->index]->map_org[e * map->dim + j], 
                    //       set->size, exec_set_list_size, non_exec_set_list_size);
                  }
                }

                if (found < 0)
                  printf("ERROR: Set %10s Element %d needed on rank %d \
                      from partition %d\n",
                        set->name, local_index, my_rank, part);
              }
            }
          }
        }
      }
      
    }
    
  }
}

void step8_renumber_mappings_prev(int exec_levels, int **part_range, int my_rank, int comm_size){

  int* parts;
  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    parts = (int*)xmalloc(comm_size * sizeof(int));

    for (int m = 0; m < OP_map_index; m++) { // for each maping table
      op_map map = OP_map_list[m];

      if (compare_sets(map->to, set) == 1) { // need to select
                                             // mappings TO this set

        halo_list nonexec_set_list = OP_import_nonexec_list[set->index];
        //get exec level size
        int exec_map_len = 0;
        for(int l = 0; l < exec_levels; l++){
          exec_map_len += OP_aug_import_exec_lists[l][map->from->index]->size;
        }

        // for each entry in this mapping table: original+execlist
        int len = map->from->size + exec_map_len;
        for (int e = 0; e < len; e++) {
          for (int j = 0; j < map->dim; j++) { // for each element
                                               // pointed at by this entry
            int part;
            int local_index = 0;
            part = get_partition(map->map[e * map->dim + j],
                                 part_range[map->to->index], &local_index,
                                 comm_size);

            check_augmented_part_range(parts, map->to->index, map->map[e * map->dim + j],
                                    my_rank, comm_size);

            if (part == my_rank) {
              OP_map_list[map->index]->map[e * map->dim + j] = local_index;
            } else {
              int found = -1;
              int rank1 = -1;
              for(int l = 0; l < exec_levels; l++){
                  found = -1;
                  rank1 = -1;

                  halo_list exec_set_list = OP_aug_import_exec_lists[l][set->index];
                  rank1 = binary_search(exec_set_list->ranks, part, 0,
                                      exec_set_list->ranks_size - 1);

                  if (rank1 >= 0) {
                    found = binary_search(exec_set_list->list, local_index,
                                        exec_set_list->disps[rank1],
                                        exec_set_list->disps[rank1] +
                                            exec_set_list->sizes[rank1] - 1);
                  }
                  //only one found should happen in this loop
                  if (found >= 0) {
                    int prev_exec_set_list_size = 0;
                    for(int l1 = 0; l1 < l; l1++){  //take the size of prev exec levels
                      prev_exec_set_list_size += OP_aug_import_exec_lists[l1][set->index]->size;
                    }
                    OP_map_list[map->index]->map[e * map->dim + j] =
                        found + map->to->size + prev_exec_set_list_size;
                    break;
                  }
              }
              // check in nonexec list
              int rank2 = binary_search(nonexec_set_list->ranks, part, 0,
                                        nonexec_set_list->ranks_size - 1);

              

              if (rank2 >= 0 && found < 0) {
                found = binary_search(nonexec_set_list->list, local_index,
                                      nonexec_set_list->disps[rank2],
                                      nonexec_set_list->disps[rank2] +
                                          nonexec_set_list->sizes[rank2] - 1);
                if (found >= 0) {
                  int exec_set_list_size = 0;
                  for(int l = 0; l < exec_levels; l++){
                    exec_set_list_size += OP_aug_import_exec_lists[l][set->index]->size;
                  }
                  OP_map_list[map->index]->map[e * map->dim + j] =
                      found + set->size + exec_set_list_size;
                }
              }

              if (found < 0)
                printf("ERROR: Set %10s Element %d needed on rank %d \
                    from partition %d\n",
                       set->name, local_index, my_rank, part);
            }
          }
        }
      }
    }
  }
}

void step4_import_nonexec(int dummy, int **part_range, int my_rank, int comm_size){

  // OP_import_nonexec_list =
  //     (halo_list *)xmalloc(OP_set_index * sizeof(halo_list));
  // OP_export_nonexec_list =
  //     (halo_list *)xmalloc(OP_set_index * sizeof(halo_list));

  // declare temporaty scratch variables to hold non-exec set export lists
  int s_i;
  int *set_list;
  int cap_s = 1000; // keep track of the temp array capacities

  s_i = 0;
  set_list = NULL;
  cap_s = 1000; // keep track of the temp array capacity

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];

    int exec_levels = 0;
    int num_levels = set->halo_info->nhalos_count;
    for(int el = 0; el < num_levels; el++){

      if(!OP_aug_import_nonexec_lists[el]){
        OP_aug_import_nonexec_lists[el] = (halo_list *)xmalloc(OP_set_index * sizeof(halo_list));
        OP_aug_export_nonexec_lists[el] = (halo_list *)xmalloc(OP_set_index * sizeof(halo_list));
      }
      
      exec_levels = set->halo_info->nhalos[el];
      printf("set=%s el[%d]=%d\n", set->name, el, exec_levels);
    
      // create a temporaty scratch space to hold nonexec export list for this set
      s_i = 0;
      set_list = (int *)xmalloc(cap_s * sizeof(int));

      for (int m = 0; m < OP_map_index; m++) { // for each maping table
        op_map map = OP_map_list[m];
        int exec_size = 0;
        for(int l = 0; l < exec_levels; l++){
          // exec_size += OP_aug_import_exec_lists[l][map->from->index]->size;
          exec_size += (OP_aug_import_exec_lists[l] && OP_aug_import_exec_lists[l][map->from->index]) ? 
          OP_aug_import_exec_lists[l][map->from->index]->size : 0;
        }

        if (compare_sets(map->to, set) == 1) { // need to select
                                              // mappings TO this set

          // for each entry in this mapping table: original+execlist
          int len = map->from->size + exec_size;
          for (int e = 0; e < len; e++) {
            int part;
            int local_index;
            for (int j = 0; j < map->dim; j++) { // for each element pointed
                                                // at by this entry
              part = get_partition(map->map[e * map->dim + j],
                                  part_range[map->to->index], &local_index,
                                  comm_size);

              if (part != my_rank) {
                int found = -1;
                int rank = -1;

                for(int l = 0; l < exec_levels; l++){
                  found = -1;
                  rank = -1;

                  halo_list exec_set_list = (OP_aug_import_exec_lists[l]) ? OP_aug_import_exec_lists[l][set->index] : NULL;
                  if(!exec_set_list)
                    continue;
                  rank = binary_search(exec_set_list->ranks, part, 0,
                                        exec_set_list->ranks_size - 1);

                  if (rank >= 0) {
                    found = binary_search(exec_set_list->list, local_index,
                                          exec_set_list->disps[rank],
                                          exec_set_list->disps[rank] +
                                              exec_set_list->sizes[rank] - 1);
                  }
                  if(found >= 0){
                    break;
                  }
                }
                if (s_i >= cap_s) {
                  cap_s = cap_s * 2;
                  set_list = (int *)xrealloc(set_list, cap_s * sizeof(int));
                }

                if (found < 0) {
                  // not in this partition and not found in
                  // exec list
                  // add to non-execute set_list
                  set_list[s_i++] = part;
                  set_list[s_i++] = local_index;
                }
              }
            }
          }
        }
      }

      // create non-exec set import list
      halo_list h_list = (halo_list)xmalloc(sizeof(halo_list_core));
      create_nonexec_ex_import_list(set, set_list, h_list, s_i, comm_size, my_rank);
      op_free(set_list); // free temp list
      // OP_import_nonexec_list[set->index] = h_list;
      OP_aug_import_nonexec_lists[el][set->index] = h_list;
    }
  }
}

void step7_halo(int exec_levels, int **part_range, int my_rank, int comm_size){

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    int num_levels = set->halo_info->nhalos_count;
    int non_exec_size = 0;

    for(int el = 0; el < num_levels; el++){
      halo_list i_list = OP_aug_import_nonexec_lists[el][set->index];
      halo_list e_list = OP_aug_export_nonexec_lists[el][set->index];

      // for each data array
      op_dat_entry *item;
      int d = -1; // d is just simply the tag for mpi comms
      TAILQ_FOREACH(item, &OP_dat_list, entries) {
        d++; // increase tag to do mpi comm for the next op_dat
        op_dat dat = item->dat;

        if (compare_sets(set, dat->set) == 1) { // if this data array is
                                                // defined on this set

          MPI_Request request_send[e_list->ranks_size];

          // prepare non-execute set element data to be exported
          char **sbuf = (char **)xmalloc(e_list->ranks_size * sizeof(char *));

          for (int i = 0; i < e_list->ranks_size; i++) {
            sbuf[i] = (char *)xmalloc(e_list->sizes[i] * dat->size);
            for (int j = 0; j < e_list->sizes[i]; j++) {
              int set_elem_index = e_list->list[e_list->disps[i] + j];
              memcpy(&sbuf[i][j * dat->size],
                    (void *)&dat->data[dat->size * (set_elem_index)], dat->size);
            }
            MPI_Isend(sbuf[i], dat->size * e_list->sizes[i], MPI_CHAR,
                      e_list->ranks[i], d, OP_MPI_WORLD, &request_send[i]);
          }

          int exec_size = 0;
          for(int l = 0; l < exec_levels; l++){
            exec_size += OP_aug_import_exec_lists[l][set->index] ? 
            OP_aug_import_exec_lists[l][set->index]->size : 0;
          }
          // prepare space for the incomming nonexec-data - realloc each
          // data array in each mpi process
          dat->data = (char *)xrealloc(
              dat->data,
              (set->size + exec_size + non_exec_size + i_list->size) * dat->size);

          int init = (set->size + exec_size + non_exec_size) * dat->size;
          for (int i = 0; i < i_list->ranks_size; i++) {
            MPI_Recv(&(dat->data[init + i_list->disps[i] * dat->size]),
                    dat->size * i_list->sizes[i], MPI_CHAR, i_list->ranks[i], d,
                    OP_MPI_WORLD, MPI_STATUS_IGNORE);
          }

          
          printf("step4 set=%s size=%d core=%d exec=%d non[%d]=%d\n", set->name, set->size, set->core_size, exec_size,  el, non_exec_size);
          MPI_Waitall(e_list->ranks_size, request_send, MPI_STATUSES_IGNORE);
          for (int i = 0; i < e_list->ranks_size; i++)
            op_free(sbuf[i]);
          op_free(sbuf);
        }
      }
      non_exec_size += OP_aug_import_nonexec_lists[el][set->index]->size;
      printf("step4 new set=%s size=%d core=%d non[%d]=%d\n", set->name, set->size, set->core_size,  el, non_exec_size);
    }
  }
}


void step7_halo_prev(int exec_levels, int **part_range, int my_rank, int comm_size){

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    halo_list i_list = OP_import_nonexec_list[set->index];
    halo_list e_list = OP_export_nonexec_list[set->index];

    // for each data array
    op_dat_entry *item;
    int d = -1; // d is just simply the tag for mpi comms
    TAILQ_FOREACH(item, &OP_dat_list, entries) {
      d++; // increase tag to do mpi comm for the next op_dat
      op_dat dat = item->dat;

      if (compare_sets(set, dat->set) == 1) { // if this data array is
                                              // defined on this set

        MPI_Request request_send[e_list->ranks_size];

        // prepare non-execute set element data to be exported
        char **sbuf = (char **)xmalloc(e_list->ranks_size * sizeof(char *));

        for (int i = 0; i < e_list->ranks_size; i++) {
          sbuf[i] = (char *)xmalloc(e_list->sizes[i] * dat->size);
          for (int j = 0; j < e_list->sizes[i]; j++) {
            int set_elem_index = e_list->list[e_list->disps[i] + j];
            memcpy(&sbuf[i][j * dat->size],
                   (void *)&dat->data[dat->size * (set_elem_index)], dat->size);
          }
          MPI_Isend(sbuf[i], dat->size * e_list->sizes[i], MPI_CHAR,
                    e_list->ranks[i], d, OP_MPI_WORLD, &request_send[i]);
        }

        int exec_size = 0;
        for(int l = 0; l < exec_levels; l++){
          exec_size += OP_aug_import_exec_lists[l][set->index] ? 
          OP_aug_import_exec_lists[l][set->index]->size : 0;
        }
        // prepare space for the incomming nonexec-data - realloc each
        // data array in each mpi process
        dat->data = (char *)xrealloc(
            dat->data,
            (set->size + exec_size + i_list->size) * dat->size);

        int init = (set->size + exec_size) * dat->size;
        for (int i = 0; i < i_list->ranks_size; i++) {
          MPI_Recv(&(dat->data[init + i_list->disps[i] * dat->size]),
                   dat->size * i_list->sizes[i], MPI_CHAR, i_list->ranks[i], d,
                   OP_MPI_WORLD, MPI_STATUS_IGNORE);
        }

        MPI_Waitall(e_list->ranks_size, request_send, MPI_STATUSES_IGNORE);
        for (int i = 0; i < e_list->ranks_size; i++)
          op_free(sbuf[i]);
        op_free(sbuf);
      }
    }
  }
}

void step9_halo(int exec_levels, int **part_range, int my_rank, int comm_size){
  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;

    op_mpi_buffer mpi_buf = (op_mpi_buffer)xmalloc(sizeof(op_mpi_buffer_core));

    int exec_e_list_size = 0;
    int exec_e_ranks_size = 0;
    int exec_i_ranks_size = 0;
    int imp_exec_size = 0;

    for(int l = 0; l < exec_levels; l++){
      if(OP_aug_export_exec_lists[l][dat->set->index]){
        exec_e_list_size += OP_aug_export_exec_lists[l][dat->set->index]->size;
        exec_e_ranks_size += OP_aug_export_exec_lists[l][dat->set->index]->ranks_size;
      }
      exec_i_ranks_size += OP_aug_import_exec_lists[l][dat->set->index]->ranks_size;
      imp_exec_size += OP_aug_import_exec_lists[l][dat->set->index]->size;
    }

    int nonexec_e_list_size = 0;
    int nonexec_e_ranks_size = 0;
    int nonexec_i_ranks_size = 0;
    int imp_nonexec_size = 0;

    int num_levels = dat->set->halo_info->nhalos_count;
    for(int l = 0; l < num_levels; l++){
      nonexec_e_list_size += OP_aug_export_nonexec_lists[l][dat->set->index]->size;
      nonexec_e_ranks_size += OP_aug_export_nonexec_lists[l][dat->set->index]->ranks_size;
      nonexec_i_ranks_size += OP_aug_import_nonexec_lists[l][dat->set->index]->ranks_size;
      imp_nonexec_size += OP_aug_import_nonexec_lists[l][dat->set->index]->size;
    }

    mpi_buf->buf_exec = (char *)xmalloc(exec_e_list_size * dat->size);
    // halo_list nonexec_e_list = OP_export_nonexec_list[dat->set->index];
    // mpi_buf->buf_nonexec = (char *)xmalloc((nonexec_e_list->size) * dat->size);
    mpi_buf->buf_nonexec = (char *)xmalloc(nonexec_e_list_size * dat->size);

    // halo_list merged_e_list = OP_merged_export_exec_list[dat->set->index];
    // halo_list merged_i_list = OP_merged_import_exec_list[dat->set->index];

    halo_list nonexec_i_list = OP_import_nonexec_list[dat->set->index];

    mpi_buf->s_req = (MPI_Request *)xmalloc(
        sizeof(MPI_Request) *
        (exec_e_ranks_size + nonexec_e_ranks_size)); //nonexec_e_list->ranks_size));
    mpi_buf->r_req = (MPI_Request *)xmalloc(
        sizeof(MPI_Request) *
        (exec_i_ranks_size + nonexec_i_ranks_size)); //nonexec_i_list->ranks_size));

    mpi_buf->s_num_req = 0;
    mpi_buf->r_num_req = 0;
    dat->mpi_buffer = mpi_buf;

    char *new_aug_data = (char *)op_malloc(dat->size * (dat->set->size + imp_exec_size + 
    imp_nonexec_size) * sizeof(char));
    dat->aug_data = new_aug_data;
  }

  // set dirty bits of all data arrays to 0
  // for each data array
  item = NULL;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    dat->dirtybit = 0;
  }
}

int remove_elements_from_array(op_set set, int my_rank, int start, int* ret_arr, int* arr, int size, int* elememts, int element_size){

  // printf("remove_elements_from_array my_rank=%d set=%s start>>>>>>>>>>>\n", my_rank, set->name);
  int index = 0;
  for(int i = 0; i < size; i++){
    if(element_size > 0){
      int found = binary_search(elememts, arr[i], 0, element_size - 1);
      if(found < 0){
        // printf("remove_elements_from_array my_rank=%d set=%s start=%d index=%d added[%d]=%d\n", my_rank, set->name, start, start + index, i, arr[i]);
        ret_arr[index++] = arr[i];
        
      }else{
        // printf("remove_elements_from_array my_rank=%d set=%s index=%d removed[%d]=%d\n", my_rank, set->name, index, i, arr[i]);
      }
    }else{
      ret_arr[index++] = arr[i];
    }
  }

  // printf("remove_elements_from_array my_rank=%d set=%s end<<<<<<<<<<\n", my_rank, set->name);

  if(size - element_size == index){
    printf("remove_elements_from_array subset size=%d elem=%d index=%d\n", size, element_size, index);
  }else{
    printf("remove_elements_from_array ERROR not a subset size=%d elem=%d index=%d\n", size, element_size, index);
    // exit(0);
  }
  return index;
}

void step10_halo(int dummy, int **part_range, int **core_elems, int **exp_elems, int my_rank, int comm_size){

  int*** temp_core_elems = (int***) xmalloc(OP_set_index *  sizeof(int**));
  int*** temp_exp_elems = (int***) xmalloc(OP_set_index *  sizeof(int**));
 
  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];

    int num_levels = set->halo_info->nhalos_count;
    int max_level = set->halo_info->max_nhalos;

    temp_core_elems[set->index] = (int **)xmalloc(num_levels * sizeof(int*));
    temp_exp_elems[set->index] = (int **)xmalloc(num_levels * sizeof(int*));

    for(int el = 0; el < num_levels; el++){

      temp_core_elems[set->index][el] = (int *)xmalloc(set->size * sizeof(int));
      int exec_levels = set->halo_info->nhalos[el];

      int exec_size = 0;
      halo_list exec[exec_levels];
      for(int l = 0; l < exec_levels; l++){
        exec[l] = OP_aug_export_exec_lists[l][set->index];
        if(exec[l])
          exec_size += exec[l]->size;
      }

      // halo_list nonexec = OP_aug_export_nonexec_lists[el][set->index]; // OP_export_nonexec_list[set->index];
      temp_exp_elems[set->index][el] = (int *)xmalloc(exec_size * sizeof(int)); //exp_elems[set->index] = (int *)xmalloc(exec_size * sizeof(int));
      if (exec_size > 0) {
        
        int prev_exec_size = 0;
        for(int l = 0; l < exec_levels; l++){
          if(exec[l]){
            memcpy(&temp_exp_elems[set->index][el][prev_exec_size], exec[l]->list, exec[l]->size * sizeof(int));
            // memcpy(&exp_elems[set->index][prev_exec_size], exec[l]->list, exec[l]->size * sizeof(int));
            prev_exec_size += exec[l]->size;
          }
        }
        quickSort(temp_exp_elems[set->index][el], 0, exec_size - 1);
        int num_exp = removeDups(temp_exp_elems[set->index][el], exec_size);
        // quickSort(exp_elems[set->index], 0, exec_size - 1);
        // int num_exp = removeDups(exp_elems[set->index], exec_size);
        
        int count = 0;
        for (int e = 0; e < set->size; e++) { // for each elment of this set

          if ((binary_search(temp_exp_elems[set->index][el], e, 0, num_exp - 1) < 0)) {
            temp_core_elems[set->index][el][count++] = e;
          }
        }
        quickSort(temp_core_elems[set->index][el], 0, count - 1);

        if (count + num_exp != set->size)
          printf("sizes not equal\n");
        
        if(exec_levels == 1){
          set->core_size = count;
        }
        
        set->core_sizes[el] = count;

        printf("step10 my_rank=%d set=%s el=%d levels=%d setsize=%d core_size=%d exec_size=%d\n", my_rank, set->name, el, exec_levels, set->size, count, num_exp);

      } else {
        temp_core_elems[set->index][el] = (int *)xmalloc(set->size * sizeof(int));
        temp_exp_elems[set->index][el] = (int *)xmalloc(0 * sizeof(int));
        for (int e = 0; e < set->size; e++) { // for each elment of this set
          temp_core_elems[set->index][el][e] = e;
        }
        if(exec_levels == 1){
          set->core_size = set->size;
        }
        set->core_sizes[el] = set->size;

        printf("step10 my_rank=%d set=%s el=%d levels=%d setsize=%d core_size=%d exec_size=%d\n", my_rank, set->name, el, exec_levels, set->size, set->core_sizes[el], 0);
      }
    }

    
    //create core_elems and exp_elems arrays based on the temp arrays

    core_elems[set->index] = (int *)xmalloc(set->size * sizeof(int));
    // exp_elems[set->index] = (int *)xmalloc(exec_size * sizeof(int));

    int min_core_size = INT_MAX;
    int min_level = -1;

    int max_core_size = -1;
    int max_core_level = -1;
    for(int l = 0; l < num_levels; l++){
      if(min_core_size > set->core_sizes[l]){ //todo: = not added to take the first occurance
        min_core_size = set->core_sizes[l];
        min_level = l; //set->dat_to_execlevels->get_val_at(l);
      }
      if(max_core_size < set->core_sizes[l]){
        max_core_size = set->core_sizes[l];
        max_core_level = l; //set->dat_to_execlevels->get_val_at(l);
      }
    }
   
    printf("step10 my_rank=%d set=%s min_level=%d min_core_size=%d max_core=%d core0=%d max_level=%d\n", 
    my_rank, set->name, min_level, min_core_size, max_core_size, set->core_sizes[0], max_core_level);

    int start = 0;
    for(int l = num_levels - 1; l >= 0; l--){
      
      int size = remove_elements_from_array(set, my_rank, start, &core_elems[set->index][start],
      temp_core_elems[set->index][l], set->core_sizes[l],
      (l == num_levels - 1) ? NULL : temp_core_elems[set->index][l + 1], (l == num_levels - 1) ? 0 : set->core_sizes[l + 1]);
      // printf("step10 ==== my_rank=%d set=%s l=%d setsize=%d start=%d  size-corediff=%d\n", my_rank, set->name, l, set->size, start, size);
     
      // printf("elemchk start my_rank=%d set=%s size=%d l=%d >>>>>>>>\n", my_rank, set->name, size,l );
      // for(int i = 0; i < size; i++){
      //   printf("elemchk my_rank=%d set=%s size=%d core[%d]=%d tmpcore[%d]=%d diff[%d]=%d\n", 
      //   my_rank, set->name, size, i, core_elems[set->index][start + i], i, temp_core_elems[set->index][l][start + i], i, core_elems[set->index][start + i] - temp_core_elems[set->index][l][start + i]);
      // }
      // printf("elemchk end my_rank=%d set=%s size=%d l=%d <<<<<<<<<<\n", my_rank, set->name, size, l);
       start = set->core_sizes[l];
    }

    // continue;

     
    // int* temp = (int*)malloc(set->core_sizes[0] * sizeof(int));
    // // int size = remove_elements_from_array(set, my_rank, temp, core_elems[set->index], max_core_size, temp_core_elems[set->index][0], max_core_size);
    // int size = remove_elements_from_array(set, my_rank, temp, temp_core_elems[set->index][max_core_level], max_core_size, core_elems[set->index], max_core_size);
   
    // printf("test my_rank=%d set=%s retsize=%d max_core_level=%d s1=%d s2=%d\n", my_rank, set->name, size, max_core_level, max_core_size, set->core_sizes[0]);
    
    
    // for(int i = 0; i < max_core_size; i++){
    //   printf("elemchk my_rank=%d set=%s size=%d core[%d]=%d tempcore[%d]=%d diff=%d\n", 
    //   my_rank, set->name, max_core_size, i, core_elems[set->index][i], i, temp_core_elems[set->index][max_core_level][i], core_elems[set->index][i] - temp_core_elems[set->index][max_core_level][i]);
    // }
    // continue;
    int exec_size = set->size - max_core_size; //set->core_sizes[0];
    exp_elems[set->index] = (int *)xmalloc(exec_size * sizeof(int));
    // printf("step10 prevexp_elems my_rank=%d set=%s set->size=%d core=%d exec_size=%d\n", my_rank, set->name, set->size, set->core_sizes[0], exec_size);

    // if(exec_size > 0)
    //   memcpy(exp_elems[set->index], temp_exp_elems[set->index][max_core_level], exec_size * sizeof(int));

    // printf("step10 exp_elems my_rank=%d set=%s set->size=%d exec_size=%d\n", my_rank, set->name, set->size, exec_size);

    

    int num_exp = 0;
    int found = -1;
    if(max_core_size >= 0){
      for (int e = 0; e < set->size; e++) { // for each elment of this set
        found = -1;
        for(int el = num_levels - 1; el >= 0; el--){
          found = binary_search(core_elems[set->index], e, (el == num_levels - 1) ? 0 : set->core_sizes[el + 1], set->core_sizes[el] - 1);
          if(found >= 0){
            break;
          }
        }
        if (found < 0) {

          if(num_exp > exec_size){
            printf("ERROR temp num_exp>exec_size my_rank=%d set=%s num_exp=%d exec_size=%d max_core_size=%d\n", my_rank, set->name, num_exp, exec_size, max_core_size);
            exit(0);
          }
          exp_elems[set->index][num_exp++] = e;
          // printf("test my_rank=%d set=%s max_core_size=%d size=%d exec[%d]=%d\n", my_rank, set->name, max_core_size, exec_size, num_exp - 1, exp_elems[set->index][num_exp - 1]);
        }
      }
      if(exec_size > 0)
        quickSort(exp_elems[set->index], 0, exec_size - 1);
    }
    
    //  printf("step10 <<<<<<<<<>>>>>>>>.>>>>>>>>> done my_rank=%d set=%s min_level=%d min_core_size=%d\n", my_rank, set->name, min_level, min_core_size);
    
    // for(int i = 0; i < exec_size; i++){
    //   printf("test my_rank=%d set=%s size=%d exec[%d]=%d\n", my_rank, set->name, set->core_sizes[0], i, exp_elems[set->index][i]);
    // }

    // char name[50];
    // int pos = 0;
    // pos += sprintf(&name[pos], "exparr_%s", set->name);
    // pos += sprintf(&name[pos], "_%d", my_rank);

    //   print_array(exp_elems[set->index], num_exp, name, my_rank);

  }

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];

    int num_levels = set->halo_info->nhalos_count;
    int max_level = set->halo_info->max_nhalos;

    int min_core_size = INT_MAX;
    int min_level = -1;

    int max_core_size = -1;
    int max_core_level = -1;
    for(int l = 0; l < num_levels; l++){
      if(min_core_size > set->core_sizes[l]){ //todo: = not added to take the first occurance
        min_core_size = set->core_sizes[l];
        min_level = l; //set->dat_to_execlevels->get_val_at(l);
      }
      if(max_core_size < set->core_sizes[l]){
        max_core_size = set->core_sizes[l];
        max_core_level = l; //set->dat_to_execlevels->get_val_at(l);
      }
    }

    int count = max_core_size; //set->core_sizes[0];
    int num_exp = set->size - count;
    // printf("step10 coreandexec my_rank=%d set=%s size=%d count=%d num_exp=%d\n", my_rank, set->name, set->size, count, num_exp);
    // for each data array defined on this set seperate its elements
    op_dat_entry *item;
    TAILQ_FOREACH(item, &OP_dat_list, entries) {
      op_dat dat = item->dat;

      if (compare_sets(set, dat->set) == 1) // if this data array is
      // defined on this set
      {
        char *new_dat = (char *)xmalloc(set->size * dat->size);
        for (int i = 0; i < count; i++) {
          memcpy(&new_dat[i * dat->size],
                &dat->data[core_elems[set->index][i] * dat->size],
                dat->size);
        }
        for (int i = 0; i < num_exp; i++) {
          memcpy(&new_dat[(count + i) * dat->size],
                &dat->data[exp_elems[set->index][i] * dat->size], dat->size);
        }
        memcpy(&dat->data[0], &new_dat[0], set->size * dat->size);
        op_free(new_dat);
      }
    }

    for (int m = 0; m < OP_map_index; m++) { // for each set
      op_map map = OP_map_list[m];

      if (compare_sets(map->from, set) == 1) { // if this mapping is
                                              // defined from this set
        //todo: this is for the standard op2
        int *new_map = (int *)xmalloc(set->size * map->dim * sizeof(int));
        for (int i = 0; i < count; i++) {
          memcpy(&new_map[i * map->dim],
                &map->map[core_elems[set->index][i] * map->dim],
                map->dim * sizeof(int));
        }
        for (int i = 0; i < num_exp; i++) {
          memcpy(&new_map[(count + i) * map->dim],
                &map->map[exp_elems[set->index][i] * map->dim],
                map->dim * sizeof(int));
        }
        memcpy(&map->map[0], &new_map[0], set->size * map->dim * sizeof(int));
        op_free(new_map);
        
        //todo: for aug maps
        //todo: this has also to be done with the common core and exec maps
        for(int el = 0; el < num_levels; el++){
          // int tmp_count = set->core_sizes[el];
          int *aug_map = (int *)xmalloc(set->size * map->dim * sizeof(int));
          for (int i = 0; i < count; i++) {
            memcpy(&aug_map[i * map->dim],
                  &map->aug_maps[el][core_elems[set->index][i] * map->dim],
                  map->dim * sizeof(int));
          }
          for (int i = 0; i < set->size - count; i++) {
            memcpy(&aug_map[(count + i) * map->dim],
                  &map->aug_maps[el][exp_elems[set->index][i] * map->dim],
                  map->dim * sizeof(int));
          }
          memcpy(&map->aug_maps[el][0], &aug_map[0], set->size * map->dim * sizeof(int));
          op_free(aug_map);
        }
        
        // for(int el = 0; el < num_levels; el++){
        //   int tmp_count = set->core_sizes[el];
        //   int *aug_map = (int *)xmalloc(set->size * map->dim * sizeof(int));
        //   for (int i = 0; i < tmp_count; i++) {
        //     memcpy(&aug_map[i * map->dim],
        //           &map->aug_maps[el][temp_core_elems[set->index][el][i] * map->dim],
        //           map->dim * sizeof(int));
        //   }
        //   for (int i = 0; i < set->size - tmp_count; i++) {
        //     memcpy(&aug_map[(tmp_count + i) * map->dim],
        //           &map->aug_maps[el][temp_exp_elems[set->index][el][i] * map->dim],
        //           map->dim * sizeof(int));
        //   }
        //   memcpy(&map->aug_maps[el][0], &aug_map[0], set->size * map->dim * sizeof(int));
        //   op_free(aug_map);
        // }
      }
    }

    // for(int el = 0; el < num_levels; el++){
      int exec_levels = set->halo_info->max_nhalos;// set->halo_info->nhalos[el];

      halo_list exec[exec_levels];
      for(int l = 0; l < exec_levels; l++){
        exec[l] = OP_aug_export_exec_lists[l][set->index];
        if(!exec[l])
          continue;
        for (int i = 0; i < exec[l]->size; i++) {
          int index =
              binary_search(exp_elems[set->index], exec[l]->list[i], 0, num_exp - 1);
          if (index < 0){ //todo: this element can be in the core list of another

            int core_index = -1;
            int start_index = 0;
            for(int l1 = num_levels - 1; l1 >= 0; l1--){
              start_index = (l1 == num_levels - 1) ? 0 : set->core_sizes[l1 + 1];
              core_index = binary_search(core_elems[set->index], exec[l]->list[i], start_index, set->core_sizes[l1] - 1);
              if(core_index >= 0){
                break;
              }
            }
            if (core_index < 0){
              printf("Problem in seperating core elements - exec list my_rank=%d set=%s val=%d core=%d index=%d count=%d exp=%d\n", 
              my_rank, set->name, exec[l]->list[i], core_index, index, count, num_exp);
              char name[50];
              int pos = 0;
              pos += sprintf(&name[pos], "core_%s", set->name);
              pos += sprintf(&name[pos], "_%d", my_rank);

               print_array(exp_elems[set->index], num_exp, name, my_rank);
            }else{
              exec[l]->list[i] = start_index + core_index;
            }
          }
          else
            exec[l]->list[i] = count + index;
        }
      }
    // }

    for(int l = 0; l < num_levels; l++){

      halo_list nonexec[num_levels];
      nonexec[l] = OP_aug_export_nonexec_lists[l][set->index];
      for (int i = 0; i < nonexec[l]->size; i++) {
        int index =
              binary_search(exp_elems[set->index], nonexec[l]->list[i], 0, num_exp - 1);
          if (index < 0){ //todo: this element can be in the core list of another

            int core_index = -1;
            int start_index = 0;
            for(int l1 = num_levels - 1; l1 >= 0; l1--){
              start_index = (l1 == num_levels - 1) ? 0 : set->core_sizes[l1 + 1];
              core_index = binary_search(core_elems[set->index], nonexec[l]->list[i], start_index, set->core_sizes[l1] - 1);
              if(core_index >= 0){
                break;
              }
            }
            if (core_index < 0){
              printf("Problem in seperating core elements - nonexec list set=%s val=%d core=%d index=%d count=%d exp=%d\n", 
              set->name, nonexec[l]->list[i], core_index, index, count, num_exp);
            }else{
              nonexec[l]->list[i] = start_index + core_index;
            }
          }
          else
            nonexec[l]->list[i] = count + index;
      }

    }


    
  }

  // for (int m = 0; m < OP_map_index; m++) { // for each set
  //   op_map map = OP_map_list[m];

  //   int num_levels = 1; //map->from->halo_info->nhalos_count;
  //   int max_level = map->from->halo_info->max_nhalos;

  //   for(int el = 0; el < num_levels; el++){
  //     int exec_levels = map->from->halo_info->nhalos[el];
  //     int imp_exec_size = 0;
  //     for(int l = 0; l < exec_levels; l++){
  //       imp_exec_size += OP_aug_import_exec_lists[l][map->from->index] ? 
  //       OP_aug_import_exec_lists[l][map->from->index]->size : 0;
  //     }
  //     // for each entry in this mapping table: original+execlist
  //     int len = map->from->size + imp_exec_size;
  //     for (int e = 0; e < len; e++) {
  //       for (int j = 0; j < map->dim; j++) { // for each element pointed
  //                                           // at by this entry
  //         if (map->map[e * map->dim + j] < map->to->size) {

  //           int index =
  //             binary_search(exp_elems[map->to->index], map->map[e * map->dim + j], 0, (map->to->size) - (map->to->core_sizes[0]) - 1);  //todo: always take 0 size
  //           if (index < 0){ //todo: this element can be in the core list of another

  //             int core_index = -1;
  //             int start_index = 0;
  //             for(int l1 = num_levels - 1; l1 >= 0; l1--){
  //               start_index = (l1 == num_levels - 1) ? 0 : map->to->core_sizes[l1 + 1];
  //               core_index = binary_search(core_elems[map->to->index], map->map[e * map->dim + j], start_index, map->to->core_sizes[l1] - 1);
  //               if(core_index >= 0){
  //                 break;
  //               }
  //             }
  //             if (core_index < 0){
  //               printf("Problem in seperating core elements - renumbering map list maporg set=%s val=%d core=%d index=%d count=%d exp=%d\n", 
  //               map->to->name, map->map[e * map->dim + j], core_index, index, map->to->core_sizes[el], (map->to->size) - (map->to->core_sizes[0]));
  //             }else{
  //               // printf("step10renumber00 my_rank=%d map=%s set=%s size=%d start=%d index=%d val[%d][%d]=%d prev=%d\n", 
  //               //   my_rank, map->name, map->from->name, len, e, j, start_index, core_index, start_index + core_index, OP_map_list[map->index]->map[e * map->dim + j]);
  //               // if(exec_levels == 1){
  //                 OP_map_list[map->index]->map[e * map->dim + j] = start_index + core_index;
                  
  //               // }
  //             }
  //           }
  //           else{
  //             // if(exec_levels == 1){
  //               // printf("step10renumber00 my_rank=%d map=%s set=%s size=%d orgval[%d][%d]=%d prev=%d\n", 
  //               //   my_rank, map->name, map->from->name, len, e, j, map->to->core_sizes[0] + index, OP_map_list[map->index]->map[e * map->dim + j]);

  //                 OP_map_list[map->index]->map[e * map->dim + j] = map->to->core_sizes[0] + index;

                  
  //               // }
  //           }
              
  //         }
  //       }
  //     }

   

  //   }
  // }

//  print_maps(my_rank);

  //todo: for augmaps

    // now need to renumber mapping tables as the elements are seperated
  for (int m = 0; m < OP_map_index; m++) { // for each set
    op_map map = OP_map_list[m];

    int num_levels = map->from->halo_info->nhalos_count;
    int max_level = map->from->halo_info->max_nhalos;

    for(int el = 0; el < num_levels; el++){
      int exec_levels = map->from->halo_info->nhalos[el];
      int imp_exec_size = 0;
      for(int l = 0; l < exec_levels; l++){
        imp_exec_size += OP_aug_import_exec_lists[l][map->from->index] ? 
        OP_aug_import_exec_lists[l][map->from->index]->size : 0;
      }
      // for each entry in this mapping table: original+execlist
      int len = map->from->size + imp_exec_size;
      for (int e = 0; e < len; e++) {
        for (int j = 0; j < map->dim; j++) { // for each element pointed
                                            // at by this entry
          if (map->aug_maps[el][e * map->dim + j] < map->to->size) {

            int index =
              binary_search(exp_elems[map->to->index], map->aug_maps[el][e * map->dim + j], 0, (map->to->size) - (map->to->core_sizes[0]) - 1);  //todo: always take 0 size
            if (index < 0){ //todo: this element can be in the core list of another

              int core_index = -1;
              int start_index = 0;
              for(int l1 = num_levels - 1; l1 >= 0; l1--){
                start_index = (l1 == num_levels - 1) ? 0 : map->to->core_sizes[l1 + 1];
                core_index = binary_search(core_elems[map->to->index], map->aug_maps[el][e * map->dim + j], start_index, map->to->core_sizes[l1] - 1);
                if(core_index >= 0){
                  break;
                }
              }
              if (core_index < 0){
                printf("Problem in seperating core elements - renumbering map list augmap set=%s val=%d core=%d index=%d count=%d exp=%d\n", 
                map->to->name, map->aug_maps[el][e * map->dim + j], core_index, index, map->to->core_sizes[el], (map->to->size) - (map->to->core_sizes[0]));
              }else{
                if(exec_levels == 1){
                  OP_map_list[map->index]->map[e * map->dim + j] = start_index + core_index;
                }
                OP_map_list[map->index]->aug_maps[el][e * map->dim + j] = start_index + core_index;
              }
            }
            else{
              if(exec_levels == 1){
                  OP_map_list[map->index]->map[e * map->dim + j] = map->to->core_sizes[0] + index;
                }
                OP_map_list[map->index]->aug_maps[el][e * map->dim + j] =
                    map->to->core_sizes[0] + index; //todo: always take 0 size
            }
              
          }
        }
      }

      // char name[50];
      // int pos = 0;
      // pos += sprintf(&name[pos], "renumarray_%s", map->name);
      // pos += sprintf(&name[pos], "_%d", el);

      // print_array(OP_map_list[map->index]->aug_maps[el], len * map->dim, name, my_rank);

    }
  }
}

void step11_halo(int exec_levels, int **part_range, int **core_elems, int **exp_elems, int my_rank, int comm_size){
  // if OP_part_list is empty, (i.e. no previous partitioning done) then
  // create it and store the seperation of elements using core_elems
  // and exp_elems
  if (OP_part_index != OP_set_index) {
    // allocate memory for list
    OP_part_list = (part *)xmalloc(OP_set_index * sizeof(part));

    for (int s = 0; s < OP_set_index; s++) { // for each set
      op_set set = OP_set_list[s];
      // printf("set %s size = %d\n", set.name, set.size);
      int *g_index = (int *)xmalloc(sizeof(int) * set->size);
      int *partition = (int *)xmalloc(sizeof(int) * set->size);
      for (int i = 0; i < set->size; i++) {
        g_index[i] =
            get_global_index(i, my_rank, part_range[set->index], comm_size);
        partition[i] = my_rank;
      }
      decl_partition(set, g_index, partition);

      // combine core_elems and exp_elems to one memory block
      int *temp = (int *)xmalloc(sizeof(int) * set->size);
      memcpy(&temp[0], core_elems[set->index], set->core_sizes[0] * sizeof(int));
      memcpy(&temp[set->core_sizes[0]], exp_elems[set->index],
             (set->size - set->core_sizes[0]) * sizeof(int));

      // update OP_part_list[set->index]->g_index
      for (int i = 0; i < set->size; i++) {
        temp[i] = OP_part_list[set->index]->g_index[temp[i]];
      }
      op_free(OP_part_list[set->index]->g_index);
      OP_part_list[set->index]->g_index = temp;
    }
  } else { // OP_part_list exists (i.e. a partitioning has been done)
           // update the seperation of elements

    for (int s = 0; s < OP_set_index; s++) { // for each set
      op_set set = OP_set_list[s];

      // combine core_elems and exp_elems to one memory block
      int *temp = (int *)xmalloc(sizeof(int) * set->size);
      memcpy(&temp[0], core_elems[set->index], set->core_sizes[0] * sizeof(int));
      memcpy(&temp[set->core_sizes[0]], exp_elems[set->index],
             (set->size - set->core_sizes[0]) * sizeof(int));

      // update OP_part_list[set->index]->g_index
      for (int i = 0; i < set->size; i++) {
        temp[i] = OP_part_list[set->index]->g_index[temp[i]];
      }
      op_free(OP_part_list[set->index]->g_index);
      OP_part_list[set->index]->g_index = temp;
    }
  }

  /*for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];
    printf("Original Index for set %s\n", set->name);
    for(int i=0; i<set->size; i++ )
    printf(" %d",OP_part_list[set->index]->g_index[i]);
    }*/

  // set up exec and nonexec sizes
  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    int num_levels = set->halo_info->nhalos_count;
    int max_level = set->halo_info->max_nhalos;

    set->exec_sizes = (int*)xmalloc(sizeof(int) * num_levels);
    set->nonexec_sizes = (int*)xmalloc(sizeof(int) * num_levels);

    for(int el = 0; el < num_levels; el++){
      int exec_levels = set->halo_info->nhalos[el];
      set->exec_sizes[el] = 0;
      for(int l = 0; l < exec_levels; l++){
        set->exec_sizes[el] += OP_aug_import_exec_lists[l][set->index] ? 
        OP_aug_import_exec_lists[l][set->index]->size : 0;
      }
      set->nonexec_sizes[el] = OP_aug_import_nonexec_lists[el][set->index]->size;  // todo: we have duplicate elements in the on exec
      if(exec_levels == 1){
        set->exec_size = OP_aug_import_exec_lists[el][set->index]->size;
        set->nonexec_size = OP_aug_import_nonexec_lists[el][set->index]->size;
      }
    }
  }
}

void step10_halo_prev(int exec_levels, int **part_range, int **core_elems, int **exp_elems, int my_rank, int comm_size){
 
  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];

    int exec_size = 0;
    halo_list exec[exec_levels];
    for(int l = 0; l < exec_levels; l++){
      exec[l] = OP_aug_export_exec_lists[l][set->index];
      if(exec[l])
        exec_size += exec[l]->size;
    }

    halo_list nonexec = OP_export_nonexec_list[set->index];

    if (exec_size > 0) {
      exp_elems[set->index] = (int *)xmalloc(exec_size * sizeof(int));
      int prev_exec_size = 0;
      for(int l = 0; l < exec_levels; l++){
        if(exec[l]){
          memcpy(&exp_elems[set->index][prev_exec_size], exec[l]->list, exec[l]->size * sizeof(int));
          prev_exec_size += exec[l]->size;
        }
      }
      quickSort(exp_elems[set->index], 0, exec_size - 1);

      int num_exp = removeDups(exp_elems[set->index], exec_size);
      core_elems[set->index] = (int *)xmalloc(set->size * sizeof(int));
      int count = 0;
      for (int e = 0; e < set->size; e++) { // for each elment of this set

        if ((binary_search(exp_elems[set->index], e, 0, num_exp - 1) < 0)) {
          core_elems[set->index][count++] = e;
        }
      }
      quickSort(core_elems[set->index], 0, count - 1);

      if (count + num_exp != set->size)
        printf("sizes not equal\n");
      set->core_size = count;

      // for each data array defined on this set seperate its elements
      op_dat_entry *item;
      TAILQ_FOREACH(item, &OP_dat_list, entries) {
        op_dat dat = item->dat;

        if (compare_sets(set, dat->set) == 1) // if this data array is
        // defined on this set
        {
          char *new_dat = (char *)xmalloc(set->size * dat->size);
          for (int i = 0; i < count; i++) {
            memcpy(&new_dat[i * dat->size],
                   &dat->data[core_elems[set->index][i] * dat->size],
                   dat->size);
          }
          for (int i = 0; i < num_exp; i++) {
            memcpy(&new_dat[(count + i) * dat->size],
                   &dat->data[exp_elems[set->index][i] * dat->size], dat->size);
          }
          memcpy(&dat->data[0], &new_dat[0], set->size * dat->size);
          op_free(new_dat);
        }
      }

      // for each mapping defined from this set seperate its elements
      for (int m = 0; m < OP_map_index; m++) { // for each set
        op_map map = OP_map_list[m];

        if (compare_sets(map->from, set) == 1) { // if this mapping is
                                                 // defined from this set
          int *new_map = (int *)xmalloc(set->size * map->dim * sizeof(int));
          for (int i = 0; i < count; i++) {
            memcpy(&new_map[i * map->dim],
                   &map->map[core_elems[set->index][i] * map->dim],
                   map->dim * sizeof(int));
          }
          for (int i = 0; i < num_exp; i++) {
            memcpy(&new_map[(count + i) * map->dim],
                   &map->map[exp_elems[set->index][i] * map->dim],
                   map->dim * sizeof(int));
          }
          memcpy(&map->map[0], &new_map[0], set->size * map->dim * sizeof(int));
          op_free(new_map);
        }
      }

      
      for(int l = 0; l < exec_levels; l++){
        if(!exec[l])
          continue;
        for (int i = 0; i < exec[l]->size; i++) {
          int index =
              binary_search(exp_elems[set->index], exec[l]->list[i], 0, num_exp - 1);
          if (index < 0)
            printf("Problem in seperating core elements - exec list\n");
          else
            exec[l]->list[i] = count + index;
        }
      }

      for (int i = 0; i < nonexec->size; i++) {
        int index = binary_search(core_elems[set->index], nonexec->list[i], 0,
                                  count - 1);
        if (index < 0) {
          index = binary_search(exp_elems[set->index], nonexec->list[i], 0,
                                num_exp - 1);
          if (index < 0)
            printf("Problem in seperating core elements - nonexec list\n");
          else
            nonexec->list[i] = count + index;
        } else
          nonexec->list[i] = index;
      }

    } else {
      core_elems[set->index] = (int *)xmalloc(set->size * sizeof(int));
      exp_elems[set->index] = (int *)xmalloc(0 * sizeof(int));
      for (int e = 0; e < set->size; e++) { // for each elment of this set
        core_elems[set->index][e] = e;
      }
      set->core_size = set->size;
    }
  }

  // now need to renumber mapping tables as the elements are seperated
  for (int m = 0; m < OP_map_index; m++) { // for each set
    op_map map = OP_map_list[m];

    int imp_exec_size = 0;
    for(int l = 0; l < exec_levels; l++){
      imp_exec_size += OP_aug_import_exec_lists[l][map->from->index] ? 
      OP_aug_import_exec_lists[l][map->from->index]->size : 0;
    }
    // for each entry in this mapping table: original+execlist
    int len = map->from->size + imp_exec_size;
    for (int e = 0; e < len; e++) {
      for (int j = 0; j < map->dim; j++) { // for each element pointed
                                           // at by this entry
        if (map->map[e * map->dim + j] < map->to->size) {
          int index = binary_search(core_elems[map->to->index],
                                    map->map[e * map->dim + j], 0,
                                    map->to->core_size - 1);
          if (index < 0) {
            index = binary_search(exp_elems[map->to->index],
                                  map->map[e * map->dim + j], 0,
                                  (map->to->size) - (map->to->core_size) - 1);
            if (index < 0)
              printf("Problem in seperating core elements - \
                  renumbering map\n");
            else
              OP_map_list[map->index]->map[e * map->dim + j] =
                  map->to->core_size + index;
          } else
            OP_map_list[map->index]->map[e * map->dim + j] = index;
        }
      }
    }
  }
}

void step11_halo_prev(int exec_levels, int **part_range, int **core_elems, int **exp_elems, int my_rank, int comm_size){
  // if OP_part_list is empty, (i.e. no previous partitioning done) then
  // create it and store the seperation of elements using core_elems
  // and exp_elems
  if (OP_part_index != OP_set_index) {
    // allocate memory for list
    OP_part_list = (part *)xmalloc(OP_set_index * sizeof(part));

    for (int s = 0; s < OP_set_index; s++) { // for each set
      op_set set = OP_set_list[s];
      // printf("set %s size = %d\n", set.name, set.size);
      int *g_index = (int *)xmalloc(sizeof(int) * set->size);
      int *partition = (int *)xmalloc(sizeof(int) * set->size);
      for (int i = 0; i < set->size; i++) {
        g_index[i] =
            get_global_index(i, my_rank, part_range[set->index], comm_size);
        partition[i] = my_rank;
      }
      decl_partition(set, g_index, partition);

      // combine core_elems and exp_elems to one memory block
      int *temp = (int *)xmalloc(sizeof(int) * set->size);
      memcpy(&temp[0], core_elems[set->index], set->core_size * sizeof(int));
      memcpy(&temp[set->core_size], exp_elems[set->index],
             (set->size - set->core_size) * sizeof(int));

      // update OP_part_list[set->index]->g_index
      for (int i = 0; i < set->size; i++) {
        temp[i] = OP_part_list[set->index]->g_index[temp[i]];
      }
      op_free(OP_part_list[set->index]->g_index);
      OP_part_list[set->index]->g_index = temp;
    }
  } else { // OP_part_list exists (i.e. a partitioning has been done)
           // update the seperation of elements

    for (int s = 0; s < OP_set_index; s++) { // for each set
      op_set set = OP_set_list[s];

      // combine core_elems and exp_elems to one memory block
      int *temp = (int *)xmalloc(sizeof(int) * set->size);
      memcpy(&temp[0], core_elems[set->index], set->core_size * sizeof(int));
      memcpy(&temp[set->core_size], exp_elems[set->index],
             (set->size - set->core_size) * sizeof(int));

      // update OP_part_list[set->index]->g_index
      for (int i = 0; i < set->size; i++) {
        temp[i] = OP_part_list[set->index]->g_index[temp[i]];
      }
      op_free(OP_part_list[set->index]->g_index);
      OP_part_list[set->index]->g_index = temp;
    }
  }

  /*for(int s=0; s<OP_set_index; s++) { //for each set
    op_set set=OP_set_list[s];
    printf("Original Index for set %s\n", set->name);
    for(int i=0; i<set->size; i++ )
    printf(" %d",OP_part_list[set->index]->g_index[i]);
    }*/

  // set up exec and nonexec sizes
  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    for(int l = 0; l < exec_levels; l++){
      set->exec_size += OP_aug_import_exec_lists[l][set->index] ? 
      OP_aug_import_exec_lists[l][set->index]->size : 0;
    }
    set->nonexec_size = OP_import_nonexec_list[set->index]->size;
  }
}

void merge_exec_halos(int exec_levels, int my_rank, int comm_size){

  OP_merged_import_exec_list = (halo_list*) xmalloc(OP_set_index * sizeof(halo_list));
  OP_merged_export_exec_list = (halo_list*) xmalloc(OP_set_index * sizeof(halo_list));

  for(int s = 0; s < OP_set_index; s++){
    op_set set = OP_set_list[s];
    OP_merged_import_exec_list[set->index] = NULL;
    OP_merged_export_exec_list[set->index] = NULL;
  }

  halo_list* h_lists = (halo_list*) xmalloc(3 * sizeof(halo_list));

  for(int s = 0; s < OP_set_index; s++){
    op_set set = OP_set_list[s];
    for(int l = 0; l < exec_levels; l++){

      h_lists[0] = OP_merged_import_exec_list[set->index];
      h_lists[1] = OP_aug_import_exec_lists[l][set->index];
      OP_merged_import_exec_list[set->index] = merge_halo_lists(2, h_lists, my_rank, comm_size);

      h_lists[0] = OP_merged_export_exec_list[set->index];
      h_lists[1] = OP_aug_export_exec_lists[l][set->index];
      OP_merged_export_exec_list[set->index] = merge_halo_lists(2, h_lists, my_rank, comm_size);

      // OP_merged_import_exec_list[set->index] = merge_halo_lists(OP_merged_import_exec_list[set->index], 
      // OP_aug_import_exec_lists[l][set->index], 0, my_rank, comm_size);
      // OP_merged_export_exec_list[set->index] = merge_halo_lists(OP_merged_export_exec_list[set->index], 
      // OP_aug_export_exec_lists[l][set->index], 0, my_rank, comm_size);
    }
  }
  
}

// halo_list* subtract_h_list(halo_list* minuend_list, halo_list* subtrahend_list){

//   halo_list* diff_list = (halo_list *)xmalloc(OP_set_index * sizeof(halo_list));
//   for(int s = 0; s < OP_set_index; s++){
//     op_set set = OP_set_list[s];
//     halo_list min_list = minuend_list[set->index];
//     halo_list sub_list = subtrahend_list[set->index];

//     int* list = (int*)malloc(min_list->size * sizeof(int)); //max size
//     int* sizes = (int*)malloc(min_list->ranks_size * sizeof(int)); //max size
//     int* disps = (int*)malloc(min_list->ranks_size * sizeof(int)); //max size
//     int* ranks = (int*)malloc(min_list->ranks_size * sizeof(int)); //max size
    
//     int total_size = 0;
//     int index = 0;
//     for(int r = 0; r < min_list->ranks_size; r++){
//       int size1 = 0, size2 = 0;
//       int disp1 = 0, disp2 = 0;
//       int rank1 = min_list->ranks[r];
//       size1 = min_list->sizes[r];
//       disp1 = min_list->disps[r];

//       int found = binary_search(sub_list->ranks, rank1, 0,
//                                       sub_list->ranks_size - 1);
      
//       if(found >= 0){
//         size2 = sub_list->sizes[found];
//         disp2 = sub_list->disps[found];
//       }

//       int size = remove_elements_from_array(&list[total_size], &min_list->list[disp1], size1, &sub_list->list[disp2], size2);

//       if(size > 0){
//         ranks[index] = rank1;
//         sizes[index] = size;
//         disps[index] = total_size;

//         total_size +=  size;
//         index++;
//       }        
//     }
//     halo_list h_list = (halo_list)xmalloc(sizeof(halo_list_core));
//     h_list->set = min_list->set;
//     h_list->ranks_size = index;
//     h_list->list = list;
//     h_list->sizes = sizes;
//     h_list->ranks = ranks;
//     h_list->disps = disps;
//     h_list->size = total_size;

//     diff_list[set->index] = h_list;

//   }
//   return diff_list;
// }

// void create_dat_list(int exec_levels, int my_rank){
//   for(int i = 0; i < exec_levels; i++){
//     halo_list* temp_import_dat = subtract_h_list(OP_aug_import_exec_lists[i], OP_import_exec_list);
//     halo_list* temp_export_dat = subtract_h_list(OP_aug_export_exec_lists[i], OP_export_exec_list);

//     OP_aug_import_exec_dat_lists[i] = subtract_h_list(temp_import_dat, OP_import_nonexec_list);
//     OP_aug_export_exec_dat_lists[i] = subtract_h_list(temp_export_dat, OP_export_nonexec_list);

//     for(int s = 0; s < OP_set_index; s++){
//       op_free(temp_import_dat[s]);
//       op_free(temp_export_dat[s]);
//     }
//     op_free(temp_import_dat);
//     op_free(temp_export_dat);
//   }

//   halo_list* temp_nonexec_import_dat = subtract_h_list(OP_aug_import_nonexec_list, OP_import_exec_list);
//   halo_list* temp_nonexec_export_dat = subtract_h_list(OP_aug_export_nonexec_list, OP_export_exec_list);

//   OP_aug_import_nonexec_dat_list = subtract_h_list(temp_nonexec_import_dat, OP_import_nonexec_list);
//   OP_aug_export_nonexec_dat_list = subtract_h_list(temp_nonexec_export_dat, OP_export_nonexec_list);

//   for(int s = 0; s < OP_set_index; s++){
//       op_free(temp_nonexec_import_dat[s]);
//       op_free(temp_nonexec_export_dat[s]);
//     }
//     op_free(temp_nonexec_import_dat);
//     op_free(temp_nonexec_export_dat);
// }

void merge_nonexec_halos(int exec_levels, int my_rank, int comm_size){

  OP_merged_import_nonexec_list = (halo_list*) xmalloc(OP_set_index * sizeof(halo_list));
  OP_merged_export_nonexec_list = (halo_list*) xmalloc(OP_set_index * sizeof(halo_list));

  for(int s = 0; s < OP_set_index; s++){
    op_set set = OP_set_list[s];
    OP_merged_import_nonexec_list[set->index] = NULL;
    OP_merged_export_nonexec_list[set->index] = NULL;
  }

  for(int s = 0; s < OP_set_index; s++){
    op_set set = OP_set_list[s];
    int num_levels = set->halo_info->nhalos_count;
    halo_list* import_h_lists = (halo_list*) xmalloc(num_levels * sizeof(halo_list));
    halo_list* export_h_lists = (halo_list*) xmalloc(num_levels * sizeof(halo_list));
    for(int i = 0; i < num_levels; i++){
      import_h_lists[i] = OP_aug_import_nonexec_lists[i][set->index];
      export_h_lists[i] = OP_aug_export_nonexec_lists[i][set->index];
    }
    OP_merged_import_nonexec_list[set->index] = merge_halo_lists(num_levels, import_h_lists, my_rank, comm_size);
    OP_merged_export_nonexec_list[set->index] = merge_halo_lists(num_levels, export_h_lists, my_rank, comm_size);
  }
}

void set_dats_mgcfd(){

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    op_mpi_add_nhalos(set, 2);
    op_mpi_add_nhalos(set, 4);
  }
  return;

  op_dat_entry *item;
  int d = -1; // d is just simply the tag for mpi comms
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    if (strncmp("p_variables_L0", dat->name, strlen("p_variables_L0")) == 0|| 
        strncmp("p_fluxes_L0", dat->name, strlen("p_fluxes_L0")) == 0) { // if this data array
      // dat->loopchain_to_execlevels->set(0, 1);
      // dat->set->dat_to_execlevels->set(dat->index, 1);

      // dat->loopchain_to_execlevels->set(1, 2);
      // dat->set->dat_to_execlevels->set(dat->index, 2);

      // dat->loopchain_to_execlevels->set(2, 4);
      // dat->set->dat_to_execlevels->set(dat->index, 4);
      printf("set dat dat=%s\n", dat->name);

    }
    else{
      printf("set dat other dat=%s\n", dat->name);
      // dat->loopchain_to_execlevels->set(0, 1);
      // dat->set->dat_to_execlevels->set(dat->index, 1);

      // dat->loopchain_to_execlevels->set(1, 2);
      // dat->set->dat_to_execlevels->set(dat->index, 2);

      // dat->loopchain_to_execlevels->set(1, 4);
      // dat->set->dat_to_execlevels->set(dat->index, 4);
    }
  }
}

int get_max_halo_level(){

  int max_level = 1;
  for(int s = 0; s < OP_set_index; s++){
    op_set set = OP_set_list[s];
    int max = set->halo_info->max_nhalos;
    if(max > max_level){
      max_level = max;
    }
  }
  return max_level;

  // int max_level = 1;
  // op_dat_entry *item;
  // TAILQ_FOREACH(item, &OP_dat_list, entries) {
  //   op_dat dat = item->dat;
  //   int max = dat->loopchain_to_execlevels->get_max_val();
  //   if(max > max_level){
  //     max_level = max;
  //   }
  // }
  // return max_level;
}

int get_max_halo_level_count(){

  int max_count = 1;
  for(int s = 0; s < OP_set_index; s++){
    op_set set = OP_set_list[s];
    int max = set->halo_info->nhalos_count;
    if(max > max_count){
      max_count = max;
    }
  }
  return max_count;

  // int max_count = 1;
  // op_dat_entry *item;
  // TAILQ_FOREACH(item, &OP_dat_list, entries) {
  //   op_dat dat = item->dat;
  //   int max = dat->loopchain_to_execlevels->get_count();
  //   if(max > max_count){
  //     max_count = max;
  //   }
  // }
  // return max_count;
}
/*******************************************************************************
 * Main MPI halo creation routine
 *******************************************************************************/

void op_halo_create_comm_avoid() {
  // declare timers

  for(int i = 0; i < 10; i++){
    OP_aug_export_exec_lists[i] = NULL;
    OP_aug_import_exec_lists[i] = NULL;

    OP_aug_export_nonexec_lists[i] = NULL;
    OP_aug_import_nonexec_lists[i] = NULL;
  }
  set_dats_mgcfd();

  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  double time;
  double max_time;
  op_timers(&cpu_t1, &wall_t1); // timer start for list create

  // create new communicator for OP mpi operation
  int my_rank, comm_size;
  // MPI_Comm_dup(OP_MPI_WORLD, &OP_MPI_WORLD);
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_WORLD, &comm_size);

  /* Compute global partition range information for each set*/
  int **part_range = (int **)xmalloc(OP_set_index * sizeof(int *));
  get_part_range(part_range, my_rank, comm_size, OP_MPI_WORLD);

  // save this partition range information if it is not already saved during
  // a call to some partitioning routine
  if (orig_part_range == NULL) {
    orig_part_range = (int **)xmalloc(OP_set_index * sizeof(int *));
    for (int s = 0; s < OP_set_index; s++) {
      op_set set = OP_set_list[s];
      orig_part_range[set->index] = (int *)xmalloc(2 * comm_size * sizeof(int));
      for (int j = 0; j < comm_size; j++) {
        orig_part_range[set->index][2 * j] = part_range[set->index][2 * j];
        orig_part_range[set->index][2 * j + 1] =
            part_range[set->index][2 * j + 1];
      }
    }
  }

  aug_part_range = (int **)xmalloc(OP_set_index * sizeof(int *));
  aug_part_range_size = (int *)xmalloc(OP_set_index * sizeof(int));
  aug_part_range_cap = (int *)xmalloc(OP_set_index * sizeof(int));
  for (int s = 0; s < OP_set_index; s++) {
    aug_part_range_cap[s] = 1000;
    aug_part_range[s] = (int *)xmalloc(aug_part_range_cap[s] * sizeof(int));
    aug_part_range_size[s] = 0;
  }

  foreign_aug_part_range = (int ***)xmalloc(comm_size * sizeof(int **));
  foreign_aug_part_range_size = (int **)xmalloc(comm_size * sizeof(int *));
  for(int i = 0; i < comm_size; i++){
    foreign_aug_part_range[i] = (int **)xmalloc(OP_set_index * sizeof(int *));
    foreign_aug_part_range_size[i] = (int *)xmalloc(OP_set_index * sizeof(int));

    for(int j = 0; j < OP_set_index; j++){
      foreign_aug_part_range[i][j] = NULL;
      foreign_aug_part_range_size[i][j] = 0;
    }
  }

  int num_halos = get_max_halo_level();//2;
  printf("max_exec_levels=%d\n", num_halos);
  
  for(int l = 0; l < num_halos; l++){
    /*----- STEP 1 - Construct export lists for execute set elements and related
    mapping table entries -----*/
    OP_aug_export_exec_lists[l] = step1_create_aug_export_exec_list(l, part_range, my_rank, comm_size);

    /*---- STEP 2 - construct import lists for mappings and execute sets------*/
    OP_aug_import_exec_lists[l] = create_handshake_h_list(OP_aug_export_exec_lists[l], part_range, my_rank, comm_size);

    /*--STEP 3 -Exchange mapping table entries using the import/export lists--*/
    step3_exchange_exec_mappings(l, part_range, my_rank, comm_size);

    /*-STEP 6 - Exchange execute set elements/data using the import/export lists--*/
    step6_exchange_exec_data(l, part_range, my_rank, comm_size);

    create_aug_part_range(l, part_range, my_rank, comm_size);
    exchange_aug_part_ranges(l, part_range, my_rank, comm_size);
  }

  OP_export_exec_list = OP_aug_export_exec_lists[0];
  OP_import_exec_list = OP_aug_import_exec_lists[0];

  /*-- STEP 4 - Create import lists for non-execute set elements using mapping
    table entries including the additional mapping table entries --*/
  step4_import_nonexec(num_halos, part_range, my_rank, comm_size);

  /*----------- STEP 5 - construct non-execute set export lists -------------*/
  // step5(part_range, my_rank, comm_size); //todo: uncomment this

  // todo: this is going upto max exec levels. but for non exec we don't need non exec for all the levels.
  // hack: check null inside create_handshake_h_list
  for(int l = 0; l < get_max_halo_level_count(); l++){
    OP_aug_export_nonexec_lists[l] = create_handshake_h_list(OP_aug_import_nonexec_lists[l], part_range, my_rank, comm_size);
  }

  OP_import_nonexec_list =
      (halo_list *)xmalloc(OP_set_index * sizeof(halo_list));
  OP_export_nonexec_list =
      (halo_list *)xmalloc(OP_set_index * sizeof(halo_list));

  OP_export_nonexec_list = OP_aug_export_nonexec_lists[0];
  OP_import_nonexec_list = OP_aug_import_nonexec_lists[0];

  /*-STEP 6 - Exchange execute set elements/data using the import/export
   * lists--*/
  // this step is moved upwards

  /*-STEP 7 - Exchange non-execute set elements/data using the import/export
   * lists--*/
  step7_halo(num_halos, part_range, my_rank, comm_size);
  
  /*-STEP 8 ----------------- Renumber Mapping tables-----------------------*/
  step8_renumber_mappings(num_halos, part_range, my_rank, comm_size);

 
  /*-STEP 9 ---------------- Create MPI send Buffers-----------------------*/
  step9_halo(num_halos, part_range, my_rank, comm_size);
   

  /*-STEP 10 -------------------- Separate core
   * elements------------------------*/
  int **core_elems = (int **)xmalloc(OP_set_index * sizeof(int *));
  int **exp_elems = (int **)xmalloc(OP_set_index * sizeof(int *));
  step10_halo(num_halos, part_range, core_elems, exp_elems, my_rank, comm_size);

  /*-STEP 11 ----------- Save the original set element
   * indexes------------------*/
  step11_halo(num_halos, part_range, core_elems, exp_elems, my_rank, comm_size);

  

  // This is to facilitate halo exchange in one message for multiple extended exec layers
  merge_exec_halos(num_halos, my_rank, comm_size);
  merge_nonexec_halos(num_halos, my_rank, comm_size);


  /*-STEP 12 ---------- Clean up and Compute rough halo size
   * numbers------------*/
  for (int i = 0; i < OP_set_index; i++) {
    op_free(part_range[i]);
    op_free(core_elems[i]);
    op_free(exp_elems[i]);
  }
  op_free(part_range);
  op_free(exp_elems);
  op_free(core_elems);

  // releasing augmented part range data structures
  for (int s = 0; s < OP_set_index; s++) {
    op_free(aug_part_range[s]);
  }
  op_free(aug_part_range);
  op_free(aug_part_range_size);
  op_free(aug_part_range_cap);

  for(int i = 0; i < comm_size; i++){
    for(int j = 0; j < OP_set_index; j++){
      op_free(foreign_aug_part_range[i][j]);
    }
    op_free(foreign_aug_part_range[i]);
    op_free(foreign_aug_part_range_size[i]);
  }
  op_free(foreign_aug_part_range);
  op_free(foreign_aug_part_range_size);

  op_timers(&cpu_t2, &wall_t2); // timer stop for list create
  // compute import/export lists creation time
  time = wall_t2 - wall_t1;
  MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);

  step12(part_range, max_time, my_rank, comm_size);
}

void op_single_halo_destroy(halo_list* h_list){

  for (int s = 0; s < OP_set_index; s++) {
    op_set set = OP_set_list[s];

    op_free(h_list[set->index]->ranks);
    op_free(h_list[set->index]->disps);
    op_free(h_list[set->index]->sizes);
    op_free(h_list[set->index]->list);
    op_free(h_list[set->index]);
  }
  op_free(h_list);
}

void op_halo_destroy_comm_avoid() {

  op_halo_destroy();

  for(int i = 0; i < 10; i++){
    if(OP_aug_import_exec_lists[i])
      op_single_halo_destroy(OP_aug_import_exec_lists[i]);
    if(OP_aug_export_exec_lists[i])
      op_single_halo_destroy(OP_aug_import_exec_lists[i]);
  }

}

int find_element_in(int* arr, int element){
  for(int i = 1; i <= arr[0]; i++){
    if(arr[i] == element){
      return i;
    }
  }
  return -1;
}

void calculate_max_values(op_set from_set, op_set to_set, int map_dim, int* map_values,
int* to_sets, int* to_set_to_core_max, int* to_set_to_exec_max, int* to_set_to_nonexec_max, int my_rank){

  int set_index = find_element_in(to_sets, to_set->index);
  if(set_index < 0){
    to_sets[0] = to_sets[0] + 1;
    to_sets[to_sets[0]] = to_set->index;
    set_index = to_sets[0];
  }
  
  int core_max = get_max_value(map_values, 0, from_set->core_size * map_dim);
  if(to_set_to_core_max[set_index] < core_max){
    to_set_to_core_max[set_index] = core_max;
  }

  int exec_max = get_max_value(map_values, from_set->core_size * map_dim,
      (from_set->size +  from_set->exec_size) * map_dim);
  if(to_set_to_exec_max[set_index] < exec_max){
    to_set_to_exec_max[set_index] = exec_max;
  }

  int nonexec_max = get_max_value(map_values, (from_set->size +  from_set->exec_size) * map_dim,
      (from_set->size +  from_set->exec_size + from_set->nonexec_size) * map_dim);
  if(to_set_to_nonexec_max[set_index] < nonexec_max){
    to_set_to_nonexec_max[set_index] = nonexec_max;
  }
  
}

int get_core_size(op_set set, int* to_sets, int* to_set_to_core_max){

  int index = find_element_in(to_sets, set->index);
  if(index != -1){
    return to_set_to_core_max[index] + 1;
  }else{
    return set->core_size;
  }
}

int get_exec_size(op_set set, int* to_sets, int* to_set_to_core_max, int* to_set_to_exec_max){

  int index = find_element_in(to_sets, set->index);
  if(index != -1){
    int it_core = to_set_to_core_max[index];
    int it_exec = to_set_to_exec_max[index];
    return ((it_exec - it_core) > 0) ? (it_exec - it_core) : 0;
  }else{
    return (set->size - set->core_size) + OP_import_exec_list[set->index]->size;
  }
}

int get_nonexec_size(op_set set, int* to_sets, int* to_set_to_exec_max, int* to_set_to_nonexec_max){

  int index = find_element_in(to_sets, set->index);
  if(index != -1){
    int it_exec = to_set_to_exec_max[index];
    int it_nonexec = to_set_to_nonexec_max[index];
    return ((it_nonexec - it_exec) > 0) ? (it_nonexec - it_exec) : 0;
  }else{
    return OP_import_nonexec_list[set->index]->size;
  }
}

int get_size_of_exec_level(op_set set, int level){
  return OP_aug_import_exec_lists[level][set->index]->size;
}