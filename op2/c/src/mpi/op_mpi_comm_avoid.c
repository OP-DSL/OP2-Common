
// mpi header
#include <mpi.h>

#include <op_lib_c.h>
#include <op_lib_mpi.h>
#include <op_util.h>

#include <op_mpi_core.h>
#include <limits.h>
#include <stdlib.h>

#ifdef COMM_AVOID_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <op_cuda_rt_support.h>
#endif


// Store augmented partition ranges
int** aug_part_range;
int* aug_part_range_size;
int* aug_part_range_cap;

int*** foreign_aug_part_range;
int** foreign_aug_part_range_size;

// Matrix to store elements and the ranks sent
int*** elem_rank_matrix;

// halo lists to store augmented halo information
halo_list* OP_aug_export_exec_lists[OP_NHALOS_MAX];
halo_list* OP_aug_import_exec_lists[OP_NHALOS_MAX];

halo_list* OP_aug_export_nonexec_lists[OP_NHALOS_MAX];
halo_list* OP_aug_import_nonexec_lists[OP_NHALOS_MAX];

// halo lists to store merged halo lists
halo_list *OP_merged_import_exec_list;
halo_list *OP_merged_export_exec_list;

halo_list *OP_merged_import_nonexec_list;
halo_list *OP_merged_export_nonexec_list;

halo_list *OP_merged_import_exec_nonexec_list;
halo_list *OP_merged_export_exec_nonexec_list;

// Group message buffers
MPI_Request *grp_send_requests;
MPI_Request *grp_recv_requests;

char *grp_send_buffer = NULL;
char *grp_recv_buffer = NULL;

char *grp_send_buffer_h = NULL;
char *grp_recv_buffer_h = NULL;

char *grp_send_buffer_d = NULL;
char *grp_recv_buffer_d = NULL;

int ca_grp_size_send_old = 0;
int ca_grp_size_recv_old = 0;

int grp_tag;

// Variables to store time measurements when debugging
double cpu_start, cpu_stop, cpu_tick, wall_start, wall_stop, wall_tick, duration;

// Default halo layer count is 1
#define DEFAULT_HALO_COUNT 1

/*******************************************************************************
 * Functions for debugging - start
 *******************************************************************************/

void start_time(int my_rank){
  // if(my_rank != 0)
  //   return;
  op_timers(&cpu_start, &wall_start);
  duration = 0;
  printf("start_time new my_rank=%d time=%f\n", my_rank, duration);
}

void tick_time(){
  op_timers(&cpu_tick, &wall_tick);
  duration += wall_tick - wall_start;
  wall_start = wall_tick;
}

void stop_time(int my_rank, const char* name){
  // if(my_rank != 0)
  //   return;
  op_timers(&cpu_stop, &wall_stop);
  duration += wall_stop - wall_start;
  printf("stop_time my_rank=%d %s time=%f\n", my_rank, name, duration);
}

void print_map(int my_rank, int map_index){
  printf("pmap my_rank=%d >>>>>>>>>>>>>>>start>>>>>>>\n", my_rank);
  for (int m = 0; m < OP_map_index; m++) { 
    op_map map = OP_map_list[m];
    int level = map->halo_info->max_nhalos;

    if (strncmp("ne", map->name, strlen("ne")) != 0){ // && strncmp("npe", map->name, strlen("npe")) != 0) {
      continue;
    }
    printf("pmap >>>>>>> start my_rank=%d map=%s map_index=%d dim=%d from=%s(s=%d c=%d e=%d n=%d) to=%s(s=%d c=%d e=%d n=%d)\n", 
        my_rank, map->name, map_index, map->dim, 
        map->from->name, map->from->size, map->from->core_sizes[map_index], map->from->exec_sizes[map_index], map->from->nonexec_sizes[map_index], 
        map->to->name, map->to->size, map->to->core_sizes[map_index], map->to->exec_sizes[map_index], map->to->nonexec_sizes[map_index]);

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
        printf("pmap my_rank=%d map=%s map[%d][%d]=%d\n", 
        my_rank, map->name, i, j, map->map[i * map->dim + j]);
      } 
    }

    printf("pmap <<<<<<<< end my_rank=%d map=%s dim=%d from=%s(s=%d c=%d e=%d n=%d) to=%s(s=%d c=%d e=%d n=%d)\n", 
        my_rank, map->name, map->dim, 
        map->from->name, map->from->size, map->from->core_size, map->from->exec_size, map->from->nonexec_size, 
        map->to->name, map->to->size, map->to->core_size, map->to->exec_size, map->to->nonexec_size);

  }
  printf("pmap my_rank=%d >>>>>>>>>>>>>>>end>>>>>>>\n", my_rank);  
}

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

  for(int l = 0; l < h_list->num_levels; l++){
    int level_disp = h_list->disps_by_level[l];
    int rank_disp = h_list->ranks_disps_by_level[l];
    

    for (int r = 0; r < h_list->ranks_size / h_list->num_levels; r++){
      int disp_in_level = h_list->disps[rank_disp + r];
      for (int i = 0; i < h_list->sizes[h_list->ranks_disps_by_level[l] + r]; i++){
        printf("print_halo my_rank=%d, name=%s, to_rank=%d, size=%d, disp=%d value[%d]=%d\n", 
        my_rank, name, h_list->ranks[r], h_list->sizes[h_list->ranks_disps_by_level[l] + r], 
        level_disp + disp_in_level, i, h_list->list[level_disp + disp_in_level + i]);
      }
    }
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

  for(int s = 0; s < OP_set_index; s++){
    for(int r = 0; r < comm_size; r++){
      for(int i = 0; i < aug_size[r][s]; i++){
        printf("foreign_aug_range my_rank=%d foreign=%d set=%s size=%d value[%d]=%d\n", my_rank, r, OP_set_list[s]->name, aug_size[s][r], i, aug_range[s][r][i]);
      }
    }
  }
  printf("foreign_aug_range end my_rank=%d\n", my_rank);
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

/*******************************************************************************
 * Functions for debugging - end
 *******************************************************************************/

/*******************************************************************************
 * Utility functions - start
 *******************************************************************************/

// Allocate memory and initialize element rank matrix
void create_elem_rank_matrix(int my_rank, int comm_size){
  elem_rank_matrix = (int***)xmalloc(OP_set_index * sizeof(int**));

  for(int i = 0; i < OP_set_index; i++){
    op_set set = OP_set_list[i];
    elem_rank_matrix[i] = (int **)xmalloc(set->size * sizeof(int *));
    for(int j = 0; j < set->size; j++){
      elem_rank_matrix[i][j] = (int *)xmalloc(comm_size * sizeof(int));
      for(int k = 0; k < comm_size; k++){
        elem_rank_matrix[i][j][k] = -1;
      }
    }
  }
}

// Free memory of element rank matrix
void free_elem_rank_matrix(int my_rank, int comm_size){
  for(int i = 0; i < OP_set_index; i++){
    op_set set = OP_set_list[i];
    for(int j = 0; j < set->size; j++){
      op_free(elem_rank_matrix[i][j]);
    }
    op_free(elem_rank_matrix[i]);
  }
}

// Allocate memory and initialize augmented partition range arrays
void create_part_range_arrays(int my_rank, int comm_size){

  aug_part_range = (int **)xmalloc(OP_set_index * sizeof(int *));
  aug_part_range_size = (int *)xmalloc(OP_set_index * sizeof(int));
  aug_part_range_cap = (int *)xmalloc(OP_set_index * sizeof(int));
  for (int s = 0; s < OP_set_index; s++) {
    aug_part_range_cap[s] = 1000;
    aug_part_range[s] = (int *)xmalloc(aug_part_range_cap[s] * sizeof(int));
    aug_part_range_size[s] = 0;
  }

  foreign_aug_part_range = (int ***)xmalloc(OP_set_index * sizeof(int **));
  foreign_aug_part_range_size = (int **)xmalloc(OP_set_index * sizeof(int *));
  for(int i = 0; i < OP_set_index; i++){
    foreign_aug_part_range[i] = (int **)xmalloc(comm_size * sizeof(int *));
    foreign_aug_part_range_size[i] = (int *)xmalloc(comm_size * sizeof(int));

    for(int j = 0; j < comm_size; j++){
      foreign_aug_part_range[i][j] = NULL;
      foreign_aug_part_range_size[i][j] = 0;
    }
  }
}

// Free memory of augmented partition range arrays
void free_part_range_arrays(int my_rank, int comm_size){
  for (int s = 0; s < OP_set_index; s++) {
    op_free(aug_part_range[s]);
  }
  op_free(aug_part_range);
  op_free(aug_part_range_size);
  op_free(aug_part_range_cap);

  for(int i = 0; i < OP_set_index; i++){
    for(int j = 0; j < comm_size; j++){
      op_free(foreign_aug_part_range[i][j]);
    }
    op_free(foreign_aug_part_range[i]);
    op_free(foreign_aug_part_range_size[i]);
  }
  op_free(foreign_aug_part_range);
  op_free(foreign_aug_part_range_size);
}


void update_elem_rank_matrix(op_set set, int elem, int rank, int comm_size){
  elem_rank_matrix[set->index][elem][rank] = 1;
}

bool is_elem_sent(op_set set, int elem, int rank, int comm_size){
  return elem_rank_matrix[set->index][elem][rank] == 1;
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


// Check whether the element is in the augmented parition range
void check_augmented_part_range(int* parts, int set_index, int value, int my_rank, int comm_size){
  for(int r = 0; r < comm_size; r++){
    parts[r] = -1;
    if(r != my_rank){
      int index = binary_search(foreign_aug_part_range[set_index][r], value, 0, foreign_aug_part_range_size[set_index][r] - 1);
      if(index >= 0){
        parts[r] = 1;
      }

      // for(int i = 0; i < foreign_aug_part_range_size[r][set_index]; i++){
      //   if(value == foreign_aug_part_range[r][set_index][i]){
      //     parts[r] = 1;
      //     break;
      //   }
      // }
    }
  }
}

//  Merge any given number of halo lists in an array to a single halo list
halo_list merge_halo_lists(int count, halo_list* h_lists, int my_rank, int comm_size){
  // printf("my_rank=%d merge_halo_lists start >>>>>>>>>>\n", my_rank);
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
    init_ranks_size += (h_lists[i]) ? h_lists[i]->ranks_size : 0;

    // if(h_lists[i])
    //   print_halo(h_lists[i], "halonew", my_rank);
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
  int *sizes = (int *)xmalloc(new_ranks_size * sizeof(int));
  int *disps = (int *)xmalloc(new_ranks_size * sizeof(int));
  
  int *level_sizes = (int *)xmalloc(new_ranks_size * sizeof(int));
  int *level_disps = (int *)xmalloc(new_ranks_size * sizeof(int));
  
  int* ranks_sizes_by_level = (int *)xmalloc(num_levels * sizeof(int));
  int* disps_by_level = (int *)xmalloc(num_levels * sizeof(int));
  int* ranks_disps_by_level = (int *)xmalloc(num_levels * sizeof(int));

  int* sizes_by_rank = (int *)xmalloc(tmp_rank_size * sizeof(int));
  int* disps_by_rank = (int *)xmalloc(tmp_rank_size * sizeof(int));

  int* sizes_upto_level_by_rank = (int *)xmalloc(num_levels * num_levels * tmp_rank_size * sizeof(int));
  int* disps_upto_level_by_rank = (int *)xmalloc(num_levels * num_levels* tmp_rank_size * sizeof(int));

  for(int i = 0; i < tmp_rank_size; i++){
    sizes_by_rank[i] = 0;
    disps_by_rank[i] = 0;
  }

  for(int i = 0; i < num_levels * num_levels * tmp_rank_size; i++){
    sizes_upto_level_by_rank[i] = 0;
    disps_upto_level_by_rank[i] = 0;
  }

  for(int i = 0; i < num_levels * tmp_rank_size; i++){
    level_sizes[i] = 0;
    level_disps[i] = 0;
  }

  if(num_levels > 0){
    ranks_disps_by_level[0] = 0;
    disps_by_level[0] = 0;
  }

  int list_start = 0;
  int rank_start = 0;
  int level_start = 0;
  for(int i = 0; i < count; i++){
    halo_list h_list = h_lists[i];
    if(h_list == NULL)
      continue;

    // print_array(h_list->sizes, h_list->ranks_size, "main_sizes", my_rank);
    // print_array(h_list->disps, h_list->ranks_size, "main_disps", my_rank);
    // print_array(h_list->ranks, h_list->ranks_size, "main_ranks", my_rank);

    memcpy(&(list[list_start]), h_list->list, h_list->size * sizeof(int));

    for(int l = 0; l < h_list->num_levels; l++){
      for(int r = 0; r < tmp_rank_size; r++){
        int rank_index = binary_search(&h_list->ranks[h_list->ranks_disps_by_level[l]], temp_ranks[r], 0, h_list->ranks_sizes_by_level[l] - 1);
        ranks[rank_start + r] = temp_ranks[r];
        int prev_level_size = 0;
        if(rank_index >= 0){
          sizes[rank_start + r] = h_list->sizes[h_list->ranks_disps_by_level[l] + rank_index];
          sizes_by_rank[r] += h_list->sizes[h_list->ranks_disps_by_level[l] + rank_index];
          level_sizes[num_levels * r + level_start + l] = h_list->sizes[h_list->ranks_disps_by_level[l] + rank_index];
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

      ranks_sizes_by_level[level_start + l] = tmp_rank_size;
      disps_by_level[level_start + l] = list_start + h_list->disps_by_level[l];

      if(level_start + l > 0)
        ranks_disps_by_level[level_start + l] = ranks_sizes_by_level[level_start + l - 1] + ranks_disps_by_level[level_start + l - 1];

      rank_start += tmp_rank_size; //h_list->ranks_size;
    }

    list_start += h_list->size;
    level_start += h_list->num_levels;
  }

  for(int r = 1; r < tmp_rank_size; r++){
    disps_by_rank[r] = sizes_by_rank[r - 1] + disps_by_rank[r - 1];
  }

  for(int r = 0; r < tmp_rank_size; r++){
    for(int l = 0; l < num_levels; l++){
      for(int l1 = 0; l1 < l + 1; l1++){

        int prev_level_size = 0;
        if(l > 0){
          prev_level_size = sizes_upto_level_by_rank[r * num_levels * num_levels + (l - 1) * num_levels + l1];
        }

        int prev_rank_size = 0;
        int prev_rank_prev_level_size = 0;
         if(r > 0){
          prev_rank_size = sizes_upto_level_by_rank[(r - 1) * num_levels * num_levels + l * num_levels + l1];


          prev_rank_prev_level_size = 0;
          if(l > 0){
            prev_rank_prev_level_size = sizes_upto_level_by_rank[(r - 1) * num_levels * num_levels + (l - 1) * num_levels + l1];
          }
        }
        int size = 0;
        if(l1 == l){
          size = level_sizes[r * num_levels + l1];
        }
        sizes_upto_level_by_rank[r * num_levels * num_levels + l * num_levels + l1] = size + prev_level_size +
        prev_rank_size - prev_rank_prev_level_size;
      }
      for(int l1 = l + 1; l1 < num_levels; l1++){
        sizes_upto_level_by_rank[r * num_levels * num_levels + l * num_levels + l1] = sizes_upto_level_by_rank[r * num_levels * num_levels + l * num_levels + l1 - 1];
      }
    }
  }


  for(int r = 0; r < tmp_rank_size; r++){
    for(int l = 1; l < num_levels; l++){
      level_disps[r * num_levels + l] = level_sizes[r * num_levels + (l - 1)] + 
                                                      level_disps[r * num_levels + (l - 1)];
    }
  }

  for(int r = 0; r < tmp_rank_size; r++){
    for(int l = 0; l < num_levels; l++){

      for(int l1 = 0; l1 < l + 1; l1++){
        int size = 0;
        if(l1 <= l){
          size = level_sizes[r * num_levels + l1];
        }
        disps_upto_level_by_rank[r * num_levels * num_levels + l * num_levels + l1] = 
        sizes_upto_level_by_rank[r * num_levels * num_levels + l * num_levels + l1] - size;
      }

      for(int l1 = l + 1; l1 < num_levels; l1++){
        disps_upto_level_by_rank[r * num_levels * num_levels + l * num_levels + l1] = disps_upto_level_by_rank[r * num_levels * num_levels + l * num_levels + l1 - 1];
      }
    }
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
  h_list->level_sizes = level_sizes;
  h_list->ranks_sizes_by_level = ranks_sizes_by_level;
  h_list->ranks_disps_by_level = ranks_disps_by_level;
  h_list->disps_by_level = disps_by_level;

  // h_list->level_disps = level_disps;
  // h_list->sizes_upto_level_by_rank = sizes_upto_level_by_rank;
  // h_list->disps_upto_level_by_rank = disps_upto_level_by_rank;
  // h_list->sizes_by_rank = sizes_by_rank;
  // h_list->disps_by_rank = disps_by_rank;

// free unused arrays
  op_free(level_disps);
  op_free(sizes_upto_level_by_rank);
  op_free(disps_upto_level_by_rank);
  op_free(sizes_by_rank);
  op_free(disps_by_rank);


  // This is to free memory, if the halo lists which are merged are no longer used in the code, 
  // but only the merged one.

  // for(int i = 0; i < count; i++){
  //   if(h_lists[i] == NULL || i == 0 || i == 1){
  //     printf("hlist null d=%d\n", i);
  //     continue;
  //   }
  //   printf("hlist not null d=%d\n", i);
  //   op_single_halo_destroy(h_lists[i]);
  //   h_lists[i] = NULL;
    // op_halos_destroy_1(h_lists[i]);

    // op_free(h_lists[i]->ranks); h_lists[i]->ranks = NULL;
    // op_free(h_lists[i]->list); h_lists[i]->list = NULL;
    // op_free(h_lists[i]->disps); h_lists[i]->disps = NULL;
    // op_free(h_lists[i]->sizes); h_lists[i]->sizes = NULL;

    // if(h_lists[i]->ranks_sizes_by_level)
    //   op_free(h_lists[i]->ranks_sizes_by_level);
    // if(h_lists[i]->ranks_disps_by_level)
    //   op_free(h_lists[i]->ranks_disps_by_level);
    // if(h_lists[i]->disps_by_level)
    //   op_free(h_lists[i]->disps_by_level);
      
    // op_free(h_lists[i]->ranks);
    // op_free(h_lists[i]->disps);
    // op_free(h_lists[i]->sizes);
    // op_free(h_lists[i]->list);
    // op_free(h_lists[i]->ranks_sizes_by_level);
    // op_free(h_lists[i]->ranks_disps_by_level);
    // op_free(h_lists[i]->disps_by_level);
    // op_free(h_lists[i]->sizes_upto_level_by_rank);
    // op_free(h_lists[i]->disps_by_rank);
    // op_free(h_lists[i]->sizes_by_rank);
    // op_free(h_lists[i]);
  // }

  // Debug by printing arrays
  // print_halo(h_list, "mergedhalo", my_rank);
  // print_array(h_list->sizes, num_levels * tmp_rank_size, "sizes", my_rank);
  // print_array(h_list->disps, num_levels * tmp_rank_size, "disps", my_rank);
  // print_array(h_list->ranks, num_levels * tmp_rank_size, "ranks", my_rank);
  // print_array(h_list->level_sizes, num_levels * tmp_rank_size, "level_sizes", my_rank);
  // print_array(h_list->level_disps, num_levels * tmp_rank_size, "level_disps", my_rank);
  // print_array(h_list->ranks_sizes_by_level, num_levels, "ranks_sizes_by_level", my_rank);
  // print_array(h_list->ranks_disps_by_level, num_levels, "ranks_disps_by_level", my_rank);
  // print_array(h_list->disps_by_level, num_levels, "disps_by_level", my_rank);
  // print_array(h_list->sizes_upto_level_by_rank, num_levels * num_levels * tmp_rank_size, "sizes_upto_level_by_rank", my_rank);
  // print_array(h_list->disps_upto_level_by_rank, num_levels * num_levels * tmp_rank_size, "disps_upto_level_by_rank", my_rank);
  // print_array(h_list->disps_upto_level_by_rank, num_levels * tmp_rank_size, "disps_upto_level_by_rank", my_rank);

  // printf("my_rank=%d merge_halo_lists end <<<<<<<<<<<<<<\n", my_rank);
  return h_list;
}

// Given a halo list (export or import), create the counter part halo list (import or export)
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
    if(!list){
      handshake_h_list[set->index] = NULL;
      continue;
    }
      
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

// Create augmented partition range considering the exchanged elements for the additional halo layers.
void create_aug_part_range(int halo_id, int **part_range, int my_rank, int comm_size){
  
  int exec_size = 0;

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];

    if(is_halo_required_for_set(set, halo_id) != 1){
      continue;
    }

    exec_size = 0;
    exec_size = set->exec_sizes[halo_id];

    int start = 0;
    if(halo_id > 0){
      start = set->size + exec_size - OP_aug_import_exec_lists[halo_id][set->index]->size;
    }
    int end = set->size + exec_size;
   
    for (int e = start; e < end; e++) {      // for each elment of this set
      for (int m = 0; m < OP_map_index; m++) { // for each maping table
        op_map map = OP_map_list[m];

        if (compare_sets(map->from, set) == 1 && is_halo_required_for_map(map, halo_id) == 1) { // need to select mappings from this set
        
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
  // print_aug_part_list(aug_part_range, aug_part_range_size, my_rank);
}

// Exchange the augmented partition range with other processes
void exchange_aug_part_ranges(int halo_id, int **part_range, int my_rank, int comm_size){

  for (int s = 0; s < OP_set_index; s++) {

    if(aug_part_range_size[s] > 0){
      quickSort(aug_part_range[s], 0, aug_part_range_size[s] - 1);
      aug_part_range_size[s] = removeDups(aug_part_range[s], aug_part_range_size[s]);
    }
    aug_part_range[s] = (int *)xrealloc(aug_part_range[s], aug_part_range_size[s] * sizeof(int));
    aug_part_range_cap[s] = aug_part_range_size[s];

    // do an allgather to findout how many elements that each process will
    // be requesting partition information about
    MPI_Allgather(&aug_part_range_size[s], 1, MPI_INT, foreign_aug_part_range_size[s], 1, MPI_INT, OP_MPI_WORLD);

    // discover global size of these required elements
    int g_count = 0;
    for (int i = 0; i < comm_size; i++)
      g_count += foreign_aug_part_range_size[s][i];

    // prepare for an allgatherv
    int disp = 0;
    int *displs = (int *)xmalloc(comm_size * sizeof(int));
    for (int i = 0; i < comm_size; i++) {
      displs[i] = disp;
      disp = disp + foreign_aug_part_range_size[s][i];
    }

    // allocate memory to hold the global part ranges of elements requiring
    // partition details
    int *g_part_range = (int *)xmalloc(sizeof(int) * g_count);

    MPI_Allgatherv(aug_part_range[s], aug_part_range_size[s], MPI_INT, g_part_range, foreign_aug_part_range_size[s], displs,
                   MPI_INT, OP_MPI_WORLD);
    // op_free(aug_part_range[s]);
    aug_part_range_size[s] = 0;

    for (int i = 0; i < comm_size; i++) {
      foreign_aug_part_range[s][i] = (int *)xmalloc(sizeof(int) * foreign_aug_part_range_size[s][i]);
      memcpy(foreign_aug_part_range[s][i], &g_part_range[displs[i]], foreign_aug_part_range_size[s][i] * sizeof(int));

      if(foreign_aug_part_range_size[s][i] > 0){
        quickSort(foreign_aug_part_range[s][i], 0, foreign_aug_part_range_size[s][i] - 1);
      }
    }
    op_free(g_part_range);
  }
  // print_foreign_aug_part_list(foreign_aug_part_range, foreign_aug_part_range_size, comm_size, my_rank);
}

// Check whether an element is exported before
// This is used instead of element rank matrix due to memory concerns
bool is_in_prev_export_exec_halos(int halo_id, int set_index, int export_rank, int export_value, int my_rank){

  for(int i = 0; i < halo_id; i++){
    halo_list h_list = OP_aug_export_exec_lists[i][set_index];

    int rank_index = binary_search(h_list->ranks, export_rank, 0, h_list->ranks_size - 1);

    if(rank_index >= 0){
      int elem_index = binary_search(h_list->list, export_value, h_list->disps[rank_index], h_list->disps[rank_index] + h_list->sizes[rank_index] - 1);
      if(elem_index >= 0){
        return true;
      }
    }
  }
  return false;
}

// Allocate memory for maps which hold mappings for additional halo layers and initialize them
void prepare_aug_maps(){

  for (int m = 0; m < OP_map_index; m++) { 
    op_map map = OP_map_list[m];
    int max_level = map->halo_info->max_calc_nhalos;
    map->aug_maps = (int **)malloc((size_t)max_level * sizeof(int *));

    OP_aug_map_ptr_list[m] =
        (int **)malloc(max_level * sizeof(int *));

    int exec_size = 0;
    for(int l = 0; l < max_level; l++){
      exec_size += OP_aug_import_exec_lists[l][map->from->index]->size;
    }
    int nonexec_size = 0;
    for(int l = 0; l < max_level - 1; l++){ // last non exec level is not included. non exec mappings are not included in maps
      nonexec_size += OP_aug_import_nonexec_lists[l][map->from->index]->size;
    }
     
    int total_map_size = map->from->size + exec_size + nonexec_size;

    map->map = (int *)xrealloc(map->map, (total_map_size) * (size_t)map->dim * sizeof(int));        
    map->map_org = (int *)malloc((size_t)(total_map_size) * (size_t)map->dim * sizeof(int));
    memcpy(map->map_org, map->map, (size_t)(total_map_size) * (size_t)map->dim * sizeof(int));

    for(int el = 0; el < max_level; el++){
      if(is_halo_required_for_map(map, el) == 1 && is_map_required_for_calc(map, el) == 1){
        int *map_vals = (int *)malloc((size_t)(total_map_size) * (size_t)map->dim * sizeof(int));
        for(int i = 0; i < total_map_size; i++){
          map_vals[i] = -1;
        }
        if (map_vals == NULL) {
          printf(" op_decl_map_core error -- error allocating memory to map\n");
          exit(-1);
        }
        map->aug_maps[el] = map_vals;
        OP_aug_map_ptr_list[m][el] = map_vals;
      }else{
        map->aug_maps[el] = NULL;
        OP_aug_map_ptr_list[m][el] = NULL;
      }
    }
  }  
}

// Free temp original map used to create the augmented maps
void free_tmp_maps(){
  for (int m = 0; m < OP_map_index; m++) {
    op_map map = OP_map_list[m];
    op_free(map->map_org);
    map->map_org = NULL;
  }  
}

// Allocate memory for the arrays in op_set which hold additional halo level sizes/info
void prepare_aug_sets(){
  for(int s = 0; s < OP_set_index; s++){
    op_set set = OP_set_list[s];
    int max_level = set->halo_info->max_nhalos;

    set->core_sizes =  (int *) malloc(max_level * sizeof(int));
    set->exec_sizes = (int*)xmalloc(sizeof(int) * max_level);
    set->nonexec_sizes = (int*)xmalloc(sizeof(int) * max_level);
    set->exp_exec_sizes = (int*)xmalloc(sizeof(int) * max_level);
    set->exp_nonexec_sizes = (int*)xmalloc(sizeof(int) * max_level);
    set->total_nonexec_size = 0;
    set->total_exec_size = 0;

    for(int i = 0; i < max_level; i++){
      set->core_sizes[i] = 0;
      set->exec_sizes[i] = 0;
      set->nonexec_sizes[i] = 0;
      set->exp_exec_sizes[i] = 0;
      set->exp_nonexec_sizes[i] = 0;
    }

    op_dat_entry *item;
    int d = -1; // d is just simply the tag for mpi comms
    TAILQ_FOREACH(item, &OP_dat_list, entries) {
      op_dat dat = item->dat;
      if (compare_sets(set, dat->set) == 1) {
        dat->exec_dirtybits = (int *)xrealloc(dat->exec_dirtybits, max_level * sizeof(int));
        dat->nonexec_dirtybits = (int *)xrealloc(dat->nonexec_dirtybits, max_level * sizeof(int));

        for(int i = 0; i < max_level; i++){
          dat->exec_dirtybits[i] = dat->dirtybit;
          if(is_halo_required_for_set(dat->set, i) == 1){
            dat->nonexec_dirtybits[i] = dat->dirtybit;
          }else{
            dat->nonexec_dirtybits[i] = -1;
          }
        }
      }
    }
  }
}


// Remove set of elements from an array
int remove_elements_from_array(op_set set, int my_rank, int start, int* ret_arr, int* arr, int size, int* elememts, int element_size){

  int index = 0;
  for(int i = 0; i < size; i++){
    if(element_size > 0){
      int found = binary_search(elememts, arr[i], 0, element_size - 1);
      if(found < 0){
        ret_arr[index++] = arr[i];
      }
    }else{
      ret_arr[index++] = arr[i];
    }
  }
  if(size - element_size != index){
    printf("remove_elements_from_array ERROR not a subset size=%d elem=%d index=%d\n", size, element_size, index);
    exit(0);
  }
  return index;
}


int get_max_nhalos(){

  int max_level = 1;
  for(int s = 0; s < OP_set_index; s++){
    op_set set = OP_set_list[s];
    int max = set->halo_info->max_nhalos;
    if(max > max_level){
      max_level = max;
    }
    // printf("get_max_nhalos set=%s maxhalos=%d max=%d\n", set->name, set->halo_info->max_nhalos, max_level);
  }
  return max_level;
}

int get_max_nhalos_count(){

  int max_count = 1;
  for(int s = 0; s < OP_set_index; s++){
    op_set set = OP_set_list[s];
    int max = set->halo_info->nhalos_count;
    if(max > max_count){
      max_count = max;
    }
  }
  return max_count;
}

int get_max_calc_nhalos(){

  int max_level = 1;
  for(int s = 0; s < OP_set_index; s++){
    op_set set = OP_set_list[s];
    int max = set->halo_info->max_calc_nhalos;
    if(max > max_level){
      max_level = max;
    }
    // printf("get_max_nhalos set=%s maxhalos=%d max=%d\n", set->name, set->halo_info->max_nhalos, max_level);
  }
  return max_level;
}

void calculate_set_sizes(int my_rank){

  int comm_size = 0;
  MPI_Comm_size(OP_MPI_WORLD, &comm_size);

  char header_str[9000];
  char result_str[9000];
  int max_nhalos = get_max_nhalos();
  int max_calc_nhalos = get_max_calc_nhalos();

  if(my_rank == 0){
    int header_len = 0;
    snprintf(&header_str[header_len], 40, "%s", "my_rank,set,avg_size,min_size,max_size,"); header_len = strlen(header_str);

    for(int i = 0; i < max_nhalos; i++){
      snprintf(&header_str[header_len], 25, "avg_of_core_%d,", i); header_len = strlen(header_str);
      snprintf(&header_str[header_len], 25, "min_of_core_%d,", i); header_len = strlen(header_str);
      snprintf(&header_str[header_len], 25, "max_of_core_%d,", i); header_len = strlen(header_str);
    }

    for(int i = 0; i < max_nhalos; i++){
      snprintf(&header_str[header_len], 25, "avg_of_imp_exec_%d,", i); header_len = strlen(header_str);
      snprintf(&header_str[header_len], 25, "min_of_imp_exec_%d,", i); header_len = strlen(header_str);
      snprintf(&header_str[header_len], 25, "max_of_imp_exec_%d,", i); header_len = strlen(header_str);
    }

    for(int i = 0; i < max_nhalos; i++){
      snprintf(&header_str[header_len], 25, "avg_of_imp_nonexec_%d,", i); header_len = strlen(header_str);
      snprintf(&header_str[header_len], 25, "min_of_imp_nonexec_%d,", i); header_len = strlen(header_str);
      snprintf(&header_str[header_len], 25, "max_of_imp_nonexec_%d,", i); header_len = strlen(header_str);
    }

    for(int i = 0; i < max_nhalos; i++){
      snprintf(&header_str[header_len], 25, "avg_of_exp_exec_%d,", i); header_len = strlen(header_str);
      snprintf(&header_str[header_len], 25, "min_of_exp_exec_%d,", i); header_len = strlen(header_str);
      snprintf(&header_str[header_len], 25, "max_of_exp_exec_%d,", i); header_len = strlen(header_str);
    }

    for(int i = 0; i < max_nhalos; i++){
      snprintf(&header_str[header_len], 25, "avg_of_exp_nonexec_%d,", i); header_len = strlen(header_str);
      snprintf(&header_str[header_len], 25, "min_of_exp_nonexec_%d,", i); header_len = strlen(header_str);
      snprintf(&header_str[header_len], 25, "max_of_exp_nonexec_%d,", i); header_len = strlen(header_str);
    }

    for(int i = 0; i < max_nhalos; i++){
      snprintf(&header_str[header_len], 25, "avg_of_exec_msg_%d,", i); header_len = strlen(header_str);
      snprintf(&header_str[header_len], 25, "min_of_exec_msg_%d,", i); header_len = strlen(header_str);
      snprintf(&header_str[header_len], 25, "max_of_exec_msg_%d,", i); header_len = strlen(header_str);
    }

    for(int i = 0; i < max_nhalos; i++){
      snprintf(&header_str[header_len], 25, "avg_of_nonexec_msg_%d,", i); header_len = strlen(header_str);
      snprintf(&header_str[header_len], 25, "min_of_nonexec_msg_%d,", i); header_len = strlen(header_str);
      snprintf(&header_str[header_len], 25, "max_of_nonexec_msg_%d,", i); header_len = strlen(header_str);
    }

    for(int i = 0; i < max_nhalos; i++){
      snprintf(&header_str[header_len], 25, "avg_of_exec_neig_%d,", i); header_len = strlen(header_str);
      snprintf(&header_str[header_len], 25, "min_of_exec_neig_%d,", i); header_len = strlen(header_str);
      snprintf(&header_str[header_len], 25, "max_of_exec_neig_%d,", i); header_len = strlen(header_str);
    }

    for(int i = 0; i < max_nhalos; i++){
      snprintf(&header_str[header_len], 25, "avg_of_nonexec_neig_%d,", i); header_len = strlen(header_str);
      snprintf(&header_str[header_len], 25, "min_of_nonexec_neig_%d,", i); header_len = strlen(header_str);
      snprintf(&header_str[header_len], 25, "max_of_nonexec_neig_%d,", i); header_len = strlen(header_str);
    }
    op_printf("setdata,%s\n", header_str);
  }

  for(int s = 0; s < OP_set_index; s++){
    op_set set = OP_set_list[s];

    int max_set_nhalos = set->halo_info->max_nhalos;
    int max_set_calc_nhalos = set->halo_info->max_calc_nhalos;

    int sizes[max_nhalos];
    int avg_sizes[max_nhalos];
    int min_sizes[max_nhalos];
    int max_sizes[max_nhalos];

    int local_avg_sizes[max_nhalos];
    int local_min_sizes[max_nhalos];
    int local_max_sizes[max_nhalos];

    int avg_size;
    int min_size;
    int max_size;

    int len = 0;
    if(my_rank == MPI_ROOT){
      snprintf(&result_str[len], 10, "%d,", my_rank); len = strlen(result_str);
      snprintf(&result_str[len], 20, "%s,", set->name); len = strlen(result_str);
    }

    // ============ set sizes ============
    // reset
    avg_size = 0; min_size = 0; max_size = 0;

    MPI_Reduce(&set->size, &avg_size, 1, MPI_INT, MPI_SUM, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(&set->size, &min_size, 1, MPI_INT, MPI_MIN, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(&set->size, &max_size, 1, MPI_INT, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);

    if(my_rank == MPI_ROOT){
      snprintf(&result_str[len], 20, "%d,", avg_size / comm_size); len = strlen(result_str);
      snprintf(&result_str[len], 20, "%d,", min_size); len = strlen(result_str);
      snprintf(&result_str[len], 20, "%d,", max_size); len = strlen(result_str);
    }

    // ============ core sizes ============
    // reset
    for(int i = 0; i < max_nhalos; i++){
      sizes[i] = 0; avg_sizes[i] = 0; min_sizes[i] = 0; max_sizes[i] = 0;
    }

    MPI_Reduce(set->core_sizes, avg_sizes, max_nhalos, MPI_INT, MPI_SUM, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(set->core_sizes, min_sizes, max_nhalos, MPI_INT, MPI_MIN, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(set->core_sizes, max_sizes, max_nhalos, MPI_INT, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);

    if(my_rank == MPI_ROOT){
      for(int i = 0; i < max_nhalos; i++){
          snprintf(&result_str[len], 20, "%d,", avg_sizes[i] / comm_size); len = strlen(result_str);
          snprintf(&result_str[len], 20, "%d,", min_sizes[i]); len = strlen(result_str);
          snprintf(&result_str[len], 20, "%d,", max_sizes[i]); len = strlen(result_str);
      }
    }

    // ============ imp exec sizes ============
    // reset
    for(int i = 0; i < max_nhalos; i++){
      sizes[i] = 0; avg_sizes[i] = 0; min_sizes[i] = 0; max_sizes[i] = 0;
    }
    
    for(int i = 0; i < max_set_calc_nhalos; i++){
      halo_list imp_exec_list = OP_aug_import_exec_lists[i][set->index];
      sizes[i] = (imp_exec_list) ? imp_exec_list->size : 0;
    }

    MPI_Reduce(sizes, avg_sizes, max_nhalos, MPI_INT, MPI_SUM, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(sizes, min_sizes, max_nhalos, MPI_INT, MPI_MIN, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(sizes, max_sizes, max_nhalos, MPI_INT, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);

    if(my_rank == MPI_ROOT){
      for(int i = 0; i < max_nhalos; i++){
          snprintf(&result_str[len], 20, "%d,", avg_sizes[i] / comm_size); len = strlen(result_str);
          snprintf(&result_str[len], 20, "%d,", min_sizes[i]); len = strlen(result_str);
          snprintf(&result_str[len], 20, "%d,", max_sizes[i]); len = strlen(result_str);
      }
    }

    // ============ imp nonexec sizes ============
    // reset
    for(int i = 0; i < max_nhalos; i++){
      sizes[i] = 0; avg_sizes[i] = 0; min_sizes[i] = 0; max_sizes[i] = 0;
    }
    
    for(int i = 0; i < max_set_calc_nhalos; i++){
      halo_list imp_nonexec_list = OP_aug_import_nonexec_lists[i][set->index];
      sizes[i] = (imp_nonexec_list) ? imp_nonexec_list->size : 0;
    }

    MPI_Reduce(sizes, avg_sizes, max_nhalos, MPI_INT, MPI_SUM, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(sizes, min_sizes, max_nhalos, MPI_INT, MPI_MIN, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(sizes, max_sizes, max_nhalos, MPI_INT, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);

    if(my_rank == MPI_ROOT){
      for(int i = 0; i < max_nhalos; i++){
          snprintf(&result_str[len], 20, "%d,", avg_sizes[i] / comm_size); len = strlen(result_str);
          snprintf(&result_str[len], 20, "%d,", min_sizes[i]); len = strlen(result_str);
          snprintf(&result_str[len], 20, "%d,", max_sizes[i]); len = strlen(result_str);
      }
    }

    // ============ exp exec sizes ============
    // reset
    for(int i = 0; i < max_nhalos; i++){
      sizes[i] = 0; avg_sizes[i] = 0; min_sizes[i] = 0; max_sizes[i] = 0;
    }
    
    for(int i = 0; i < max_set_calc_nhalos; i++){
      halo_list exp_exec_list = OP_aug_export_exec_lists[i][set->index];
      sizes[i] = (exp_exec_list) ? exp_exec_list->size : 0;
    }

    MPI_Reduce(sizes, avg_sizes, max_nhalos, MPI_INT, MPI_SUM, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(sizes, min_sizes, max_nhalos, MPI_INT, MPI_MIN, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(sizes, max_sizes, max_nhalos, MPI_INT, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);

    if(my_rank == MPI_ROOT){
      for(int i = 0; i < max_nhalos; i++){
          snprintf(&result_str[len], 20, "%d,", avg_sizes[i] / comm_size); len = strlen(result_str);
          snprintf(&result_str[len], 20, "%d,", min_sizes[i]); len = strlen(result_str);
          snprintf(&result_str[len], 20, "%d,", max_sizes[i]); len = strlen(result_str);
      }
    }

    // ============ exp nonexec sizes ============
    // reset
    for(int i = 0; i < max_nhalos; i++){
      sizes[i] = 0; avg_sizes[i] = 0; min_sizes[i] = 0; max_sizes[i] = 0;
    }
    
    for(int i = 0; i < max_set_calc_nhalos; i++){
      halo_list exp_nonexec_list = OP_aug_export_nonexec_lists[i][set->index];
      sizes[i] = (exp_nonexec_list) ? exp_nonexec_list->size : 0;
    }

    MPI_Reduce(sizes, avg_sizes, max_nhalos, MPI_INT, MPI_SUM, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(sizes, min_sizes, max_nhalos, MPI_INT, MPI_MIN, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(sizes, max_sizes, max_nhalos, MPI_INT, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);

    if(my_rank == MPI_ROOT){
      for(int i = 0; i < max_nhalos; i++){
          snprintf(&result_str[len], 20, "%d,", avg_sizes[i] / comm_size); len = strlen(result_str);
          snprintf(&result_str[len], 20, "%d,", min_sizes[i]); len = strlen(result_str);
          snprintf(&result_str[len], 20, "%d,", max_sizes[i]); len = strlen(result_str);
      }
    }

    // ============ exec message sizes ============
    // reset
    for(int i = 0; i < max_nhalos; i++){
      sizes[i] = 0; avg_sizes[i] = 0; min_sizes[i] = 0; max_sizes[i] = 0;
      local_avg_sizes[i] = 0; local_min_sizes[i] = 0; local_max_sizes[i] = 0;
    }

    // calculate average, min, max of export exec halos
    for(int i = 0; i < max_set_calc_nhalos; i++){
      halo_list exp_exec_list = OP_aug_export_exec_lists[i][set->index];
      if(!exp_exec_list)
        continue;

      int exec_total = 0;
      int max_exec = 0;
      int min_exec = INT_MAX;
      int rank_count = exp_exec_list->ranks_size / exp_exec_list->num_levels;
      for(int r = 0; r < rank_count; r++){
        int size = exp_exec_list->sizes[r];
        if(size > max_exec){
          max_exec = size;
        }
        if(size < min_exec){
          min_exec = size;
        }
        exec_total += size; 
      }
      // printf("rank_count=%d set=%s maxhalo=%d exec_total=%d size=%d, levels=%d\n", rank_count, 
      // set->name, max_nhalos, exec_total,
      // exp_exec_list->ranks_size, exp_exec_list->num_levels);
      local_avg_sizes[i] = (rank_count > 0) ? exec_total / rank_count : 0;
      local_min_sizes[i] = (rank_count > 0) ? min_exec : 0;
      local_max_sizes[i] = max_exec;
    }

    MPI_Reduce(local_avg_sizes, avg_sizes, max_nhalos, MPI_INT, MPI_SUM, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(local_min_sizes, min_sizes, max_nhalos, MPI_INT, MPI_MIN, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(local_max_sizes, max_sizes, max_nhalos, MPI_INT, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);

    if(my_rank == MPI_ROOT){
      for(int i = 0; i < max_nhalos; i++){
          snprintf(&result_str[len], 20, "%d,", avg_sizes[i] / comm_size); len = strlen(result_str);
          snprintf(&result_str[len], 20, "%d,", min_sizes[i]); len = strlen(result_str);
          snprintf(&result_str[len], 20, "%d,", max_sizes[i]); len = strlen(result_str);
      }
    }

    // ============ nonexec message sizes ============
    // reset
    for(int i = 0; i < max_nhalos; i++){
      sizes[i] = 0; avg_sizes[i] = 0; min_sizes[i] = 0; max_sizes[i] = 0;
      local_avg_sizes[i] = 0; local_min_sizes[i] = 0; local_max_sizes[i] = 0;
    }

    // calculate average, min, max of export nonexec halos
    for(int i = 0; i < max_set_calc_nhalos; i++){
      halo_list exp_nonexec_list = OP_aug_export_nonexec_lists[i][set->index];
      if(!exp_nonexec_list)
        continue;
      int exec_total = 0;
      int max_exec = 0;
      int min_exec = INT_MAX;
      int rank_count = exp_nonexec_list->ranks_size / exp_nonexec_list->num_levels;
      for(int r = 0; r < rank_count; r++){
        int size = exp_nonexec_list->sizes[r];
        if(size > max_exec){
          max_exec = size;
        }
        if(size < min_exec){
          min_exec = size;
        }
        exec_total += size; 
      }
      // printf("rank_count=%d set=%s maxhalo=%d exec_total=%d size=%d, levels=%d\n", rank_count, 
      // set->name, max_nhalos, exec_total,
      // exp_exec_list->ranks_size, exp_exec_list->num_levels);
      local_avg_sizes[i] = (rank_count > 0) ? exec_total / rank_count : 0;
      local_min_sizes[i] = (rank_count > 0) ? min_exec : 0;
      local_max_sizes[i] = max_exec;
    }

    MPI_Reduce(local_avg_sizes, avg_sizes, max_nhalos, MPI_INT, MPI_SUM, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(local_min_sizes, min_sizes, max_nhalos, MPI_INT, MPI_MIN, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(local_max_sizes, max_sizes, max_nhalos, MPI_INT, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);

    if(my_rank == MPI_ROOT){
      for(int i = 0; i < max_nhalos; i++){
          snprintf(&result_str[len], 20, "%d,", avg_sizes[i] / comm_size); len = strlen(result_str);
          snprintf(&result_str[len], 20, "%d,", min_sizes[i]); len = strlen(result_str);
          snprintf(&result_str[len], 20, "%d,", max_sizes[i]); len = strlen(result_str);
      }
    }

    // ============ exec halo neighbors ============
    // reset
    for(int i = 0; i < max_nhalos; i++){
      sizes[i] = 0; avg_sizes[i] = 0; min_sizes[i] = 0; max_sizes[i] = 0;
    }
    
    for(int i = 0; i < max_set_calc_nhalos; i++){
      halo_list exp_exec_list = OP_aug_export_exec_lists[i][set->index];
      sizes[i] = (exp_exec_list) ? exp_exec_list->ranks_size / exp_exec_list->num_levels: 0;
    }

    MPI_Reduce(sizes, avg_sizes, max_nhalos, MPI_INT, MPI_SUM, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(sizes, min_sizes, max_nhalos, MPI_INT, MPI_MIN, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(sizes, max_sizes, max_nhalos, MPI_INT, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);

    if(my_rank == MPI_ROOT){
      for(int i = 0; i < max_nhalos; i++){
          snprintf(&result_str[len], 20, "%d,", avg_sizes[i] / comm_size); len = strlen(result_str);
          snprintf(&result_str[len], 20, "%d,", min_sizes[i]); len = strlen(result_str);
          snprintf(&result_str[len], 20, "%d,", max_sizes[i]); len = strlen(result_str);
      }
    }

    // ============ nonexec halo neighbors ============
    // reset
    for(int i = 0; i < max_nhalos; i++){
      sizes[i] = 0; avg_sizes[i] = 0; min_sizes[i] = 0; max_sizes[i] = 0;
    }
    
    for(int i = 0; i < max_set_calc_nhalos; i++){
      halo_list exp_nonexec_list = OP_aug_export_nonexec_lists[i][set->index];
      sizes[i] = (exp_nonexec_list) ? exp_nonexec_list->ranks_size / exp_nonexec_list->num_levels: 0;
    }

    MPI_Reduce(sizes, avg_sizes, max_nhalos, MPI_INT, MPI_SUM, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(sizes, min_sizes, max_nhalos, MPI_INT, MPI_MIN, MPI_ROOT, OP_MPI_WORLD);
    MPI_Reduce(sizes, max_sizes, max_nhalos, MPI_INT, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);

    if(my_rank == MPI_ROOT){
      for(int i = 0; i < max_nhalos; i++){
          snprintf(&result_str[len], 20, "%d,", avg_sizes[i] / comm_size); len = strlen(result_str);
          snprintf(&result_str[len], 20, "%d,", min_sizes[i]); len = strlen(result_str);
          snprintf(&result_str[len], 20, "%d,", max_sizes[i]); len = strlen(result_str);
      }
    }
    op_printf("setdata,%s\n", result_str);
  }
}

void calculate_dat_sizes(int my_rank){

  int max_nhalos = get_max_nhalos();
  char header_str[300];

  if(my_rank == MPI_ROOT){
    int header_len = 0;
    snprintf(&header_str[header_len], 40, "%s", "my_rank,to_rank,dat,set,dat_size,"); header_len = strlen(header_str);
  }
  op_printf("cadata,%s\n", header_str);

  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    op_set set = dat->set;
    char result_str[300];

    int len = 0;
    if(my_rank == MPI_ROOT){
      snprintf(&result_str[len], 10, "%d,", my_rank); len = strlen(result_str);
      snprintf(&result_str[len], 10, "%d,", my_rank); len = strlen(result_str);
      snprintf(&result_str[len], 40, "%s,", dat->name); len = strlen(result_str);
      snprintf(&result_str[len], 40, "%s,", set->name); len = strlen(result_str);
      snprintf(&result_str[len], 10, "%d,", dat->size); len = strlen(result_str);
    }
    op_printf("cadata,%s\n", result_str);
  }
}

#ifdef COMM_AVOID_CUDA
void ca_realloc_comm_buffer(char **send_buffer_host, char **recv_buffer_host, 
      char **send_buffer_device, char **recv_buffer_device, int device, 
      unsigned size_send, unsigned size_recv) {
  if (ca_grp_size_recv_old < size_recv) {
    if (*recv_buffer_device != NULL) cutilSafeCall(cudaFree(*recv_buffer_device));
    cutilSafeCall(cudaMalloc(recv_buffer_device, size_recv));
    if (ca_grp_size_recv_old > 0) cutilSafeCall(cudaHostUnregister ( *recv_buffer_host ));
    *recv_buffer_host = (char*)op_realloc(*recv_buffer_host, size_recv);
    cutilSafeCall(cudaHostRegister ( *recv_buffer_host, size_recv, cudaHostRegisterDefault ));
    ca_grp_size_recv_old = size_recv;
  }
  if (ca_grp_size_send_old < size_send) {
    //if (*send_buffer_host != NULL) cutilSafeCall(cudaFreeHost(*send_buffer_host));
    if (*send_buffer_device != NULL) cutilSafeCall(cudaFree(*send_buffer_device));
    cutilSafeCall(cudaMalloc(send_buffer_device, size_send));
    //cutilSafeCall(cudaMallocHost(send_buffer_host, size_recv));
    if (ca_grp_size_send_old > 0) cutilSafeCall(cudaHostUnregister ( *send_buffer_host ));
    *send_buffer_host = (char*)op_realloc(*send_buffer_host, size_send);
    cutilSafeCall(cudaHostRegister ( *send_buffer_host, size_send, cudaHostRegisterDefault ));
    ca_grp_size_send_old = size_send;
  }
}
#endif

void set_group_halo_envt(){

  grp_tag = 100;
  int max_dat_count = 5;  // 1 for optimistic, 16 for pessimistic - MGCFD

  int max_send_buff_size = 0;
  int max_recv_buff_size = 0;

  int max_send_rank_size = 0;
  int max_recv_rank_size = 0;

  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    op_set set = dat->set;

    halo_list exp_list = OP_merged_export_exec_nonexec_list[set->index];
    halo_list imp_list = OP_merged_import_exec_nonexec_list[set->index];
    
    int send_buff_size = exp_list->size * dat->size; // * dat->dim;
    int recv_buff_size = imp_list->size * dat->size; // * dat->dim;

    int send_rank_size = exp_list->ranks_size / exp_list->num_levels;
    int recv_rank_size = imp_list->ranks_size / imp_list->num_levels;

    if(send_buff_size > max_send_buff_size)
      max_send_buff_size = send_buff_size;

    if(recv_buff_size > max_recv_buff_size)
      max_recv_buff_size = recv_buff_size;

    if(send_rank_size > max_send_rank_size)
      max_send_rank_size = send_rank_size;

    if(recv_rank_size > max_recv_rank_size)
      max_recv_rank_size = recv_rank_size;

  }

  grp_send_requests = (MPI_Request *)xmalloc(sizeof(MPI_Request) * max_send_rank_size * max_dat_count);
  grp_recv_requests = (MPI_Request *)xmalloc(sizeof(MPI_Request) * max_recv_rank_size * max_dat_count);

  grp_send_buffer = (char *)xmalloc(max_send_buff_size * max_dat_count);
  grp_recv_buffer = (char *)xmalloc(max_recv_buff_size * max_dat_count);
#ifdef COMM_AVOID_CUDA
  // op_printf("max_send_buff_size=%d ca_grp_size_send_old=%d max_recv_buff_size=%d ca_grp_size_recv_old=%d max_dat_count=%d\n", max_send_buff_size, ca_grp_size_send_old, max_recv_buff_size, ca_grp_size_recv_old, max_dat_count);
  ca_realloc_comm_buffer(&grp_send_buffer_h, &grp_recv_buffer_h, 
      &grp_send_buffer_d, &grp_recv_buffer_d, 1, 
      max_send_buff_size * max_dat_count, max_recv_buff_size * max_dat_count);
#endif
}


// Merge all the exec and non-exec halos together for the all the halo levels
// exec 0 | non-exec 0 | exec 1 | non-exec 1 | ...
// If we don't have a non-exec level for an exec level since it is not required for calculations,
// it will be kept as null in the merging halo list array and merged only the available halo lists 
// together
void merge_exec_nonexec_halos(int halo_levels, int my_rank, int comm_size){

  OP_merged_import_exec_nonexec_list = (halo_list*) xmalloc(OP_set_index * sizeof(halo_list));
  OP_merged_export_exec_nonexec_list = (halo_list*) xmalloc(OP_set_index * sizeof(halo_list));

  for(int s = 0; s < OP_set_index; s++){
    op_set set = OP_set_list[s];
    OP_merged_import_exec_nonexec_list[set->index] = NULL;
    OP_merged_export_exec_nonexec_list[set->index] = NULL;
  }

  int max_exp_rank_count = 0;

  for(int s = 0; s < OP_set_index; s++){
    op_set set = OP_set_list[s];
    int max_nhalos = set->halo_info->max_nhalos;
    int max_calc_nhalos = set->halo_info->max_calc_nhalos;
    int halo_merge_count = set->halo_info->max_calc_nhalos;

    halo_list* import_h_lists = (halo_list*) xmalloc((2 * halo_merge_count) * sizeof(halo_list));
    halo_list* export_h_lists = (halo_list*) xmalloc((2 * halo_merge_count) * sizeof(halo_list));

    for(int l = 0; l < halo_merge_count; l++){
      import_h_lists[2 * l] = OP_aug_import_exec_lists[l][set->index];
      import_h_lists[2 * l + 1] = OP_aug_import_nonexec_lists[l][set->index];
      export_h_lists[2 * l] = OP_aug_export_exec_lists[l][set->index];
      export_h_lists[2 * l + 1] = OP_aug_export_nonexec_lists[l][set->index];
    }

    OP_merged_import_exec_nonexec_list[set->index] = merge_halo_lists(2 * halo_merge_count, import_h_lists, my_rank, comm_size);
    OP_merged_export_exec_nonexec_list[set->index] = merge_halo_lists(2 * halo_merge_count, export_h_lists, my_rank, comm_size);

    for(int l = 1; l < max_nhalos; l++){
      op_single_halo_destroy(OP_aug_import_exec_lists[l][set->index]); OP_aug_import_exec_lists[l][set->index] = NULL;
      op_single_halo_destroy(OP_aug_import_nonexec_lists[l][set->index]); OP_aug_import_nonexec_lists[l][set->index] = NULL;
      op_single_halo_destroy(OP_aug_export_exec_lists[l][set->index]); OP_aug_export_exec_lists[l][set->index] = NULL;
      op_single_halo_destroy(OP_aug_export_nonexec_lists[l][set->index]); OP_aug_export_nonexec_lists[l][set->index] = NULL;
    }

    op_free(import_h_lists);
    op_free(export_h_lists);

    int rank_count = OP_merged_export_exec_nonexec_list[set->index]->ranks_size / OP_merged_export_exec_nonexec_list[set->index]->num_levels;
    if(rank_count > max_exp_rank_count){
      max_exp_rank_count = rank_count;
    }
  }

  ca_buf_pos = (int*) xmalloc(max_exp_rank_count* sizeof(int));
  ca_send_sizes = (int*) xmalloc(max_exp_rank_count* sizeof(int));

  for(int i = 0; i < max_exp_rank_count; i++){
    ca_buf_pos[i] = 0;
    ca_send_sizes[i] = 0;
  }
}

// Setting additional halo layers for maps in hydra
void set_maps_hydra(){
  // printf("set_maps_hydra maps and dats\n");
  for (int m = 0; m < OP_map_index; m++) { // for each maping table
    op_map map = OP_map_list[m];
    if (strncmp("ne", map->name, strlen("ne")) == 0) {
      op_mpi_add_nhalos_map(map, 2);
      op_mpi_add_nhalos_map_calc(map, 2);
    }
    if (strncmp("npe", map->name, strlen("npe")) == 0) {
      op_mpi_add_nhalos_map(map, 2);
      op_mpi_add_nhalos_map_calc(map, 2);
    }
    if (strncmp("ncb", map->name, strlen("ncb")) == 0) {
      op_mpi_add_nhalos_map(map, 2);
      op_mpi_add_nhalos_map_calc(map, 2);
    }
    if (strncmp("nb", map->name, strlen("nb")) == 0) {
      op_mpi_add_nhalos_map(map, 2);
      op_mpi_add_nhalos_map_calc(map, 2);
    }
    if (strncmp("nwe", map->name, strlen("nwe")) == 0 && strlen("nwe") == strlen(map->name)) {
      op_mpi_add_nhalos_map(map, 2);
      op_mpi_add_nhalos_map_calc(map, 2);
    }
  }
  return;
}

// Calculate halo layer sizes and oter information required for op_set s
void calc_set_metrics(int l){

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    int max_calc_level = set->halo_info->max_calc_nhalos;
    int max_level = set->halo_info->max_nhalos;

    if(l >= max_level)
      continue;

    set->exec_sizes[l] = (l > 0) ? set->exec_sizes[l - 1] + ((OP_aug_import_exec_lists[l][set->index]) ? OP_aug_import_exec_lists[l][set->index]->size : 0) : 
      ((OP_aug_import_exec_lists[l][set->index]) ? OP_aug_import_exec_lists[l][set->index]->size : 0);
    set->exp_exec_sizes[l] = (l > 0) ? set->exp_exec_sizes[l - 1] + ((OP_aug_export_exec_lists[l][set->index]) ? OP_aug_export_exec_lists[l][set->index]->size : 0) : 
      ((OP_aug_export_exec_lists[l][set->index]) ? OP_aug_export_exec_lists[l][set->index]->size : 0);

    if(l + 1 == DEFAULT_HALO_COUNT){
      set->exec_size = (l > 0) ? set->exec_sizes[l - 1] + (OP_aug_import_exec_lists[l][set->index] ? 
          OP_aug_import_exec_lists[l][set->index]->size : 0): set->exec_sizes[l];
    }

    set->nonexec_sizes[l] = (OP_aug_import_nonexec_lists[l][set->index]) ?
                              OP_aug_import_nonexec_lists[l][set->index]->size : 0;  //duplicate elements in the on exec. so no +=

    set->exp_nonexec_sizes[l] = (l < max_calc_level) ?
                              OP_aug_export_nonexec_lists[l][set->index]->size : 0;
    if(l + 1 == DEFAULT_HALO_COUNT){
      set->nonexec_size = 0;
      // for(int l = 0; l <= el; l++){
        set->nonexec_size = (OP_aug_import_nonexec_lists[l][set->index]) ? OP_aug_import_nonexec_lists[l][set->index]->size : 0;
      // }
    }
    if(l >= max_calc_level)
      continue;
    set->total_nonexec_size +=  (l < max_calc_level) ? OP_aug_import_nonexec_lists[l][set->index]->size : 0;
    set->total_exec_size += OP_aug_import_exec_lists[l][set->index] ? OP_aug_import_exec_lists[l][set->index]->size : 0;
  }
}

/*******************************************************************************
 * Utility functions - end
 *******************************************************************************/

/*******************************************************************************
 * Unused functions - start
 *******************************************************************************/

// Merge all the exec halos together - not used now
void merge_exec_halos(int exec_levels, int my_rank, int comm_size){

  OP_merged_import_exec_list = (halo_list*) xmalloc(OP_set_index * sizeof(halo_list));
  OP_merged_export_exec_list = (halo_list*) xmalloc(OP_set_index * sizeof(halo_list));

  for(int s = 0; s < OP_set_index; s++){
    op_set set = OP_set_list[s];
    OP_merged_import_exec_list[set->index] = NULL;
    OP_merged_export_exec_list[set->index] = NULL;
  }

  halo_list* import_h_lists = (halo_list*) xmalloc(exec_levels * sizeof(halo_list));
  halo_list* export_h_lists = (halo_list*) xmalloc(exec_levels * sizeof(halo_list));

  for(int s = 0; s < OP_set_index; s++){
    op_set set = OP_set_list[s];
    for(int l = 0; l < exec_levels; l++){
      import_h_lists[l] = OP_aug_import_exec_lists[l][set->index];
      export_h_lists[l] = OP_aug_export_exec_lists[l][set->index];
    }

    OP_merged_import_exec_list[set->index] = merge_halo_lists(exec_levels, import_h_lists, my_rank, comm_size);
    OP_merged_export_exec_list[set->index] = merge_halo_lists(exec_levels, export_h_lists, my_rank, comm_size);
  }
  op_free(import_h_lists);
  op_free(export_h_lists);
}

// Merge all the non-exec halos together - not used now
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

    op_free(import_h_lists);
    op_free(export_h_lists);
  }
}

// not used
void set_maps_halo_extension(){
  for (int m = 0; m < OP_map_index; m++) { // for each maping table
    op_map map = OP_map_list[m];
    if (strncmp("pecell", map->name, strlen("pecell")) == 0) {
      op_mpi_add_nhalos_map(map, 2);
      printf("haloextension map dat dat=%s\n", map->name);
    }
    if (strncmp("pedge", map->name, strlen("pedge")) == 0) {
      op_mpi_add_nhalos_map(map, 2);
      printf("haloextension1 map dat dat=%s\n", map->name);
    }
    if (strncmp("pcell", map->name, strlen("pcell")) == 0) {
      op_mpi_add_nhalos_map(map, 2);
      printf("haloextension1 map dat dat=%s\n", map->name);
    }
  }
}

// not used
void set_dats_mgcfd(){

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    op_mpi_add_nhalos_set(set, 2);
    op_mpi_add_nhalos_set(set, 3);
    printf("setdat set=%s max_halo=%d count=%d cap=%d nhalos[0]=%d\n", 
    set->name, set->halo_info->max_nhalos, set->halo_info->nhalos_count, set->halo_info->nhalos_cap, set->halo_info->nhalos[0]);
  }
  return;

  op_dat_entry *item;
  int d = -1; // d is just simply the tag for mpi comms
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    if (strncmp("p_variables_L0", dat->name, strlen("p_variables_L0")) == 0|| 
        strncmp("p_fluxes_L0", dat->name, strlen("p_fluxes_L0")) == 0) { // if this data array
      printf("set dat dat=%s\n", dat->name);

    }
    else{
      printf("set dat other dat=%s\n", dat->name);
    }
  }
}

// not used
void set_maps_mgcfd(){
   printf("set_maps_mgcfd maps and dats\n");
  for (int m = 0; m < OP_map_index; m++) { // for each maping table
    op_map map = OP_map_list[m];
    if (strncmp("edge-->node", map->name, strlen(map->name)) == 0) {
      op_mpi_add_nhalos_map(map, 4);
      printf("op_mpi_add_nhalos_map map=%s\n", map->name);
      
    }
  }
  return;
  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    if(
      (strncmp("vol", dat->name, strlen(dat->name)) == 0)
       ){
         printf("op_mpi_add_nhalos_dat dat=%s\n", dat->name);
          op_mpi_add_nhalos_dat(dat, 2);
    }
  }
}

// Remove augmented maps which are not required for calculations
void op_remove_aug_map(op_map map, int map_id){
  if(map->aug_maps[map_id] != NULL){
    op_free(map->aug_maps[map_id]);
    map->aug_maps[map_id] = NULL;
  }
}

/*******************************************************************************
 * Unused functions - end
 *******************************************************************************/


// Create export exec list (including the standard OP2 halo layer and augmented layers)
halo_list* create_export_exec_list(int halo_id, int **part_range, int my_rank, int comm_size){

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

        if (compare_sets(map->from, set) == 1 && is_halo_required_for_map(map, halo_id) == 1) { // need to select mappings
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

            if (halo_id == 0 && part != my_rank) {
              set_list[s_i++] = part; // add to set export list
              set_list[s_i++] = e;
              // update_elem_rank_matrix(set, e, part, comm_size);
            }

            if(halo_id > 0){
              for(int r = 0; r < comm_size; r++){
                // We omited using elem rank matrix due to high memory consumption
                // if(r != part && parts[r] == 1 && !is_elem_sent(set, e, r, comm_size)){
                if(r != part && parts[r] == 1 && !is_in_prev_export_exec_halos(halo_id, set->index, r, e, my_rank)){
                  set_list[s_i++] = r; // add to set export list
                  set_list[s_i++] = e;
                  // update_elem_rank_matrix(set, e, r, comm_size);
                }
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

// Exchange exec halo mappings
void exchange_exec_mappings(int exec_level, int **part_range, int my_rank, int comm_size){

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
    int prev_nonexec_size = 0;
    for(int i = 0; i < exec_level; i++){
      prev_nonexec_size += map->from->nonexec_sizes[i];
    }
    if(exec_level > 0)
      prev_exec_size = map->from->exec_sizes[exec_level - 1];

    OP_map_list[map->index]->map = (int *)xrealloc(
        OP_map_list[map->index]->map,
        (map->dim * (map->from->size + prev_exec_size + prev_nonexec_size + i_list->size)) * sizeof(int));

    int init = map->dim * (map->from->size + prev_exec_size + prev_nonexec_size);
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

// Exchange exec halo data
void exchange_exec_data(int exec_level, int **part_range, int my_rank, int comm_size){
  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];

    if(is_set_required_for_calc(set, exec_level) != 1){
      continue;
    }

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
        int prev_nonexec_size = 0;
        for(int i = 0; i < exec_level; i++){
          halo_list prev_h_list = OP_aug_import_exec_lists[i][set->index];
          prev_exec_size += prev_h_list->size;
          prev_h_list = OP_aug_import_nonexec_lists[i][set->index];
          prev_nonexec_size += (prev_h_list) ? prev_h_list->size : 0;
        }

        dat->data =
            (char *)xrealloc(dat->data, (set->size + prev_exec_size + prev_nonexec_size + i_list->size) * dat->size);

        int init = (set->size + prev_exec_size + prev_nonexec_size) * dat->size;
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


// Renumber the mappings inicluding augmented maps
void renumber_mappings(int dummy, int **part_range, int my_rank, int comm_size){

  prepare_aug_maps();
  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];

    for (int m = 0; m < OP_map_index; m++) { // for each maping table
      op_map map = OP_map_list[m];

      if (compare_sets(map->to, set) == 1) { // need to select
                                            // mappings TO this set

        int max_level = map->halo_info->max_calc_nhalos;
        for(int el = 0; el < max_level; el++){

          if(is_halo_required_for_map(map, el) != 1 || is_map_required_for_calc(map, el) != 1)
            continue;
          
          if(is_halo_required_for_set(map->from, el) != 1)
            continue;

          int exec_levels = el + 1; // map->from->halo_info->nhalos[el];
          halo_list nonexec_set_list = OP_aug_import_nonexec_lists[el][set->index]; // this should be not NULL

          // for each entry in this mapping table: original+execlist
          int start = 0;
          int end = 0;
          int current_nonexec_size = 0;
          for(int sl = 0; sl < exec_levels; sl++){
            //get exec level size
            int exec_map_len = 0;
            int nonexec_map_len = 0;
            for(int l = 0; l < sl + 1; l++){
              exec_map_len += OP_aug_import_exec_lists[l][map->from->index]->size;
              if(l > 0){
                nonexec_map_len += (OP_aug_import_nonexec_lists[l - 1][map->from->index])?
                                  OP_aug_import_nonexec_lists[l - 1][map->from->index]->size : 0;
              }
            }

            start = end + current_nonexec_size;
            end = map->from->size + exec_map_len + nonexec_map_len;
            current_nonexec_size = (OP_aug_import_nonexec_lists[sl][map->from->index])?
                                  OP_aug_import_nonexec_lists[sl][map->from->index]->size : 0;

            for (int e = start; e < end; e++) {
              for (int j = 0; j < map->dim; j++) { // for each element
                                                  // pointed at by this entry
                int part;
                int local_index = 0;
                part = get_partition(map->map_org[e * map->dim + j],
                                    part_range[map->to->index], &local_index,
                                    comm_size);

                if (part == my_rank) {
                  if(exec_levels == DEFAULT_HALO_COUNT){
                    OP_map_list[map->index]->map[e * map->dim + j] = local_index;
                    
                  }

                  OP_map_list[map->index]->aug_maps[el][e * map->dim + j] = local_index;
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
                        int prev_nonexec_set_list_size = 0;
                        for(int l1 = 0; l1 < l; l1++){  //take the size of prev exec levels
                          prev_nonexec_set_list_size += (OP_aug_import_nonexec_lists[l1][set->index]) ? 
                                                          OP_aug_import_nonexec_lists[l1][set->index]->size : 0;
                        }
                        if(exec_levels == DEFAULT_HALO_COUNT){
                          OP_map_list[map->index]->map[e * map->dim + j] =
                              found + map->to->size + prev_exec_set_list_size + prev_nonexec_set_list_size;
                        }
                        
                        OP_map_list[map->index]->aug_maps[el][e * map->dim + j] =
                            found + map->to->size + prev_exec_set_list_size + prev_nonexec_set_list_size;
                        break;
                      }
                  }
                  // check in nonexec list
                  if(nonexec_set_list == NULL)  // additional check.
                    continue;
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
                        exec_set_list_size += (OP_aug_import_exec_lists[l][set->index]) ?
                                                OP_aug_import_exec_lists[l][set->index]->size : 0;
                      }

                      int non_exec_set_list_size = 0;
                      for(int l = 0; l < exec_levels - 1; l++){
                        non_exec_set_list_size += (OP_aug_import_nonexec_lists[l][set->index]) ?
                                                    OP_aug_import_nonexec_lists[l][set->index]->size : 0;
                      }

                      if(exec_levels == DEFAULT_HALO_COUNT){
                        OP_map_list[map->index]->map[e * map->dim + j] =
                            found + set->size + exec_set_list_size + non_exec_set_list_size;
                      }

                      OP_map_list[map->index]->aug_maps[el][e * map->dim + j] =
                          found + set->size + exec_set_list_size + non_exec_set_list_size;
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
}

// Create import non-exec halo lists (including the standard OP2 halo layer and augmented layers)
void create_import_nonexec_list(int halo_id, int **part_range, int my_rank, int comm_size){

  OP_aug_import_nonexec_lists[halo_id] = (halo_list *)xmalloc(OP_set_index * sizeof(halo_list));
  OP_aug_export_nonexec_lists[halo_id] = (halo_list *)xmalloc(OP_set_index * sizeof(halo_list));

  int s_i;
  int *set_list;
  int cap_s = 1000; // keep track of the temp array capacities

  s_i = 0;
  set_list = NULL;

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];
    int from_exec_levels = 0;
    int to_exec_levels = 0;
    int el = halo_id;

    if(is_halo_required_for_set(set, el) != 1 || is_set_required_for_calc(set, el) != 1){
      OP_aug_import_nonexec_lists[el][set->index] = NULL;
      continue;
    }
    to_exec_levels = el + 1;

    // create a temporaty scratch space to hold nonexec export list for this set
    s_i = 0;
    cap_s = 1000; // keep track of the temp array capacity
    set_list = (int *)xmalloc(cap_s * sizeof(int));

    for (int m = 0; m < OP_map_index; m++) { // for each maping table
      op_map map = OP_map_list[m];
      if(is_halo_required_for_set(map->from, el) != 1 || is_set_required_for_calc(map->from, el) != 1){
        continue;
      }
      int start = 0;
      int end = 0;
      int current_nonexec_size = 0;
      from_exec_levels = el + 1;
      
      for(int sl = 0; sl < from_exec_levels; sl++){
        int exec_size = 0;
        int nonexec_size = 0;
        for(int l = 0; l < sl + 1; l++){
          exec_size += (OP_aug_import_exec_lists[l] && OP_aug_import_exec_lists[l][map->from->index]) ? 
          OP_aug_import_exec_lists[l][map->from->index]->size : 0;
          if(l > 0){
            nonexec_size += (OP_aug_import_nonexec_lists[l - 1][map->from->index])?
                              OP_aug_import_nonexec_lists[l - 1][map->from->index]->size : 0;
          }
        }

        if (compare_sets(map->to, set) == 1) { // need to select
                                              // mappings TO this set
          if(sl > 0){
            current_nonexec_size = (OP_aug_import_nonexec_lists[sl - 1][map->from->index])?
                              OP_aug_import_nonexec_lists[sl - 1][map->from->index]->size : 0;
          }        
          start = end + current_nonexec_size;
          end = map->from->size + exec_size + nonexec_size;

          // for each entry in this mapping table: original+execlist
          for (int e = start; e < end; e++) {
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

                for(int l = 0; l < to_exec_levels; l++){
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
    }

    // create non-exec set import list
    halo_list h_list = (halo_list)xmalloc(sizeof(halo_list_core));
    create_nonexec_ex_import_list(set, set_list, h_list, s_i, comm_size, my_rank);
    op_free(set_list); // free temp list
    OP_aug_import_nonexec_lists[el][set->index] = h_list;
  }
}

// Exchange non-exec data
void exchange_nonexec_data(int exec_level, int **part_range, int my_rank, int comm_size){

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];

    if(is_set_required_for_calc(set, exec_level) != 1){
      continue;
    }

    halo_list i_list = OP_aug_import_nonexec_lists[exec_level][set->index];
    halo_list e_list = OP_aug_export_nonexec_lists[exec_level][set->index];

    if(!i_list || !e_list)
      continue;

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

        int prev_exec_size = 0;
        int prev_nonexec_size = 0;
        for(int i = 0; i <= exec_level; i++){
          halo_list prev_h_list = OP_aug_import_exec_lists[i][set->index];
          prev_exec_size += prev_h_list->size;
        }
        for(int i = 0; i < exec_level; i++){
          halo_list prev_h_list = OP_aug_import_nonexec_lists[i][set->index];
          prev_nonexec_size += (prev_h_list) ? prev_h_list->size : 0;
        }

        // prepare space for the incomming nonexec-data - realloc each
        // data array in each mpi process
        dat->data = (char *)xrealloc(
            dat->data,
            (set->size + prev_exec_size + prev_nonexec_size + i_list->size) * dat->size);

        int init = (set->size + prev_exec_size + prev_nonexec_size) * dat->size;
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


// Create halo exchange buffers and set dirty bits
void create_mpi_send_buffers(int exec_levels, int **part_range, int my_rank, int comm_size){
  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    op_set set = dat->set;

    int max_calc_nhalos = set->halo_info->max_calc_nhalos;

    op_mpi_buffer mpi_buf = (op_mpi_buffer)xmalloc(sizeof(op_mpi_buffer_core));

    int exec_e_list_size = 0;
    int exec_e_ranks_size = 0;
    int exec_i_ranks_size = 0;
    int imp_exec_size = 0;

    int nonexec_e_list_size = 0;
    int nonexec_e_ranks_size = 0;
    int nonexec_i_ranks_size = 0;
    int imp_nonexec_size = 0;

    for(int l = 0; l < max_calc_nhalos; l++){
      if(OP_aug_export_exec_lists[l][dat->set->index]){
        exec_e_list_size += OP_aug_export_exec_lists[l][dat->set->index]->size;
        exec_e_ranks_size += OP_aug_export_exec_lists[l][dat->set->index]->ranks_size / 
        OP_aug_export_exec_lists[l][dat->set->index]->num_levels;
      }
      if(OP_aug_import_exec_lists[l][dat->set->index]){
        exec_i_ranks_size += OP_aug_import_exec_lists[l][dat->set->index]->ranks_size / 
        OP_aug_import_exec_lists[l][dat->set->index]->num_levels;
        imp_exec_size += OP_aug_import_exec_lists[l][dat->set->index]->size;
      }

      if(OP_aug_export_nonexec_lists[l][dat->set->index]){
        nonexec_e_list_size += OP_aug_export_nonexec_lists[l][dat->set->index]->size;
        nonexec_e_ranks_size += OP_aug_export_nonexec_lists[l][dat->set->index]->ranks_size / 
        OP_aug_export_nonexec_lists[l][dat->set->index]->num_levels;
      }
      if(OP_aug_import_exec_lists[l][dat->set->index]){
        nonexec_i_ranks_size += OP_aug_import_nonexec_lists[l][dat->set->index]->ranks_size / 
        OP_aug_import_nonexec_lists[l][dat->set->index]->num_levels;
        imp_nonexec_size += OP_aug_import_nonexec_lists[l][dat->set->index]->size;
      }

    }

    mpi_buf->buf_exec = (char *)xmalloc(exec_e_list_size * dat->size);
    mpi_buf->buf_nonexec = (char *)xmalloc(nonexec_e_list_size * dat->size);


    halo_list nonexec_i_list = OP_import_nonexec_list[dat->set->index];

    mpi_buf->s_req = (MPI_Request *)xmalloc(
        sizeof(MPI_Request) *
        (exec_e_ranks_size + nonexec_e_ranks_size));
    mpi_buf->r_req = (MPI_Request *)xmalloc(
        sizeof(MPI_Request) *
        (exec_i_ranks_size + nonexec_i_ranks_size));

    mpi_buf->s_num_req = 0;
    mpi_buf->r_num_req = 0;
    dat->mpi_buffer = mpi_buf;
  }

  // set dirty bits of all data arrays to 0
  // for each data array
  item = NULL;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    dat->dirtybit = 0;

    for(int i = 0; i < dat->set->halo_info->max_calc_nhalos; i++){
      dat->exec_dirtybits[i] = dat->dirtybit;
      if(is_halo_required_for_set(dat->set, i) == 1){
        dat->nonexec_dirtybits[i] = dat->dirtybit;
      }
      // else{
      //   dat->nonexec_dirtybits[i] = -1;
      // }
    }
  }
}


// Separate core elements
void separate_core_elements(int dummy, int **part_range, int **core_elems, int **exp_elems, int my_rank, int comm_size){

  int*** temp_core_elems = (int***) xmalloc(OP_set_index *  sizeof(int**));
  int*** temp_exp_elems = (int***) xmalloc(OP_set_index *  sizeof(int**));
 
  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];

    int num_levels = set->halo_info->nhalos_count;
    int max_level = set->halo_info->max_nhalos;

    temp_core_elems[set->index] = (int **)xmalloc(max_level * sizeof(int*));
    temp_exp_elems[set->index] = (int **)xmalloc(max_level * sizeof(int*));

    for(int el = 0; el < max_level; el++){

      temp_core_elems[set->index][el] = (int *)xmalloc(set->size * sizeof(int));
      int exec_levels = el + 1; //set->halo_info->nhalos[el];

      int exec_size = 0;
      halo_list exec[exec_levels];
      for(int l = 0; l < exec_levels; l++){
        exec[l] = OP_aug_export_exec_lists[l][set->index];
        if(exec[l])
          exec_size += exec[l]->size;
      }

      temp_exp_elems[set->index][el] = (int *)xmalloc(exec_size * sizeof(int));
      if (exec_size > 0) {
        
        int prev_exec_size = 0;
        for(int l = 0; l < exec_levels; l++){
          if(exec[l]){
            memcpy(&temp_exp_elems[set->index][el][prev_exec_size], exec[l]->list, exec[l]->size * sizeof(int));
            prev_exec_size += exec[l]->size;
          }
        }
        quickSort(temp_exp_elems[set->index][el], 0, exec_size - 1);
        int num_exp = removeDups(temp_exp_elems[set->index][el], exec_size);
        
        int count = 0;
        for (int e = 0; e < set->size; e++) { // for each elment of this set

          if ((binary_search(temp_exp_elems[set->index][el], e, 0, num_exp - 1) < 0)) {
            temp_core_elems[set->index][el][count++] = e;
          }
        }
        if(count > 0)
          quickSort(temp_core_elems[set->index][el], 0, count - 1);

        if (count + num_exp != set->size)
          printf("sizes not equal\n");
        
        if(exec_levels == DEFAULT_HALO_COUNT){
          set->core_size = count;
        }
        set->core_sizes[el] = count;
      } else {
        temp_core_elems[set->index][el] = (int *)xmalloc(set->size * sizeof(int));
        temp_exp_elems[set->index][el] = (int *)xmalloc(0 * sizeof(int));
        for (int e = 0; e < set->size; e++) { // for each elment of this set
          temp_core_elems[set->index][el][e] = e;
        }
        if(exec_levels == DEFAULT_HALO_COUNT){
          set->core_size = set->size;
        }
        set->core_sizes[el] = set->size;
      }
    }
    
    //create core_elems and exp_elems arrays based on the temp arrays
    core_elems[set->index] = (int *)xmalloc(set->size * sizeof(int));

    int min_core_size = INT_MAX;
    int min_level = -1;

    int max_core_size = -1;
    int max_core_level = -1;
    for(int l = 0; l < max_level; l++){
      if(min_core_size > set->core_sizes[l]){ // = not added to take the first occurance
        min_core_size = set->core_sizes[l];
        min_level = l;
      }
      if(max_core_size < set->core_sizes[l]){
        max_core_size = set->core_sizes[l];
        max_core_level = l;
      }
    }

    int start = 0;
    for(int l = max_level - 1; l >= 0; l--){
      int size = remove_elements_from_array(set, my_rank, start, &core_elems[set->index][start],
      temp_core_elems[set->index][l], set->core_sizes[l],
      (l == max_level - 1) ? NULL : temp_core_elems[set->index][l + 1], 
      (l == max_level - 1) ? 0 : set->core_sizes[l + 1]);

       start = set->core_sizes[l];
    }
    int exec_size = set->size - max_core_size;
    exp_elems[set->index] = (int *)xmalloc(exec_size * sizeof(int));

    int num_exp = 0;
    int found = -1;
    if(max_core_size >= 0){
      for (int e = 0; e < set->size; e++) { // for each elment of this set
        found = -1;
        for(int el = max_level - 1; el >= 0; el--){
          found = binary_search(core_elems[set->index], e, (el == max_level - 1) ? 0 : set->core_sizes[el + 1], set->core_sizes[el] - 1);
          if(found >= 0){
            break;
          }
        }
        if (found < 0) {
          if(num_exp > exec_size){
            printf("ERROR temp num_exp > exec_size my_rank=%d set=%s num_exp=%d exec_size=%d max_core_size=%d\n", 
            my_rank, set->name, num_exp, exec_size, max_core_size);
            exit(0);
          }
          exp_elems[set->index][num_exp++] = e;
        }
      }
      if(exec_size > 0)
        quickSort(exp_elems[set->index], 0, exec_size - 1);
    }
    // char name[50];
    // int pos = 0;
    // pos += sprintf(&name[pos], "exparr_%s", set->name);
    // pos += sprintf(&name[pos], "_%d", my_rank);
    // print_array(exp_elems[set->index], num_exp, name, my_rank);

    for(int el = 0; el < max_level; el++){
      op_free(temp_core_elems[set->index][el]);
      op_free(temp_exp_elems[set->index][el]);
    }
    op_free(temp_core_elems[set->index]);
    op_free(temp_exp_elems[set->index]);
  }

  for (int s = 0; s < OP_set_index; s++) { // for each set
    op_set set = OP_set_list[s];

    int num_levels = set->halo_info->nhalos_count;
    int max_level = set->halo_info->max_nhalos;

    int min_core_size = INT_MAX;
    int min_level = -1;

    int max_core_size = -1;
    int max_core_level = -1;
    for(int l = 0; l < max_level; l++){
      if(min_core_size > set->core_sizes[l]){
        min_core_size = set->core_sizes[l];
        min_level = l;
      }
      if(max_core_size < set->core_sizes[l]){
        max_core_size = set->core_sizes[l];
        max_core_level = l;
      }
    }

    int count = max_core_size;
    int num_exp = set->size - count;
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
        //for the standard op2 maps
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
        
        //for aug maps
        for(int el = 0; el < map->halo_info->max_calc_nhalos; el++){
          if(is_halo_required_for_map(map, el) != 1 || is_map_required_for_calc(map, el) != 1){
            continue;
          }
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
      }
    }

    int max_calc_level = set->halo_info->max_calc_nhalos;

    halo_list exec[max_calc_level];
    for(int l = 0; l < max_calc_level; l++){
      exec[l] = OP_aug_export_exec_lists[l][set->index];
      if(!exec[l])
        continue;
      for (int i = 0; i < exec[l]->size; i++) {
        int index =
            binary_search(exp_elems[set->index], exec[l]->list[i], 0, num_exp - 1);
        if (index < 0){ //this element can be in the core list of another
          int core_index = -1;
          int start_index = 0;
          for(int l1 = max_level - 1; l1 >= 0; l1--){
            start_index = (l1 == max_level - 1) ? 0 : set->core_sizes[l1 + 1];
            core_index = binary_search(core_elems[set->index], exec[l]->list[i], start_index, set->core_sizes[l1] - 1);

            if(core_index >= 0){
              break;
            }
          }
          if (core_index < 0){
            printf("Problem in seperating core elements - exec list my_rank=%d set=%s val=%d core=%d index=%d count=%d exp=%d\n", 
            my_rank, set->name, exec[l]->list[i], core_index, index, count, num_exp);
            // char name[50];
            // int pos = 0;
            // pos += sprintf(&name[pos], "core_%s", set->name);
            // pos += sprintf(&name[pos], "_%d", my_rank);
            // print_array(exp_elems[set->index], num_exp, name, my_rank);
          }else{
            exec[l]->list[i] = core_index;
          }
        }
        else{
          exec[l]->list[i] = count + index;
        }
          
      }
    }

    for(int l = 0; l < max_calc_level; l++){
      halo_list nonexec[max_calc_level];
      nonexec[l] = OP_aug_export_nonexec_lists[l][set->index];
      if(!nonexec[l]){
        continue;
      }
      for (int i = 0; i < nonexec[l]->size; i++) {
        int index =
            binary_search(exp_elems[set->index], nonexec[l]->list[i], 0, num_exp - 1);
        if (index < 0){ //this element can be in the core list of another
          int core_index = -1;
          int start_index = 0;
          for(int l1 = max_level - 1; l1 >= 0; l1--){
            start_index = (l1 == max_level - 1) ? 0 : set->core_sizes[l1 + 1];
            core_index = binary_search(core_elems[set->index], nonexec[l]->list[i], start_index, set->core_sizes[l1] - 1);
            if(core_index >= 0){
              break;
            }
          }
          if (core_index < 0){
            printf("Problem in seperating core elements - nonexec list set=%s val=%d core=%d index=%d count=%d exp=%d\n", 
            set->name, nonexec[l]->list[i], core_index, index, count, num_exp);
          }else{
            nonexec[l]->list[i] = core_index;
          }
        }
        else
          nonexec[l]->list[i] = count + index;
      }
    }
  }

  // now need to renumber mapping tables as the elements are seperated
  for (int m = 0; m < OP_map_index; m++) { // for each set
    op_map map = OP_map_list[m];

    int num_levels = map->halo_info->nhalos_count;  // = to from set levels
    int max_level = map->halo_info->max_nhalos;

    int to_max_level = map->to->halo_info->max_nhalos;

    for(int el = 0; el < max_level; el++){

      if(is_halo_required_for_map(map, el) != 1 || is_map_required_for_calc(map, el) != 1)
        continue;

      int exec_levels =  el + 1; //map->from->halo_info->nhalos[el];
      int imp_exec_size = 0;

      int start = 0;
      int end = 0;
      int current_nonexec_size = 0;
      for(int sl = 0; sl < exec_levels; sl++){

        int imp_exec_size = 0;
        int imp_nonexec_size = 0;
        for(int l = 0; l < sl + 1; l++){
          imp_exec_size +=  OP_aug_import_exec_lists[l][map->from->index] ? 
            OP_aug_import_exec_lists[l][map->from->index]->size : 0;
          if(l > 0){
            imp_nonexec_size += (OP_aug_import_nonexec_lists[l - 1][map->from->index])?
                              OP_aug_import_nonexec_lists[l - 1][map->from->index]->size : 0;
          }
        }

        start = end + current_nonexec_size;
        end = map->from->size + imp_exec_size + imp_nonexec_size;
        current_nonexec_size = (OP_aug_import_nonexec_lists[sl][map->from->index])?
                              OP_aug_import_nonexec_lists[sl][map->from->index]->size : 0;


        // for each entry in this mapping table: original+execlist
        for (int e = start; e < end; e++) {
          for (int j = 0; j < map->dim; j++) { // for each element pointed
                                              // at by this entry
            if (map->aug_maps[el][e * map->dim + j] < map->to->size) {
              int index =
                binary_search(exp_elems[map->to->index], map->aug_maps[el][e * map->dim + j], 0, 
                (map->to->size) - (map->to->core_sizes[0]) - 1);  //todo: check always take 0th size
              if (index < 0){
                int core_index = -1;
                int start_index = 0;
                for(int l1 = to_max_level - 1; l1 >= 0; l1--){
                  if(is_halo_required_for_set(map->to, l1) != 1)
                    continue;
                  start_index = (l1 == to_max_level - 1) ? 0 : map->to->core_sizes[l1 + 1];
                  core_index = binary_search(core_elems[map->to->index], map->aug_maps[el][e * map->dim + j], start_index, map->to->core_sizes[l1] - 1);
                  if(core_index >= 0){
                    break;
                  }
                }
                if (core_index < 0){
                  printf("Problem in seperating core elements - renumbering map list augmap set=%s val=%d core=%d index=%d count=%d exp=%d\n", 
                  map->to->name, map->aug_maps[el][e * map->dim + j], core_index, index, map->to->core_sizes[el], (map->to->size) - (map->to->core_sizes[0]));
                }else{
                  if(exec_levels == DEFAULT_HALO_COUNT){
                    OP_map_list[map->index]->map[e * map->dim + j] = core_index;
                  }
                  OP_map_list[map->index]->aug_maps[el][e * map->dim + j] = core_index;       
                }
              }
              else{
                if(exec_levels == DEFAULT_HALO_COUNT){
                    OP_map_list[map->index]->map[e * map->dim + j] = map->to->core_sizes[0] + index;
                  }
                  OP_map_list[map->index]->aug_maps[el][e * map->dim + j] =
                      map->to->core_sizes[0] + index; //todo: check always take 0th size      
              } 
            }
          }
        }
      }
    }
  }
  op_free(temp_core_elems);
  op_free(temp_exp_elems);
 
}

// Rearrange data elements according to core separation
void rearrange_data_elements(int exec_levels, int **part_range, int **core_elems, int **exp_elems, int my_rank, int comm_size){
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
}

void compute_halo_numbers(int **part_range, double max_time, int my_rank, int comm_size){
  step12(part_range, max_time, my_rank, comm_size);
}



/*******************************************************************************
 * Main MPI halo creation routine
 *******************************************************************************/

void op_halo_create_comm_avoid() {
  // op_printf("op_halo_create_comm_avoid start <<<<<<<<<<< new >>>>>>>\n");

  // declare timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  double time;
  double max_time;

  // Init arrays
  for(int i = 0; i < 10; i++){
    OP_aug_export_exec_lists[i] = NULL;
    OP_aug_import_exec_lists[i] = NULL;

    OP_aug_export_nonexec_lists[i] = NULL;
    OP_aug_import_nonexec_lists[i] = NULL;
  }

  // Setup hydra maps
  set_maps_hydra();

  // Create new communicator for OP mpi operation
  int my_rank, comm_size;
  MPI_Comm_rank(OP_MPI_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_WORLD, &comm_size);

  /* Compute global partition range information for each set*/
  int **part_range = (int **)xmalloc(OP_set_index * sizeof(int *));
  get_part_range(part_range, my_rank, comm_size, OP_MPI_WORLD);

  // Save this partition range information if it is not already saved during
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

  create_part_range_arrays(my_rank, comm_size);
  // Elem rank matrix is not used due to high memory consumption, though fast in searching
  // create_elem_rank_matrix(my_rank, comm_size);

  op_timers(&cpu_t1, &wall_t1); // timer start for list create

  int num_halos = get_max_nhalos();
  prepare_aug_sets();
  
  for(int l = 0; l < num_halos; l++){
    // STEP 1 - Construct export lists for execute set elements and related  mapping table entries
    OP_aug_export_exec_lists[l] = create_export_exec_list(l, part_range, my_rank, comm_size);

    // STEP 2 - Construct import lists for mappings and execute sets
    OP_aug_import_exec_lists[l] = create_handshake_h_list(OP_aug_export_exec_lists[l], part_range, my_rank, comm_size);

    // STEP 3 - Exchange mapping table entries using the import/export lists
    exchange_exec_mappings(l, part_range, my_rank, comm_size);

    // STEP 4 - Construct import lists for mappings and non-execute sets
    create_import_nonexec_list(l, part_range, my_rank, comm_size);

    // STEP 5 - Construct export lists for mappings and non-execute sets
    OP_aug_export_nonexec_lists[l] = create_handshake_h_list(OP_aug_import_nonexec_lists[l], part_range, my_rank, comm_size);

    // STEP 6 - Exchange execute set elements/data using the import/export lists
    exchange_exec_data(l, part_range, my_rank, comm_size);
    
    // STEP 7 - Exchange non-execute set elements/data using the import/export lists
    exchange_nonexec_data(l, part_range, my_rank, comm_size);

    // STEP 8 - Calculate set matics for the generated halo layer
    calc_set_metrics(l);
    
    // STEP 9 - Create and exchange augmented partition ranges
    if(l < num_halos - 1){
      create_aug_part_range(l, part_range, my_rank, comm_size);
      exchange_aug_part_ranges(l, part_range, my_rank, comm_size);
    }

    // Release memory of unused halo lists
    for (int s = 0; s < OP_set_index; s++) { // for each set
      op_set set = OP_set_list[s];

      if(is_set_required_for_calc(set, l) == 1){
        continue;
      }
      
      op_single_halo_destroy(OP_aug_import_exec_lists[l][set->index]); OP_aug_import_exec_lists[l][set->index] = NULL;
      op_single_halo_destroy(OP_aug_import_nonexec_lists[l][set->index]); OP_aug_import_nonexec_lists[l][set->index] = NULL;
      // export exec halo lists are required to check whether an element was sent for the previous halo layers. So memory is not 
      // released for export exec halo lists. This is due to removing the elem rank matrix due to its high memory consumption.
      // op_single_halo_destroy(OP_aug_export_exec_lists[l][set->index]); OP_aug_export_exec_lists[l][set->index] = NULL;
      op_single_halo_destroy(OP_aug_export_nonexec_lists[l][set->index]); OP_aug_export_nonexec_lists[l][set->index] = NULL;
    }
  }

  free_part_range_arrays(my_rank, comm_size);
  // free_elem_rank_matrix(my_rank, comm_size);

  // This is required for sttandard OP2 operation
  OP_export_exec_list = OP_aug_export_exec_lists[0];
  OP_import_exec_list = OP_aug_import_exec_lists[0];

  OP_export_nonexec_list = OP_aug_export_nonexec_lists[0];
  OP_import_nonexec_list = OP_aug_import_nonexec_lists[0];

  // STEP 10 - Renumber Mapping tables
  renumber_mappings(num_halos, part_range, my_rank, comm_size);

  // STEP 11 - Create MPI send Buffers
  create_mpi_send_buffers(num_halos, part_range, my_rank, comm_size); 

  // STEP 12 - Separate core elements
  int **core_elems = (int **)xmalloc(OP_set_index * sizeof(int *));
  int **exp_elems = (int **)xmalloc(OP_set_index * sizeof(int *));
  separate_core_elements(num_halos, part_range, core_elems, exp_elems, my_rank, comm_size);

  free_tmp_maps();

  // STEP 13 - Save the original set element indexes
  rearrange_data_elements(num_halos, part_range, core_elems, exp_elems, my_rank, comm_size);
  // Merging halos required for additional halos halo exchange as a grouped message
  merge_exec_nonexec_halos(num_halos, my_rank, comm_size);
  set_group_halo_envt();

  op_timers(&cpu_t2, &wall_t2); // timer stop for list create
  // compute import/export lists creation time
  time = wall_t2 - wall_t1;

  // This is for message sizes and other information for performance model
  calculate_dat_sizes(my_rank);
  calculate_set_sizes(my_rank);
  
  // STEP 14 - Clean up and Compute rough halo size numbers
  for (int i = 0; i < OP_set_index; i++) {
    op_free(part_range[i]);
    op_free(core_elems[i]);
    op_free(exp_elems[i]);
  }
  op_free(part_range);
  op_free(exp_elems);
  op_free(core_elems);

  MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_ROOT, OP_MPI_WORLD);
  compute_halo_numbers(part_range, max_time, my_rank, comm_size);
}


// Memory release when op_exit
void op_halo_destroy_comm_avoid() {
  for(int i = 0; i < OP_NHALOS_MAX; i++){
    if(OP_aug_import_exec_lists[i])
      op_halos_destroy(OP_aug_import_exec_lists[i]);
    if(OP_aug_export_exec_lists[i])
      op_halos_destroy(OP_aug_export_exec_lists[i]);
    if(OP_aug_import_nonexec_lists[i])
      op_halos_destroy(OP_aug_import_nonexec_lists[i]);
    if(OP_aug_export_nonexec_lists[i])
      op_halos_destroy(OP_aug_export_nonexec_lists[i]);
  }
  op_halos_destroy(OP_merged_import_exec_nonexec_list);
  op_halos_destroy(OP_merged_export_exec_nonexec_list);
}

// Memory release when op_exit
void op_mpi_exit_comm_avoid() {
  op_mpi_destroy();
  op_halo_destroy_comm_avoid();
}
