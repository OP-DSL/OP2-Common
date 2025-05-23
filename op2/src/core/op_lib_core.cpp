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
 * This file implements the OP2 core library functions used by *any*
 * OP2 implementation
 */

#include "op_lib_core.h"
#include <malloc.h>
#include <string.h>
#include <sys/time.h>
#include <signal.h>

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <regex>
#include <string>
#include <fstream>

/*
 * OP2 global state variables
 */

int OP_diags = 0, OP_part_size = 0, OP_block_size = 64,
    OP_cache_line_size = 128, OP_gpu_direct = 0;

double OP_hybrid_balance = 1.0;
int OP_hybrid_gpu = 0;
int OP_auto_soa = 0;
int OP_maps_base_index = 0;
int OP_realloc = 1;

int OP_set_index = 0, OP_set_max = 0, OP_map_index = 0, OP_map_max = 0,
    OP_dat_index = 0, OP_kern_max = 0, OP_kern_curr = 0;

int OP_mpi_test_frequency = 1<<30;
int OP_partial_exchange = 0;
int OP_is_partitioned = 0;

int OP_disable_mpi_reductions = 0;
int OP_cuda_reductions_mib = 5;

std::vector<std::regex> OP_whitelist = {};
bool OP_disable_device_execution = false;

/*
 * Lists of sets, maps and dats declared in OP2 programs
 */

op_set *OP_set_list;
op_map *OP_map_list = NULL;
std::unordered_map<int*, int> OP_map_ptr_table = {};
Double_linked_list OP_dat_list; /*Head of the double linked list*/
op_kernel *OP_kernels;

const char *doublestr = "double";
const char *floatstr = "float";
const char *intstr = "int";
const char *boolstr = "bool";

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Utility functions
 */
static char *copy_str(char const *src) {
  const size_t len = strlen(src) + 1;
  char *dest = (char *)op_calloc(len, sizeof(char));
  return strncpy(dest, src, len);
}

int compare_sets(op_set set1, op_set set2) {
  if (set1->size == set2->size && set1->index == set2->index &&
      strcmp(set1->name, set2->name) == 0)
    return 1;
  else
    return 0;
}

op_dat op_search_dat(op_set set, int dim, char const *type, int size,
                     char const *name) {
  op_dat_entry *item;
  op_dat_entry *tmp_item;
  for (item = TAILQ_FIRST(&OP_dat_list); item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    op_dat item_dat = item->dat;

    if (strcmp(item_dat->name, name) == 0 && item_dat->dim == dim &&
        (item_dat->size / dim) == size &&
        compare_sets(item_dat->set, set) == 1 &&
        strcmp(item_dat->type, type) == 0) {
      return item_dat;
    }
  }

  return NULL;
}

op_map op_search_map_ptr(int *map_ptr) {
  if (auto result = OP_map_ptr_table.find(map_ptr); result != OP_map_ptr_table.end())
    return OP_map_list[result->second];

  return NULL;
}

/*check if map points to elements within set range*/
void check_map(char const *name, op_set from, op_set to, int dim, int *map) {

  // first find global set sizes
  // int g_from = op_get_size(from);
  int g_to = op_get_size(to);
  // printf("%s from->size = %d (%d)\n",from->name, from->size, g_from);
  // printf("%s to->size = %d (%d)\n",to->name, to->size, g_to);

  for (int d = 0; d < dim; d++) {
    for (int n = 0; n < from->size; n++) {
      if (map[d + n * dim] < 0 || map[d + n * dim] >= g_to) {
        printf("op_decl_map error -- invalid data for map %s\n", name);
        printf("element = %d, dimension = %d, map = %d\n", n, d,
               map[d + n * dim]);
        exit(-1);
      }
    }
  }
}

void op_disable_device_execution(bool disable) {
  OP_disable_device_execution = disable;
}

bool op_check_whitelist(const char *name) {
  if (OP_disable_device_execution)
    return false;

  if (OP_whitelist.size() == 0)
    return true;

  for (const auto& regex : OP_whitelist) {
    if (std::regex_match(name, regex))
      return true;
  }

  return false;
}

void op_disable_mpi_reductions(bool disable) {
  OP_disable_mpi_reductions = disable;
}

/* Special function to get commandline arguments, articularly useful as argv
*  is not easy to pass through from frotran to C
*/
void op_set_args(int argc, char *argv) {

  char temp[64];
  char *pch;
  pch = strstr(argv, "OP_BLOCK_SIZE=");
  if (pch != NULL) {
    strncpy(temp, pch, 20);
    OP_block_size = atoi(temp + 14);
    op_printf("\n OP_block_size = %d \n", OP_block_size);
  }
  pch = strstr(argv, "OP_PART_SIZE=");
  if (pch != NULL) {
    strncpy(temp, pch, 20);
    OP_part_size = atoi(temp + 13);
    op_printf("\n OP_part_size = %d \n", OP_part_size);
  }
  pch = strstr(argv, "OP_NO_REALLOC");
  if (pch != NULL) {
    strncpy(temp, pch, 20);
    OP_realloc = 0;
    op_printf("\n Disable user memory reallocation/internal copy\n");
  }
  pch = strstr(argv, "OP_CACHE_LINE_SIZE=");
  if (pch != NULL) {
    strncpy(temp, pch, 25);
    OP_cache_line_size = atoi(temp + 19);
    op_printf("\n OP_cache_line_size  = %d \n", OP_cache_line_size);
  }
  pch = strstr(argv, "OP_TEST_FREQ=");
  if (pch != NULL) {
    strncpy(temp, pch, 25);
    OP_mpi_test_frequency = atoi(temp + 13);
    op_printf("\n OP_mpi_test_frequency  = %d \n", OP_mpi_test_frequency);
  }
  pch = strstr(argv, "-gpudirect");
  if (pch != NULL) {
    OP_gpu_direct = 1;
    op_printf("\n Enabling GPU Direct\n");
  }
  pch = strstr(argv, "OP_AUTO_SOA");
  if (pch != NULL) {
    OP_auto_soa = 1;
    op_printf("\n Enabling Automatic AoS->SoA Conversion\n");
  }
  pch = strstr(argv, "OP_PARTIAL_EXCHANGE");
  if (pch != NULL) {
    OP_partial_exchange = 1;
    op_printf("\n Enabling partial MPI halo exchanges\n");
  }
  pch = strstr(argv, "OP_HYBRID_BALANCE=");
  if (pch != NULL) {
    strncpy(temp, pch, 25);
    OP_hybrid_balance = atof(temp + 18);
    op_printf("\n OP_hybrid_balance  = %g \n", OP_hybrid_balance);
  }
  pch = strstr(argv, "OP_MAPS_BASE_INDEX=");
  if (pch != NULL) {
    strncpy(temp, pch, 25);
    int prev = OP_maps_base_index;
    OP_maps_base_index = atoi(temp + 19);
    if (OP_maps_base_index == 0 || OP_maps_base_index == 1) {
      if (prev != OP_maps_base_index)
        op_printf("\n OP_maps_base_index  = %d (Default value %d overridden)\n",
                  OP_maps_base_index, prev);
    } else {
      op_printf("\n Unsupported OP_maps_base_index : %d -- \
      reverting to C/C++ style (i.e. 0 besed) indexing \n",
                OP_maps_base_index);
    }
  }
  pch = strstr(argv, "OP_CUDA_REDUCTIONS_MIB=");
  if (pch != NULL) {
    strncpy(temp, pch, 30);
    OP_cuda_reductions_mib = atoi(temp + 23);
    op_printf("\n OP_cuda_reductions_mib  = %d \n", OP_cuda_reductions_mib);
  }
}

/*
 * OP core functions: these must be called by back-end specific functions
 */

void op_init_core(int argc, char **argv, int diags) {
  op_printf("\n Initializing OP2\n");
  OP_diags = diags;

  if (getenv("OP_HYBRID_BALANCE")) {
    char *val = getenv("OP_HYBRID_BALANCE");
    OP_hybrid_balance = atof(val);
    op_printf("\n OP_hybrid_balance  = %g \n", OP_hybrid_balance);
  }

  if (getenv("OP_AUTO_SOA") || OP_auto_soa == 1) {
    OP_auto_soa = 1;
    op_printf("\n Enabling Automatic AoS->SoA Conversion\n");
  }

  if (getenv("OP_WHITELIST")) {
    char *val = getenv("OP_WHITELIST");
    op_printf("\n Using kernel whitelist: %s\n", val);

    std::ifstream file(val);

    if (!file.is_open()) {
      op_printf(" Error: could not open kernel whitelist, skipping\n");
    } else {
      std::string line;

      while (std::getline(file, line)) {
        OP_whitelist.push_back(std::regex(line));
      }
    }
  }

#ifdef OP_BLOCK_SIZE
  OP_block_size = OP_BLOCK_SIZE;
#endif
#ifdef OP_PART_SIZE
  OP_part_size = OP_PART_SIZE;
#endif

  // if OP_maps_base_index has been set to 1 by Fortran backend
  if (OP_maps_base_index)
    op_printf("\n Application written using Fortran API: \
    \n Default OP_maps_base_index = %d assumed\n",
              OP_maps_base_index);

  for (int n = 1; n < argc; n++) {

    op_set_args(argc, argv[n]);

    /*if ( strncmp ( argv[n], "OP_BLOCK_SIZE=", 14 ) == 0 ) {
      OP_block_size = atoi ( argv[n] + 14 );
      op_printf ( "\n OP_block_size = %d \n", OP_block_size );
    }

    if ( strncmp ( argv[n], "OP_PART_SIZE=", 13 ) == 0 ) {
      OP_part_size = atoi ( argv[n] + 13 );
      op_printf ( "\n OP_part_size  = %d \n", OP_part_size );
    }

    if ( strncmp ( argv[n], "OP_CACHE_LINE_SIZE=", 19 ) == 0 ) {
      OP_cache_line_size = atoi ( argv[n] + 19 );
      op_printf ( "\n OP_cache_line_size  = %d \n", OP_cache_line_size );
    }

    if ( strncmp ( argv[n], "-gpudirect", 10 ) == 0 ) {
      OP_gpu_direct = 1;
      op_printf ( "\n Enabling GPU Direct \n" );
    }
    if ( strncmp ( argv[n], "OP_AUTO_SOA", 9 ) == 0 ) {
      OP_auto_soa = 1;
      op_printf ( "\n Enabling Automatic AoS->SoA Conversion\n" );
    }
    if ( strncmp ( argv[n], "OP_HYBRID_BALANCE=", 18 ) == 0 ) {
      OP_hybrid_balance = atof ( argv[n] + 18 );;
      op_printf ( "\n OP_hybrid_balance  = %g \n", OP_hybrid_balance );
    }*/
  }

  /*Initialize the double linked list to hold op_dats*/
  TAILQ_INIT(&OP_dat_list);
}

op_set op_decl_set_core(int size, char const *name) {
  if (size < 0) {
    printf(" op_decl_set error -- negative/zero size for set: %s\n", name);
    exit(-1);
  }

  if (OP_set_index == OP_set_max) {
    OP_set_max += 10;
    OP_set_list =
        (op_set *)op_realloc(OP_set_list, OP_set_max * sizeof(op_set));

    if (OP_set_list == NULL) {
      printf(" op_decl_set error -- error reallocating memory\n");
      exit(-1);
    }
  }

  op_set set = (op_set)op_malloc(sizeof(op_set_core));
  set->index = OP_set_index;
  set->size = size;
  set->core_size = size;
  set->name = copy_str(name);
  set->exec_size = 0;
  set->nonexec_size = 0;
  OP_set_list[OP_set_index++] = set;

  return set;
}

op_map op_decl_map_core(op_set from, op_set to, int dim, int *imap,
                        char const *name) {
  if (from == NULL) {
    printf(" op_decl_map error -- invalid 'from' set for map %s\n", name);
    exit(-1);
  }

  if (to == NULL) {
    printf("op_decl_map error -- invalid 'to' set for map %s\n", name);
    exit(-1);
  }

  if (dim <= 0) {
    printf("op_decl_map error -- negative/zero dimension for map %s\n", name);
    exit(-1);
  }

  // check if map points to elements within set range
  // check_map(name, from, to, dim, imap);

  /*This check breaks for MPI - check_map() above does the required check now */
  /*for ( int d = 0; d < dim; d++ ) {
    for ( int n = 0; n < from->size; n++ ) {
      if ( imap[d + n * dim] < 0 || imap[d + n * dim] >= to->size ) {
        printf ( "op_decl_map error -- invalid data for map %s\n", name );
        printf ( "element = %d, dimension = %d, map = %d\n", n, d, imap[d + n *
  dim] );
        exit ( -1 );
      }
    }
  }*/

  int *m = (int *)malloc((size_t)from->size * (size_t)dim * sizeof(int));
  if (m == NULL) {
    printf(" op_decl_map_core error -- error allocating memory to map\n");
    exit(-1);
  }
  memcpy(m, imap, sizeof(int) * from->size * dim);

  if (OP_map_index == OP_map_max) {
    OP_map_max += 10;
    OP_map_list = (op_map *)op_realloc(OP_map_list, OP_map_max * sizeof(op_map));

    if (OP_map_list == NULL) {
      printf(" op_decl_map error -- error reallocating memory\n");
      exit(-1);
    }
  }

  if (OP_maps_base_index == 1) {
    // convert imap to 0 based indexing -- i.e. reduce each imap value by 1
    for (int i = 0; i < from->size * dim; i++)
      // imap[i]--;
      m[i]--; // modify op2's copy
  }
  // else OP_maps_base_index == 0
  // do nothing -- aready in C style indexing

  op_map map = (op_map)op_malloc(sizeof(op_map_core));
  map->index = OP_map_index;
  map->from = from;
  map->to = to;
  map->dim = dim;
  map->map = m; // use op2's copy instead of imap;
  map->map_d = NULL;
  map->name = copy_str(name);
  map->user_managed = 1;
  map->force_part = false;

  OP_map_list[OP_map_index] = map;
  OP_map_ptr_table.insert({imap, OP_map_index});

  OP_map_index++;
  return map;
}

op_dat op_decl_dat_core(op_set set, int dim, char const *type, int size,
                        char *data, char const *name) {
  if (set == NULL) {
    printf("op_decl_dat error -- invalid set for data: %s\n", name);
    exit(-1);
  }

  if (dim <= 0) {
    printf("op_decl_dat error -- negative/zero dimension for data: %s\n", name);
    exit(-1);
  }

  op_dat dat = (op_dat)op_malloc(sizeof(op_dat_core));
  dat->index = OP_dat_index;
  dat->set = set;
  dat->dim = dim;
  size_t bytes = (size_t)dim * (size_t)size * (size_t)
                 (set->size+set->exec_size+set->nonexec_size) * sizeof(char);

  if (OP_realloc) {
    char *new_data = (char *)op_calloc(bytes, sizeof(char));
    if (data != NULL)
      memcpy(new_data, data, (size_t)dim * (size_t)size * (size_t)set->size * sizeof(char));

    dat->data = new_data;
    dat->user_managed = 0;
  } else {
    if (data != NULL) {
      dat->data = data;
      dat->user_managed = 1;
    } else {
      char *new_data = (char *)op_calloc(bytes, sizeof(char));
      dat->data = new_data;
      dat->user_managed = 0;
    }
  }

  dat->data_d = NULL;
  dat->name = copy_str(name);
  dat->type = copy_str(type);
  dat->size = dim * size;
  dat->mpi_buffer = NULL;
  dat->buffer_d = NULL;
  dat->buffer_d_r = NULL;
  dat->dirty_hd = 0;
  dat->dirtybit = 1;

  /* Create a pointer to an item in the op_dats doubly linked list */
  op_dat_entry *item;

  // add the newly created op_dat to list
  item = (op_dat_entry *)op_malloc(sizeof(op_dat_entry));
  if (item == NULL) {
    printf(" op_decl_dat error -- error allocating memory to double linked "
           "list entry\n");
    exit(-1);
  }
  item->dat = dat;

  if (data != NULL) {
    item->orig_ptr = data;
  } else {
    item->orig_ptr = dat->data;
  }

  if (TAILQ_EMPTY(&OP_dat_list)) {
    TAILQ_INSERT_HEAD(&OP_dat_list, item, entries);
  } else {
    TAILQ_INSERT_TAIL(&OP_dat_list, item, entries);
  }

  OP_dat_index++;

  return dat;
}

/*
 * overlay dats
 */
op_dat op_decl_dat_overlay_core(op_set set, op_dat dat) {
  if (set == NULL) {
    printf("op_decl_dat_overlay error -- invalid set for data: %s\n", dat->name);
    exit(-1);
  }

  size_t orig_set_size = dat->set->size + dat->set->exec_size + dat->set->nonexec_size;
  size_t new_set_size = set->size + set->exec_size + set->nonexec_size;

  if (OP_diags > 1) {
    op_printf("op_decl_dat_overlay: dat: %s, set %s (%lu elems) -> %s (%lu elems)\n",
            dat->name, dat->set->name, orig_set_size, set->name, new_set_size);
  }

  if (new_set_size > orig_set_size) {
    printf("op_decl_dat_overlay error -- new set size bigger than overlayed size (%lu > %lu)\n",
            new_set_size, orig_set_size);
    exit(-1);
  }

  op_dat overlay_dat = (op_dat)op_malloc(sizeof(op_dat_core));
  overlay_dat->index = OP_dat_index;

  overlay_dat->set = set;
  overlay_dat->dim = dat->dim;
  overlay_dat->data = dat->data;
  overlay_dat->data_d = dat->data_d;
  overlay_dat->name = dat->name;
  overlay_dat->type = dat->type;
  overlay_dat->size = dat->size;
  overlay_dat->user_managed = dat->user_managed;
  overlay_dat->mpi_buffer = dat->mpi_buffer;
  overlay_dat->buffer_d = dat->buffer_d;
  overlay_dat->buffer_d_r = dat->buffer_d_r;
  overlay_dat->dirty_hd = dat->dirty_hd;
  overlay_dat->dirtybit = dat->dirtybit;

  op_dat_entry *item = (op_dat_entry *)op_malloc(sizeof(op_dat_entry));
  if (item == NULL) {
    printf(" op_decl_dat_overlay error -- error allocating memory to double linked "
           "list entry\n");
    exit(-1);
  }

  item->dat = overlay_dat;
  item->orig_ptr = (char *)(unsigned long)overlay_dat->index;

  if (TAILQ_EMPTY(&OP_dat_list)) {
    TAILQ_INSERT_HEAD(&OP_dat_list, item, entries);
  } else {
    TAILQ_INSERT_TAIL(&OP_dat_list, item, entries);
  }

  OP_dat_index++;

  return overlay_dat;
}

/*
 * temporary dats
 */

op_dat op_decl_dat_temp_core(op_set set, int dim, char const *type, int size,
                             char *data, char const *name) {
  // Check if this dat already exists in the double linked list
  op_dat found_dat = op_search_dat(set, dim, type, size, name);
  if (found_dat != NULL) {
    printf(
        "op_dat with name %s already exists, cannot create temporary op_dat\n ",
        name);
    exit(2);
  }
  // if not found ...
  return op_decl_dat_core(set, dim, type, size, data, name);
}

int op_free_dat_temp_core(op_dat dat) {
  int success = -1;
  op_dat_entry *item;
  op_dat_entry *tmp_item;
  for (item = TAILQ_FIRST(&OP_dat_list); item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    op_dat item_dat = item->dat;
    if (strcmp(item_dat->name, dat->name) == 0 && item_dat->dim == dat->dim &&
        item_dat->size == dat->size &&
        compare_sets(item_dat->set, dat->set) == 1 &&
        strcmp(item_dat->type, dat->type) == 0) {
      if (!(item->dat)->user_managed)
        free((item->dat)->data);
      free((char *)(item->dat)->name);
      free((char *)(item->dat)->type);
      TAILQ_REMOVE(&OP_dat_list, item, entries);
      free(item->dat);
      free(item);
      success = 1;
      break;
    }
  }
  return success;
}

void op_decl_const_core(int dim, char const *type, int typeSize, char *data,
                        char const *name) {
  (void)dim;
  (void)type;
  (void)typeSize;
  (void)data;
  (void)name;
}

void op_exit_core() {
  // free storage and pointers for sets, maps and data

  for (int i = 0; i < OP_set_index; i++) {
    free((char *)OP_set_list[i]->name);
    free(OP_set_list[i]);
  }
  free(OP_set_list);
  OP_set_list = NULL;

  for (int i = 0; i < OP_map_index; i++) {
    if (!OP_map_list[i]->user_managed)
      free(OP_map_list[i]->map);

    free((char *)OP_map_list[i]->name);
    free(OP_map_list[i]);
  }

  free(OP_map_list);
  OP_map_list = NULL;

  OP_map_ptr_table = {};

  /*free doubl linked list holding the op_dats */
  op_dat_entry *item;
  while ((item = TAILQ_FIRST(&OP_dat_list))) {
    if (!(item->dat)->user_managed)
      free((item->dat)->data);
    free((char *)(item->dat)->name);
    free((char *)(item->dat)->type);
    TAILQ_REMOVE(&OP_dat_list, item, entries);
    free(item->dat);
    free(item);
  }

  // free storage for timing info

  free(OP_kernels);
  OP_kernels = NULL;

  // reset initial values

  OP_set_index = 0;
  OP_set_max = 0;
  OP_map_index = 0;
  OP_map_max = 0;
  OP_dat_index = 0;
  OP_kern_max = 0;
}

/*
 * op_arg routines
 */

void op_err_print(const char *error_string, int m, const char *name) {
  printf("error: arg %d in kernel \"%s\"\n", m, name);
  printf("%s \n", error_string);
  //  exit ( 1 );
}

void op_arg_check(op_set set, int m, op_arg arg, int *ninds, const char *name) {
  /* error checking for op_arg_dat */
  if (arg.opt == 0)
    return;
  if (arg.argtype == OP_ARG_DAT) {
    if (set == NULL)
      op_err_print("invalid set", m, name);

    if (arg.map == NULL && arg.dat->set != set) {
      // op_err_print("dataset set does not match loop set", m, name);
      if (arg.dat->set != set)
        printf("dataset dat %s with data pointer %lu (%p), on set %p (%s) does not "
               "match loop set %p (%s)\n",
               arg.dat->name, arg.dat->data, arg.dat->data, arg.dat->set, arg.dat->set->name,
               set, set->name);
    }

    if (arg.map != NULL) {
      if (arg.map->from != set) {
        op_err_print("mapping error", m, name);
        printf("map from set %s does not match set \n", arg.map->from->name);
      }
      if (arg.map->to != arg.dat->set) {
        op_err_print("mapping error", m, name);
        printf("map %s to set %s does not match dat %s set %s\n", arg.map->name,
               arg.map->to->name, arg.dat->name, arg.dat->set->name);
      }
    }

    if ((arg.map == NULL && arg.idx != -1) ||
        (arg.map != NULL &&
         (arg.idx >= arg.map->dim || arg.idx < -1 * arg.map->dim)))
      op_err_print("invalid index", m, name);

    if (arg.dat->dim != arg.dim) {
      op_err_print("dataset dim does not match declared dim", m, name);
    }

    if (strcmp(arg.dat->type, arg.type)) {
      if ((strcmp(arg.dat->type, "double") == 0 && strcmp(arg.type, "real(RK8)")==0) ||
          (strcmp(arg.dat->type, "double:soa") == 0 &&
           (int)arg.type[0] == 114 && (int)arg.type[1] == 56 &&
           (int)arg.type[2] == 58) ||
          (strcmp(arg.dat->type, "int") == 0 && strcmp(arg.type, "integer(IK4)")==0)) {
      } else {
        printf("%s %s %s (%s)\n", set->name, arg.dat->type, arg.type,
               arg.dat->name);
        op_err_print("dataset type does not match declared type", m, name);
      }
    }

    if (arg.idx >= 0)
      (*ninds)++;
  }

  /* error checking for op_arg_gbl */

  if (arg.argtype == OP_ARG_GBL || arg.argtype == OP_ARG_INFO) {
    if (!strcmp(arg.type, "error"))
      op_err_print("datatype does not match declared type", m, name);

    if (arg.dim < 0) {
      op_err_print("dimension should be positive", m, name);
    }

    if (arg.data == NULL)
      op_err_print("NULL pointer for global data", m, name);
  }

  if (arg.argtype == OP_ARG_IDX) {
    if (arg.idx == -1 && arg.map != NULL)
      op_err_print("op_arg_idx with -1 index but non-OP_ID map", m, name);
    if (arg.idx != -1 && arg.map == NULL)
      op_err_print("op_arg_idx map index not -1, but no valid map given", m, name);
  }
}

op_arg op_arg_idx(int idx, op_map map) {
  op_arg arg;
  arg.index = -1;
  arg.opt = 1;
  arg.argtype = OP_ARG_IDX;
  arg.dat = NULL;
  arg.map = map;
  arg.idx = idx;
  arg.dim = 1;
  arg.size = sizeof(int);
  arg.data = NULL;
  arg.data_d = NULL;
  arg.map_data_d = (idx == -1 ? NULL : map->map_d);
  arg.map_data = (idx == -1 ? NULL : map->map);
  arg.type = intstr;
  arg.acc = OP_READ;
  arg.sent = 0;
  return arg;
}

op_arg op_arg_idx_ptr(int idx, int *map) {
  op_map item_map = op_search_map_ptr(map);

  if (item_map == NULL && idx == -2)
    idx = -1;

  if (item_map == NULL && idx != -1) {
    printf("ERROR: op_map not found for %p pointer\n", map);
    exit(-1);
  }

  return op_arg_idx(idx, item_map);
}

op_arg op_arg_dat_core(op_dat dat, int idx, op_map map, int dim,
                       const char *typ, op_access acc) {
  op_arg arg;

  /* index is not used for now */
  arg.index = -1;
  arg.opt = 1;
  arg.argtype = OP_ARG_DAT;

  arg.dat = dat;
  arg.map = map;
  arg.dim = dim;
  arg.idx = idx;

  if (dat != NULL) {
    arg.size = dat->size;
    arg.data = dat->data;
    arg.data_d = dat->data_d;
    arg.map_data_d = (idx == -1 ? NULL : map->map_d);
    arg.map_data = (idx == -1 ? NULL : map->map);
  } else {
    /* set default values */
    arg.size = -1;
    arg.data = NULL;
    arg.data_d = NULL;
    arg.map_data_d = NULL;
    arg.map_data = NULL;
  }

  if (strcmp(typ, "double") == 0 || strcmp(typ, "r8") == 0 ||
      strcmp(typ, "real*8") == 0)
    arg.type = doublestr;
  else if (strcmp(typ, "float") == 0 || strcmp(typ, "r4") == 0 ||
           strcmp(typ, "real*4") == 0)
    arg.type = floatstr;
  else if (strcmp(typ, "int") == 0 || strcmp(typ, "i4") == 0 ||
           strcmp(typ, "integer*4") == 0)
    arg.type = intstr;
  else if (strcmp(typ, "bool") == 0)
    arg.type = boolstr;
  else
    arg.type = copy_str(typ); //Warning this is going to leak

  arg.acc = acc;

  /*initialize to 0 states no-mpi messages inflight for this arg*/
  arg.sent = 0;

  return arg;
}

op_arg op_opt_arg_dat_core(int opt, op_dat dat, int idx, op_map map, int dim,
                           const char *typ, op_access acc) {
  op_arg arg;

  /* index is not used for now */
  arg.index = -1;
  arg.opt = opt;
  arg.argtype = OP_ARG_DAT;

  arg.dat = dat;
  arg.map = map;
  arg.dim = dim;
  arg.idx = idx;

  if (dat != NULL) {
    arg.size = dat->size;
    arg.data = dat->data;
    arg.data_d = dat->data_d;
    arg.map_data_d = (map == NULL ? NULL : map->map_d);
    arg.map_data = map == NULL ? NULL : map->map;
  } else {
    /* set default values */
    arg.size = -1;
    arg.data = NULL;
    arg.data_d = NULL;
    arg.map_data_d = (map == NULL ? NULL : map->map_d);
    arg.map_data = (map == NULL ? NULL : map->map);
  }

  if (strcmp(typ, "double") == 0 || strcmp(typ, "r8") == 0 ||
      strcmp(typ, "real*8") == 0)
    arg.type = doublestr;
  else if (strcmp(typ, "float") == 0 || strcmp(typ, "r4") == 0 ||
           strcmp(typ, "real*4") == 0)
    arg.type = floatstr;
  else if (strcmp(typ, "int") == 0 || strcmp(typ, "i4") == 0 ||
           strcmp(typ, "integer*4") == 0)
    arg.type = intstr;
  else if (strcmp(typ, "bool") == 0)
    arg.type = boolstr;
  else
    arg.type = copy_str(typ); //Warning this is going to leak

  arg.acc = acc;

  /*initialize to 0 states no-mpi messages inflight for this arg*/
  arg.sent = 0;

  return arg;
}

op_arg op_arg_info_char(char *data, int dim, const char *typ, int size, int ref) {
  op_arg arg;
  arg.argtype = OP_ARG_INFO;

  arg.dat = NULL;
  arg.opt = 1;
  arg.map = NULL;
  arg.dim = dim;
  arg.idx = -1;
  arg.size = dim * size;
  arg.data = data;
  if (strcmp(typ, "double") == 0 || strcmp(typ, "r8") == 0 ||
      strcmp(typ, "real*8") == 0)
    arg.type = doublestr;
  else if (strcmp(typ, "float") == 0 || strcmp(typ, "r4") == 0 ||
           strcmp(typ, "real*4") == 0)
    arg.type = floatstr;
  else if (strcmp(typ, "int") == 0 || strcmp(typ, "i4") == 0 ||
           strcmp(typ, "integer*4") == 0)
    arg.type = intstr;
  else if (strcmp(typ, "bool") == 0)
    arg.type = boolstr;

  arg.acc = ref;
  arg.map_data_d = NULL;
  arg.map_data = NULL;

  /* setting default values for remaining fields */
  arg.index = -1;
  arg.data_d = NULL;

  /*not used in global args*/
  arg.sent = 0;

  /* TODO: properly??*/
  if (data == NULL)
    arg.opt = 0;

  return arg;
}

op_arg op_arg_gbl_core(int opt, char *data, int dim, const char *typ, int size,
                       op_access acc) {
  op_arg arg;

  arg.argtype = OP_ARG_GBL;

  arg.dat = NULL;
  arg.opt = opt;
  arg.map = NULL;
  arg.dim = dim;
  arg.idx = -1;
  arg.size = dim * size;
  arg.data = data;
  if (strcmp(typ, "double") == 0 || strcmp(typ, "r8") == 0 ||
      strcmp(typ, "real*8") == 0)
    arg.type = doublestr;
  else if (strcmp(typ, "float") == 0 || strcmp(typ, "r4") == 0 ||
           strcmp(typ, "real*4") == 0)
    arg.type = floatstr;
  else if (strcmp(typ, "int") == 0 || strcmp(typ, "i4") == 0 ||
           strcmp(typ, "integer*4") == 0)
    arg.type = intstr;
  else if (strcmp(typ, "bool") == 0)
    arg.type = boolstr;

  arg.acc = acc;
  arg.map_data_d = NULL;
  arg.map_data = NULL;

  /* setting default values for remaining fields */
  arg.index = -1;
  arg.data_d = NULL;

  /*not used in global args*/
  arg.sent = 0;

  /* TODO: properly??*/
  if (data == NULL)
    arg.opt = 0;

  return arg;
}

/*
 * diagnostic routines
 */

void op_diagnostic_output() {
  if (OP_diags > 1) {
    printf("\n  OP diagnostic output\n");
    printf("  --------------------\n");

    printf("\n       set       size\n");
    printf("  -------------------\n");
    for (int n = 0; n < OP_set_index; n++) {
      printf("%10s %10d\n", OP_set_list[n]->name, OP_set_list[n]->size);
    }

    printf("\n       map        dim       from         to\n");
    printf("  -----------------------------------------\n");
    for (int n = 0; n < OP_map_index; n++) {
      printf("%10s %10d %10s %10s\n", OP_map_list[n]->name, OP_map_list[n]->dim,
             OP_map_list[n]->from->name, OP_map_list[n]->to->name);
    }

    printf("\n       dat        dim        set\n");
    printf("  ------------------------------\n");
    op_dat_entry *item;
    TAILQ_FOREACH(item, &OP_dat_list, entries) {
      printf("%10s %10d %10s\n", (item->dat)->name, (item->dat)->dim,
             (item->dat)->set->name);
    }
    printf("\n");
  }
}

void op_timing_output_core() {
  if (OP_kern_max == 0) return;

  op_printf("\n  count   plan time     MPI time(std)        time(std)         "
           "  GB/s      GB/s   kernel name ");
  op_printf("\n "
           "-----------------------------------------------------------------"
           "--------------------------\n");

  std::vector<op_kernel_timing_summary> summaries;
  for (int n = 0; n < OP_kern_max; n++) {
    if (OP_kernels[n].count == 0)
      continue;

    if (OP_kernels[n].ntimes == 1 && OP_kernels[n].times[0] == 0 &&
        OP_kernels[n].time != 0)
      OP_kernels[n].times[0] = OP_kernels[n].time;

    op_kernel_timing_summary summary;

    summary.name = OP_kernels[n].name;
    summary.count = OP_kernels[n].count;
    summary.plan_time = OP_kernels[n].plan_time;

    double moments_mpi_time[2];
    double moments_time[2];
    op_compute_moment_across_times(OP_kernels[n].times, OP_kernels[n].ntimes, true,
                                   &moments_time[0], &moments_time[1]);
    op_compute_moment(OP_kernels[n].mpi_time, &moments_mpi_time[0], &moments_mpi_time[1]);

    summary.mpi_time = moments_mpi_time[0];
    summary.mpi_time_std = sqrt(moments_mpi_time[1] - moments_mpi_time[0] * moments_mpi_time[0]);

    summary.time = moments_time[0];
    summary.time_std = sqrt(moments_time[1] - moments_time[0] * moments_time[0]);

    double kern_time = OP_kernels[n].times[0];
    for (int i = 1; i < OP_kernels[n].ntimes; i++)
        kern_time = MAX(kern_time, OP_kernels[n].times[i]);

    if (OP_kernels[n].transfer2 < 1e-8f) {
      float gsec = 1e9f * kern_time - OP_kernels[n].mpi_time;

      summary.transfer  = MAX(0.0f, OP_kernels[n].transfer / gsec);
      summary.transfer2 = 0;
    } else {
      float gsec = 1e9f * kern_time - OP_kernels[n].plan_time - OP_kernels[n].mpi_time;

      summary.transfer  = MAX(0.0f, OP_kernels[n].transfer / gsec);
      summary.transfer2 = MAX(0.0f, OP_kernels[n].transfer2 / gsec);
    }

    summaries.push_back(summary);
  }

  std::sort(summaries.begin(), summaries.end(),
      [] (auto const& a, auto const& b) { return b.time < a.time; });

  for (auto& s : summaries) {
    if (s.transfer2 < 1e-8f) {
      op_printf(" %6d;  %8.4f;  %8.4f(%8.4f);  %8.4f(%8.4f);  %8.4f;        "
                " ;   %s \n",
                s.count, s.plan_time, s.mpi_time, s.mpi_time_std,
                s.time, s.time_std, s.transfer, s.name);
    } else {
      op_printf(" %6d;  %8.4f;  %8.4f(%8.4f);  %8.4f(%8.4f); %8.4f; %8.4f;  "
                " %s \n",
                s.count, s.plan_time, s.mpi_time, s.mpi_time_std,
                s.time, s.time_std, s.transfer, s.transfer2, s.name);
    }
  }
}

void op_timing_output_2_file(const char *outputFileName) {
  FILE *outputFile = NULL;
  float totalKernelTime = 0.0f;

  outputFile = fopen(outputFileName, "w+");
  if (outputFile == NULL) {
    printf("Bad output file\n");
    exit(1);
  }

  if (OP_kern_max > 0) {
    fprintf(outputFile, "\n  count     time     GB/s     GB/s   kernel name ");
    fprintf(outputFile,
            "\n ----------------------------------------------- \n");
    for (int n = 0; n < OP_kern_max; n++) {
      if (OP_kernels[n].count > 0) {
        double kern_time = OP_kernels[n].times[0];
        for (int i=1; i<OP_kernels[n].ntimes; i++) {
          if (OP_kernels[n].times[i] > kern_time)
            kern_time = OP_kernels[n].times[i];
        }
        if (OP_kernels[n].transfer2 < 1e-8f) {
          totalKernelTime += kern_time;
          fprintf(outputFile, " %6d  %8.4f %8.4f            %s \n",
                  OP_kernels[n].count, kern_time,
                  OP_kernels[n].transfer / (1e9f * kern_time),
                  OP_kernels[n].name);
        } else {
          totalKernelTime += kern_time;
          fprintf(outputFile, " %6d  %8.4f %8.4f %8.4f   %s \n",
                  OP_kernels[n].count, kern_time,
                  OP_kernels[n].transfer / (1e9f * kern_time),
                  OP_kernels[n].transfer2 / (1e9f * kern_time),
                  OP_kernels[n].name);
        }
      }
    }
    fprintf(outputFile, "Total kernel time = %f\n", totalKernelTime);
  }

  fclose(outputFile);
}

void op_timers_core(double *cpu, double *et) {
  (void)cpu;
  struct timeval t;

  gettimeofday(&t, (struct timezone *)0);
  *et = t.tv_sec + t.tv_usec * 1.0e-6;
}

void op_timing_realloc(int kernel) {
  op_timing_realloc_manytime(kernel, 1);
}

void op_timing_realloc_manytime(int kernel, int num_timers) {
  int OP_kern_max_new;
  OP_kern_curr = kernel;

  if (kernel >= OP_kern_max) {
    OP_kern_max_new = kernel + 10;
    OP_kernels = (op_kernel *)op_realloc(OP_kernels,
                                         OP_kern_max_new * sizeof(op_kernel));
    if (OP_kernels == NULL) {
      printf(" op_timing_realloc error \n");
      exit(-1);
    }

    for (int n = OP_kern_max; n < OP_kern_max_new; n++) {
      OP_kernels[n].count = 0;
      OP_kernels[n].time = 0.0f;
      OP_kernels[n].times = (double*)op_malloc(num_timers * sizeof(double));
      for (int t = 0; t < num_timers; t++) {
        OP_kernels[n].times[t] = 0.0f;
      }
      OP_kernels[n].ntimes = num_timers;
      OP_kernels[n].plan_time = 0.0f;
      OP_kernels[n].transfer = 0.0f;
      OP_kernels[n].transfer2 = 0.0f;
      OP_kernels[n].mpi_time = 0.0f;
      OP_kernels[n].name = "unused";
    }
    OP_kern_max = OP_kern_max_new;
  }
}

void op_dump_dat(op_dat data) {
  fflush(stdout);

  if (data != NULL) {
    if (strncmp("real", data->type, 4) == 0) {
      for (int i = 0; i < data->dim * data->set->size; i++)
        printf("%lf\n", ((double *)data->data)[i]);
    } else if (strncmp("integer", data->type, 7) == 0) {
      for (int i = 0; i < data->dim * data->set->size; i++)
        printf("%d\n", data->data[i]);
    } else {
      printf("Unsupported type for dumping %s\n", data->type);
      exit(0);
    }
  }

  fflush(stdout);
}

void op_print_dat_to_binfile_core(op_dat dat, const char *file_name) {
  size_t elem_size = dat->dim;
  int count = dat->set->size;

  FILE *fp;
  if ((fp = fopen(file_name, "wb")) == NULL) {
    printf("can't open file %s\n", file_name);
    exit(2);
  }

  if (fwrite(&count, sizeof(int), 1, fp) < 1) {
    printf("error writing to %s", file_name);
    exit(2);
  }
  if (fwrite(&elem_size, sizeof(int), 1, fp) < 1) {
    printf("error writing to %s\n", file_name);
    exit(2);
  }

  if (fwrite(dat->data, dat->size, dat->set->size, fp) < dat->set->size) {
    printf("error writing to %s\n", file_name);
    exit(2);
  }
  fclose(fp);
}

void op_print_dat_to_txtfile_core(op_dat dat, const char *file_name) {
  FILE *fp;
  if ((fp = fopen(file_name, "w")) == NULL) {
    printf("can't open file %s\n", file_name);
    exit(2);
  }

  if (fprintf(fp, "%d %d\n", dat->set->size, dat->dim) < 0) {
    printf("error writing to %s\n", file_name);
    exit(2);
  }

  for (int i = 0; i < dat->set->size; i++) {
    for (int j = 0; j < dat->dim; j++) {
      if (strcmp(dat->type, "double") == 0 ||
          strcmp(dat->type, "double:soa") == 0 ||
          strcmp(dat->type, "double precision") == 0 ||
          strcmp(dat->type, "real(8)") == 0) {
        // if (fprintf(fp, "%2.15lf ", ((double *)dat->data)[i * dat->dim + j])
        // <
        if (((double *)dat->data)[i * dat->dim + j] == -0.0)
          ((double *)dat->data)[i * dat->dim + j] = +0.0;
        if (fprintf(fp, " %+2.15lE", ((double *)dat->data)[i * dat->dim + j]) <
            0) {
          printf("error writing to %s\n", file_name);
          exit(2);
        }
      } else if (strcmp(dat->type, "float") == 0 ||
                 strcmp(dat->type, "float:soa") == 0 ||
                 strcmp(dat->type, "real(4)") == 0 ||
                 strcmp(dat->type, "real") == 0) {
        if (fprintf(fp, " %+f", ((float *)dat->data)[i * dat->dim + j]) < 0) {
          printf("error writing to %s\n", file_name);
          exit(2);
        }
      } else if (strcmp(dat->type, "int") == 0 ||
                 strcmp(dat->type, "int:soa") == 0 ||
                 strcmp(dat->type, "int(4)") == 0 ||
                 strcmp(dat->type, "integer") == 0 ||
                 strcmp(dat->type, "integer(4)") == 0) {
        if (fprintf(fp, " %+d", ((int *)dat->data)[i * dat->dim + j]) < 0) {
          printf("error writing to %s\n", file_name);
          exit(2);
        }
      } else if ((strcmp(dat->type, "long") == 0) ||
                 (strcmp(dat->type, "long:soa") == 0)) {
        if (fprintf(fp, " %+ld", ((long *)dat->data)[i * dat->dim + j]) < 0) {
          printf("error writing to %s\n", file_name);
          exit(2);
        }
      } else {
        printf("Unknown type %s, cannot be written to file %s\n", dat->type,
               file_name);
        exit(2);
      }
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

int op_size_of_set(const char *name) {
  for (int i = 0; i < OP_set_index; i++) {
    if (strcmp(name, OP_set_list[i]->name) == 0)
      return OP_set_list[i]->size + OP_set_list[i]->exec_size +
             OP_set_list[i]->nonexec_size;
  }
  printf("Error: set %s not found\n", name);
  exit(-1);
  return -1;
}

void set_maps_base(int base) {
  if (base == 1 || base == 0) {
    OP_maps_base_index = base;
  }
}

void *op_malloc(size_t size) {
  if (size == 0) return malloc(0);
  return aligned_alloc(OP2_ALIGNMENT, (size + OP2_ALIGNMENT) - 1 & (-OP2_ALIGNMENT));
}

// malloc to be exposed in Fortran API for use with Cray pointers
void op_malloc2(void **data, int *size){
  *data = op_malloc(*size);
}

void *op_calloc(size_t num, size_t size) {
  void *p = op_malloc(num * size);
  memset(p, 0, num * size);

  return p;
}

void *op_realloc(void *ptr, size_t size) {
  void *new_ptr = realloc(ptr, size);

  if (((unsigned long)new_ptr & (OP2_ALIGNMENT - 1)) == 0)
    return new_ptr;

  void *new_ptr2 = op_malloc(size);
  memcpy(new_ptr2, new_ptr, size);

  return new_ptr2;
}

void op_free(void *ptr) {
  free(ptr);
}

op_arg op_arg_dat(op_dat, int, op_map, int, char const *, op_access);
op_arg op_opt_arg_dat(int, op_dat, int, op_map, int, char const *, op_access);

op_arg op_arg_dat_ptr(int opt, char *dat, int idx, int *map, int dim,
                      char const *type, op_access acc) {
  if (opt == 0)
    return op_opt_arg_dat_core(opt, NULL, idx, NULL, dim, type, acc);
  //  printf("op_arg_dat_ptr with %p\n", dat);
  op_dat_entry *item;
  op_dat_entry *tmp_item;
  op_dat item_dat = NULL;
  for (item = TAILQ_FIRST(&OP_dat_list); item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    if (item->orig_ptr == dat) {
      // printf("%s(%p), ", item->dat->name, item->dat->data);
      item_dat = item->dat;
      break;
    }
  }
  // printf("\n");
  if (item_dat == NULL) {
    printf("ERROR: op_dat not found for dat with %p pointer\n", dat);
  }
  // if(strcmp(item_dat->name,"x")== 0 || strcmp(item_dat->name, "pjaca") == 0
  // ||
  // strcmp(item_dat->name, "ewt") == 0 ||  strcmp(item_dat->name, "vol") == 0)
  // printf(" Found OP2 pointer for dat %s orig_ptr = %lu, dat->data = %lu  \n",
  // item_dat->name, (unsigned long)item->orig_ptr, (unsigned
  // long)item_dat->data);

  op_map item_map = op_search_map_ptr(map);

  if (item_map == NULL && idx == -2)
    idx = -1;

  if (item_map == NULL && idx != -1) {
    printf("ERROR: op_map not found for %p pointer\n", map);
    exit(-1);
  }

  return op_arg_dat_core(item_dat, idx, item_map, dim, type, acc);
}

int *op_set_registry = NULL;
int op_set_registry_size = 0;
void op_register_set(int idx, op_set set) {
  if (idx >= op_set_registry_size) {
    op_set_registry_size = idx + 10;
    op_set_registry =
        (int *)op_realloc(op_set_registry, op_set_registry_size * sizeof(int));
  }
  op_set_registry[idx] = set->index;
}

op_set op_get_set(int idx) {
  if (idx < op_set_registry_size)
    return OP_set_list[op_set_registry[idx]];
  else
    return NULL;
}

void op_dat_write_index(op_set set, int *dat) {
  op_arg arg = op_arg_dat_ptr(1, (char *)dat, -2, NULL, 1, "int", OP_WRITE);
  if (set != arg.dat->set) {
    op_printf("Error: op_dat_write_index set and arg.dat->set do not match\n");
    exit(-1);
  }
  for (int i = 0; i < set->size + set->exec_size + set->nonexec_size; i++) {
    ((int *)arg.dat->data)[i] = i + 1;
  }
  arg.dat->dirty_hd = 1;
}

void op_print_dat_to_txtfile2(int *dat, const char *file_name) {
  op_arg arg = op_arg_dat_ptr(1, (char *)dat, -2, NULL, 1, "int", OP_WRITE);
  op_print_dat_to_txtfile_core(arg.dat, file_name);
}

/*******************************************************************************
 * Routines for accessing inernally held OP2 dats - JM76-type sliding planes
 *******************************************************************************/

unsigned long op_get_data_ptr(op_dat d) {
  op_download_dat(d);
  return (unsigned long)(d->data);
}

unsigned long op_get_data_ptr2(unsigned long data) {
  op_dat_entry *item;
  op_dat_entry *tmp_item;
  op_dat item_dat = NULL;
  for (item = TAILQ_FIRST(&OP_dat_list); item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    if (item->orig_ptr == (char*)data) {
      item_dat = item->dat;
      break;
    }
  }
  if (item_dat == NULL) {
    printf("ERROR: op_dat not found for dat with %lu pointer\n", data);
  }
  op_download_dat(item_dat);
  return (unsigned long)(item_dat->data);
}

unsigned long op_reset_data_ptr(char *data, int mode) {
  op_dat_entry *item;
  op_dat_entry *tmp_item;
  op_dat item_dat = NULL;
  for (item = TAILQ_FIRST(&OP_dat_list); item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    if (item->orig_ptr == data) {
      item_dat = item->dat;
      break;
    }
  }
  if (item_dat == NULL) {
    printf("ERROR: op_dat not found for dat with %p pointer\n", data);
  }
  // printf(" op2 pointer for dat %s before = %lu, after change = %lu  \n",
  // item_dat->name, (unsigned long)item->orig_ptr, (unsigned
  // long)item_dat->data);

  // Download dat from device (if required)
  // op_download_dat(item_dat);

  // reset orig pointer
  // free(data)

  if (mode == 1) {
    item->orig_ptr = (char*)(unsigned long)item_dat->index;
    return (unsigned long)(item->orig_ptr);
  } else if (mode == 2) {
    item->orig_ptr = item_dat->data;
    return (unsigned long)(item->orig_ptr);
  } else {
    printf("Unknown mode\n");
    exit(-1);
    return 0;
  }
}

/*******************************************************************************
 * Common routines for accessing inernally held OP2 maps
 *******************************************************************************/

unsigned long op_get_map_ptr(op_map m) { return (unsigned long)(m->map); }

unsigned long op_reset_map_ptr(int *map) {
  op_map item_map = op_search_map_ptr(map);

  if (item_map == NULL) {
    printf("ERROR: op_map not found for map with %p pointer\n", map);
    exit(-1);
  }

  OP_map_ptr_table.erase(map);
  OP_map_ptr_table.insert({item_map->map, item_map->index});

  return (unsigned long)(item_map->map);
}

unsigned long op_copy_map_to_fort(int *map) {
  op_map item_map = op_search_map_ptr(map);

  if (item_map == NULL) {
    printf("ERROR: op_map not found for map with %p pointer\n", map);
    exit(-1);
  }

  int *fort_map = (int *) malloc((item_map->from->size + item_map->from->exec_size+1)* item_map->dim * sizeof(int));
  fort_map++; //Sometimes this will return an allocation previously deallocated, but still on the OP_map_ptr_table. Increment by one to avoid
  if (OP_map_ptr_table.count(fort_map)>0) {
    printf("Error: Map pointer %p already in map_ptr_table\n", fort_map);
    exit(-1);
  }
  OP_map_ptr_table.insert({fort_map, item_map->index});

  for (int i = 0; i < (item_map->from->size + item_map->from->exec_size)* item_map->dim; i++)
    fort_map[i] = item_map->map[i] + 1;

  return (unsigned long)(fort_map);
}

void op_force_part(int *map) {
  op_map item_map = op_search_map_ptr(map);

  if (item_map == NULL) {
    printf("ERROR: op_map not found for map with %p pointer\n", map);
    exit(-1);
  }

  item_map->force_part = true;
}

/*******************************************************************************
 * Get the local size of a set
 *******************************************************************************/

int op_get_size_local_core(op_set set) { return set->core_size; }
int op_get_size_local(op_set set) { return set->size; }

/*******************************************************************************
 * Get the local exec size of a set
 *******************************************************************************/

int op_get_size_local_exec(op_set set) { return set->exec_size + set->size; }
int op_get_size_local_full(op_set set) { return set->exec_size + set->nonexec_size + set->size; }
int op_mpi_get_test_frequency() { return OP_mpi_test_frequency; }

int op_is_partitioned() { return OP_is_partitioned; }

#ifdef __cplusplus
}
#endif
