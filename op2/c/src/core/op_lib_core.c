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

/*
 * OP2 global state variables
 */

int OP_diags = 0, OP_part_size = 0, OP_block_size = 64,
    OP_cache_line_size = 128, OP_gpu_direct = 0;

double OP_hybrid_balance = 1.0;
int OP_hybrid_gpu = 0;
int OP_auto_soa = 0;
int OP_maps_base_index = 0;

int OP_set_index = 0, OP_set_max = 0, OP_map_index = 0, OP_map_max = 0,
    OP_dat_index = 0, OP_kern_max = 0, OP_kern_curr = 0;

/*
 * Lists of sets, maps and dats declared in OP2 programs
 */

op_set *OP_set_list;
op_map *OP_map_list;
Double_linked_list OP_dat_list; /*Head of the double linked list*/
op_kernel *OP_kernels;

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

op_dat search_dat(op_set set, int dim, char const *type, int size,
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
  pch = strstr(argv, "OP_CACHE_LINE_SIZE=");
  if (pch != NULL) {
    strncpy(temp, pch, 25);
    OP_cache_line_size = atoi(temp + 19);
    op_printf("\n OP_cache_line_size  = %d \n", OP_cache_line_size);
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

  if (OP_map_index == OP_map_max) {
    OP_map_max += 10;
    OP_map_list =
        (op_map *)op_realloc(OP_map_list, OP_map_max * sizeof(op_map));

    if (OP_map_list == NULL) {
      printf(" op_decl_map error -- error reallocating memory\n");
      exit(-1);
    }
  }

  if (OP_maps_base_index == 1) {
    // convert imap to 0 based indexing -- i.e. reduce each imap value by 1
    for (int i = 0; i < from->size * dim; i++)
      imap[i]--;
  }
  // else OP_maps_base_index == 0
  // do nothing -- aready in C style indexing

  op_map map = (op_map)op_malloc(sizeof(op_map_core));
  map->index = OP_map_index;
  map->from = from;
  map->to = to;
  map->dim = dim;
  map->map = imap;
  map->name = copy_str(name);
  map->user_managed = 1;

  OP_map_list[OP_map_index++] = map;

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
  dat->data = data;
  dat->data_d = NULL;
  dat->name = copy_str(name);
  dat->type = copy_str(type);
  dat->size = dim * size;
  dat->user_managed = 1;
  dat->mpi_buffer = NULL;
  dat->dirty_hd = 0;

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

  // add item to the end of the list
  TAILQ_INSERT_TAIL(&OP_dat_list, item, entries);
  OP_dat_index++;

  return dat;
}

/*
 * temporary dats
 */

op_dat op_decl_dat_temp_core(op_set set, int dim, char const *type, int size,
                             char *data, char const *name) {
  // Check if this dat already exists in the double linked list
  op_dat found_dat = search_dat(set, dim, type, size, name);
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

  /*free doubl linked list holding the op_dats */
  op_dat_entry *item;
  while ((item = TAILQ_FIRST(&OP_dat_list))) {
    if (!(item->dat)->user_managed)
      free((item->dat)->data);
    free((char *)(item->dat)->name);
    free((char *)(item->dat)->type);
    TAILQ_REMOVE(&OP_dat_list, item, entries);
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

    if (arg.map == NULL && arg.dat->set != set)
      op_err_print("dataset set does not match loop set", m, name);

    if (arg.map != NULL &&
        (arg.map->from != set || arg.map->to != arg.dat->set))
      op_err_print("mapping error", m, name);

    if ((arg.map == NULL && arg.idx != -1) ||
        (arg.map != NULL &&
         (arg.idx >= arg.map->dim || arg.idx < -1 * arg.map->dim)))
      op_err_print("invalid index", m, name);

    if (arg.dat->dim != arg.dim)
      op_err_print("dataset dim does not match declared dim", m, name);

    if (strcmp(arg.dat->type, arg.type)) {
      if ((strcmp(arg.dat->type, "double") == 0 && (int)arg.type[0] == 114 &&
           (int)arg.type[1] == 56 && (int)arg.type[2] != 58) ||
          (strcmp(arg.dat->type, "double:soa") == 0 &&
           (int)arg.type[0] == 114 && (int)arg.type[1] == 56 &&
           (int)arg.type[2] == 58) ||
          (strcmp(arg.dat->type, "int") == 0 && (int)arg.type[0] == 105 &&
           (int)arg.type[1] == 52)) {
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

  if (arg.argtype == OP_ARG_GBL) {
    if (!strcmp(arg.type, "error"))
      op_err_print("datatype does not match declared type", m, name);

    if (arg.dim <= 0)
      op_err_print("dimension should be strictly positive", m, name);

    if (arg.data == NULL)
      op_err_print("NULL pointer for global data", m, name);
  }
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

  // NJH
  // arg.type = typ;
  arg.type = copy_str(typ);
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

  // NJH
  // arg.type = typ;
  arg.type = copy_str(typ);
  arg.acc = acc;

  /*initialize to 0 states no-mpi messages inflight for this arg*/
  arg.sent = 0;

  return arg;
}

op_arg op_arg_gbl_core(char *data, int dim, const char *typ, int size,
                       op_access acc) {
  op_arg arg;

  arg.argtype = OP_ARG_GBL;

  arg.dat = NULL;
  arg.opt = 1;
  arg.map = NULL;
  arg.dim = dim;
  arg.idx = -1;
  arg.size = dim * size;
  arg.data = data;
  // NJH
  // arg.type = typ;
  arg.type = copy_str(typ);
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
  if (OP_kern_max > 0) {
    if (op_is_root())
      printf("\n  count   plan time     MPI time(std)        time(std)         "
             "  GB/s      GB/s   kernel name ");
    if (op_is_root())
      printf("\n "
             "-----------------------------------------------------------------"
             "--------------------------\n");
    for (int n = 0; n < OP_kern_max; n++) {
      if (OP_kernels[n].count > 0) {
        double moments_mpi_time[2];
        double moments_time[2];
        op_compute_moment(OP_kernels[n].time, &moments_time[0],
                          &moments_time[1]);
        op_compute_moment(OP_kernels[n].mpi_time, &moments_mpi_time[0],
                          &moments_mpi_time[1]);
        if (OP_kernels[n].transfer2 < 1e-8f) {
          float transfer =
              MAX(0.0f, OP_kernels[n].transfer / (1e9f * OP_kernels[n].time -
                                                  OP_kernels[n].mpi_time));

          if (op_is_root())
            printf(" %6d;  %8.4f;  %8.4f(%8.4f);  %8.4f(%8.4f);  %8.4f;        "
                   " ;   %s \n",
                   OP_kernels[n].count, OP_kernels[n].plan_time,
                   moments_mpi_time[0],
                   sqrt(moments_mpi_time[1] -
                        moments_mpi_time[0] * moments_mpi_time[0]),
                   moments_time[0],
                   sqrt(moments_time[1] - moments_time[0] * moments_time[0]),
                   transfer, OP_kernels[n].name);
        } else {
          float transfer =
              MAX(0.0f, OP_kernels[n].transfer /
                            (1e9f * OP_kernels[n].time -
                             OP_kernels[n].plan_time - OP_kernels[n].mpi_time));
          float transfer2 =
              MAX(0.0f, OP_kernels[n].transfer2 /
                            (1e9f * OP_kernels[n].time -
                             OP_kernels[n].plan_time - OP_kernels[n].mpi_time));
          if (op_is_root())
            printf(" %6d;  %8.4f;  %8.4f(%8.4f);  %8.4f(%8.4f); %8.4f; %8.4f;  "
                   " %s \n",
                   OP_kernels[n].count, OP_kernels[n].plan_time,
                   moments_mpi_time[0],
                   sqrt(moments_mpi_time[1] -
                        moments_mpi_time[0] * moments_mpi_time[0]),
                   moments_time[0],
                   sqrt(moments_time[1] - moments_time[0] * moments_time[0]),
                   transfer, transfer2, OP_kernels[n].name);
        }
      }
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
        if (OP_kernels[n].transfer2 < 1e-8f) {
          totalKernelTime += OP_kernels[n].time;
          fprintf(outputFile, " %6d  %8.4f %8.4f            %s \n",
                  OP_kernels[n].count, OP_kernels[n].time,
                  OP_kernels[n].transfer / (1e9f * OP_kernels[n].time),
                  OP_kernels[n].name);
        } else {
          totalKernelTime += OP_kernels[n].time;
          fprintf(outputFile, " %6d  %8.4f %8.4f %8.4f   %s \n",
                  OP_kernels[n].count, OP_kernels[n].time,
                  OP_kernels[n].transfer / (1e9f * OP_kernels[n].time),
                  OP_kernels[n].transfer2 / (1e9f * OP_kernels[n].time),
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
        if (fprintf(fp, "%lf ", ((double *)dat->data)[i * dat->dim + j]) < 0) {
          printf("error writing to %s\n", file_name);
          exit(2);
        }
      } else if (strcmp(dat->type, "float") == 0 ||
                 strcmp(dat->type, "float:soa") == 0 ||
                 strcmp(dat->type, "real(4)") == 0 ||
                 strcmp(dat->type, "real") == 0) {
        if (fprintf(fp, "%f ", ((float *)dat->data)[i * dat->dim + j]) < 0) {
          printf("error writing to %s\n", file_name);
          exit(2);
        }
      } else if (strcmp(dat->type, "int") == 0 ||
                 strcmp(dat->type, "int:soa") == 0 ||
                 strcmp(dat->type, "int(4)") == 0 ||
                 strcmp(dat->type, "integer") == 0 ||
                 strcmp(dat->type, "integer(4)") == 0) {
        if (fprintf(fp, "%d ", ((int *)dat->data)[i * dat->dim + j]) < 0) {
          printf("error writing to %s\n", file_name);
          exit(2);
        }
      } else if ((strcmp(dat->type, "long") == 0) ||
                 (strcmp(dat->type, "long:soa") == 0)) {
        if (fprintf(fp, "%ld ", ((long *)dat->data)[i * dat->dim + j]) < 0) {
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
#ifdef __INTEL_COMPILER
  // return _mm_malloc(size, OP2_ALIGNMENT);
  return memalign(OP2_ALIGNMENT, size);
#else
  return malloc(size);
#endif
}

void *op_calloc(size_t num, size_t size) {
#ifdef __INTEL_COMPILER
  // void * ptr = _mm_malloc(num*size, OP2_ALIGNMENT);
  void *ptr = memalign(OP2_ALIGNMENT, num * size);
  memset(ptr, 0, num * size);
  return ptr;
#else
  return calloc(num, size);
#endif
}

void *op_realloc(void *ptr, size_t size) {
#ifdef __INTEL_COMPILER
  void *newptr = realloc(ptr, size);
  if (((unsigned long)newptr & (OP2_ALIGNMENT - 1)) != 0) {
    void *newptr2 = memalign(OP2_ALIGNMENT, size);
    // void *newptr2 = _mm_malloc(size, OP2_ALIGNMENT);
    memcpy(newptr2, newptr, size);
    free(newptr);
    return newptr2;
  } else {
    return newptr;
  }
#else
  return realloc(ptr, size);
#endif
}

void op_free(void *ptr) {
#ifdef __INTEL_COMPILER
  //_mm_free(ptr);
  free(ptr);
#else
  free(ptr);
#endif
}
