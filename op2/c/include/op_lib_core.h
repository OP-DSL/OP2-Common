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

#ifndef __OP_LIB_CORE_H
#define __OP_LIB_CORE_H

/*
 * This header file declares all types and functions required by *any* OP2
 * implementation, i.e. independently of the back-end and the target
 * languages.  It is typically used by language or back-end specific top level
 * OP2 libraries
 */

#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/queue.h> //contains double linked list implementation

#ifndef OP2_ALIGNMENT
#define OP2_ALIGNMENT 64
#endif

/*
 * essential typedefs
 */
#ifndef __PGI
typedef unsigned int uint;
#endif
typedef long long ll;
typedef unsigned long long ull;

/*
 * OP2 global state variables
 */

/*
 * OP diagnostics level
   0            none
   1 or above   error-checking
   2 or above   info on plan construction
   3 or above   report execution of parallel loops
   4 or above   report use of old plans
   7 or above   report positive checks in op_plan_check
*/

extern int OP_diags;
extern int OP_cache_line_size;
extern double OP_hybrid_balance;
extern int OP_hybrid_gpu;
extern int OP_maps_base_index;

/*
 * enum list for op_par_loop
 */

#define OP_READ 0
#define OP_WRITE 1
#define OP_RW 2
#define OP_INC 3
#define OP_MIN 4
#define OP_MAX 5

#define OP_ARG_GBL 0
#define OP_ARG_DAT 1

#define OP_STAGE_NONE 0
#define OP_STAGE_INC 1
#define OP_STAGE_ALL 2
#define OP_STAGE_PERMUTE 3
#define OP_COLOR2 4

typedef int op_access; // holds OP_READ, OP_WRITE, OP_RW, OP_INC, OP_MIN, OP_MAX
typedef int op_arg_type; // holds OP_ARG_GBL, OP_ARG_DAT

/*
 * structures
 */

typedef struct {
  int index;        /* index */
  int size;         /* number of elements in set */
  char const *name; /* name of set */
                    // for MPI support
  int core_size;    /* number of core elements in an mpi process*/
  int exec_size;    /* number of additional imported elements to be executed */
  int nonexec_size; /* number of additional imported elements that are not
                       executed */
} op_set_core;

typedef op_set_core *op_set;

typedef struct {
  int index;        /* index */
  op_set from,      /* set pointed from */
      to;           /* set pointed to */
  int dim,          /* dimension of pointer */
      *map;         /* array defining pointer */
  int *map_d;       /* array on device */
  char const *name; /* name of pointer */
  int user_managed; /* indicates whether the user is managing memory */
} op_map_core;

typedef op_map_core *op_map;

typedef struct {
  int index;        /* index */
  op_set set;       /* set on which data is defined */
  int dim,          /* dimension of data */
      size;         /* size of each element in dataset */
  char *data,       /* data on host */
      *data_d;      /* data on device (GPU) */
  char const *type, /* datatype */
      *name;        /* name of dataset */
  char *buffer_d;   /* buffer for MPI halo sends on the devidce */
  char *buffer_d_r; /* buffer for MPI halo receives on the devidce */
  int dirtybit;     /* flag to indicate MPI halo exchange is needed*/
  int dirty_hd;     /* flag to indicate dirty status on host and device */
  int user_managed; /* indicates whether the user is managing memory */
  void *mpi_buffer; /* ponter to hold the mpi buffer struct for the op_dat*/
} op_dat_core;

typedef op_dat_core *op_dat;

typedef struct {
  int index;        /* index */
  op_dat dat;       /* dataset */
  op_map map;       /* indirect mapping */
  int dim,          /* dimension of data */
      idx, size;    /* size (for sequential execution) */
  char *data,       /* data on host */
      *data_d;      /* data on device (for CUDA execution) */
  int *map_data,    /* data on host */
      *map_data_d;  /* data on device (for CUDA execution) */
  char const *type; /* datatype */
  op_access acc;
  op_arg_type argtype;
  int sent; /* flag to indicate if this argument has
               data in flight under non-blocking MPI comms*/
  int opt;  /* flag to indicate if this argument is in use */
} op_arg;

typedef struct {
  char const *name; /* name of kernel function */
  int count;        /* number of times called */
  double time;      /* total execution time */
  double* times;    /* total execution time of each thread */
  int ntimes;       /* number of thread-level times */
  float plan_time;  /* time spent in op_plan_get */
  float transfer;   /* bytes of data transfer (used) */
  float transfer2;  /* bytes of data transfer (total) */
  double mpi_time;   /* time spent in MPI calls */
} op_kernel;

// struct definition for a double linked list entry to hold an op_dat
struct op_dat_entry_core {
  op_dat dat;
  void *orig_ptr;
  TAILQ_ENTRY(op_dat_entry_core)
  entries; /*holds pointers to next and
           previous entries in the list*/
};

typedef struct op_dat_entry_core op_dat_entry;

typedef TAILQ_HEAD(, op_dat_entry_core) Double_linked_list;

/*
 * min / max definitions
 */

#ifndef MIN
#define MIN(a, b) ((a < b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a > b) ? (a) : (b))
#endif

/*
 * alignment macro based on example on page 50 of CUDA Programming Guide version
 * 3.0
 * rounds up to nearest multiple of 16 bytes
 */

#define ROUND_UP(bytes) (((bytes) + 15) & ~15)
#define ROUND_UP_64(bytes) (((bytes) + 63) & ~63)

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************
 * Core lib function prototypes
*******************************************************************************/

void op_init_core(int, char **, int);

void op_exit_core(void);

op_set op_decl_set_core(int, char const *);

op_map op_decl_map_core(op_set, op_set, int, int *, char const *);

op_dat op_decl_dat_core(op_set, int, char const *, int, char *, char const *);

op_dat op_decl_dat_temp_core(op_set, int, char const *, int, char *,
                             char const *);

int op_free_dat_temp_core(op_dat);

void op_decl_const_core(int dim, char const *type, int typeSize, char *data,
                        char const *name);

void op_printf(const char *format, ...);

void op_print(const char *line);

void op_err_print(const char *error_string, int m, const char *name);

void op_arg_check(op_set, int, op_arg, int *, char const *);

op_arg op_arg_dat_core(op_dat dat, int idx, op_map map, int dim,
                       const char *typ, op_access acc);

op_arg op_opt_arg_dat_core(int opt, op_dat dat, int idx, op_map map, int dim,
                           const char *typ, op_access acc);

op_arg op_arg_gbl_core(int, char *, int, const char *, int, op_access);

void op_diagnostic_output(void);

void op_timing_output_core(void);

void op_timing_output_2_file(const char *);

void op_timings_to_csv(const char *);

void op_timing_realloc(int);

void op_timing_realloc_manytime(int kernel, int num_timers);

void op_timers_core(double *cpu, double *et);

void op_dump_dat(op_dat data);

void op_print_dat_to_binfile_core(op_dat dat, const char *file_name);

void op_print_dat_to_txtfile_core(op_dat dat, const char *file_name);

void op_compute_moment(double t, double *first, double *second);

void op_compute_moment_across_times(double* times, int ntimes, bool ignore_zeros, double *first, double *second);

int op_size_of_set(const char *);

int op_get_size(op_set set);

void check_map(char const *name, op_set from, op_set to, int dim, int *map);

void op_upload_dat(op_dat dat);

void op_download_dat(op_dat dat);

/*******************************************************************************
* Core MPI lib function prototypes
*******************************************************************************/

int op_mpi_halo_exchanges(op_set set, int nargs, op_arg *args);

int op_mpi_halo_exchanges_cuda(op_set set, int nargs, op_arg *args);

void op_mpi_set_dirtybit(int nargs, op_arg *args);

void op_mpi_set_dirtybit_cuda(int nargs, op_arg *args);

void op_mpi_wait_all(int nargs, op_arg *args);

void op_mpi_wait_all_cuda(int nargs, op_arg *args);

void op_mpi_reset_halos(int nargs, op_arg *args);

void op_mpi_reduce_combined(op_arg *args, int nargs);

void op_mpi_reduce_float(op_arg *args, float *data);

void op_mpi_reduce_double(op_arg *args, double *data);

void op_mpi_reduce_int(op_arg *args, int *data);

void op_mpi_reduce_bool(op_arg *args, bool *data);

void op_mpi_barrier();

void op_realloc_comm_buffer(char **send_buffer_host, char **recv_buffer_host, 
      char **send_buffer_device, char **recv_buffer_device, int device, 
      unsigned size_send, unsigned size_recv);
int op_mpi_halo_exchanges_grouped(op_set set, int nargs, op_arg *args, int device);
void op_mpi_wait_all_grouped(int nargs, op_arg *args, int device);


/*******************************************************************************
* Toplevel partitioning selection function - also triggers halo creation
*******************************************************************************/
void op_partition(const char *lib_name, const char *lib_routine,
                  op_set prime_set, op_map prime_map, op_dat coords);

/*******************************************************************************
* Other partitioning related routine prototypes
*******************************************************************************/

void op_partition_destroy();

void *op_mpi_perf_time(const char *name, double time);
#ifdef COMM_PERF
void op_mpi_perf_comms(void *k_i, int nargs, op_arg *args);
#endif

void op_renumber(op_map base);

/*******************************************************************************
* Utility function to compare two op_sets and return 1 if they are identical
*******************************************************************************/
int compare_sets(op_set set1, op_set set2);
op_dat search_dat(op_set set, int dim, char const *type, int size,
                  char const *name);

int op_is_root();

/*******************************************************************************
* Memory allocation functions
*******************************************************************************/
void *op_malloc(size_t size);
void *op_realloc(void *ptr, size_t size);
void op_free(void *ptr);
void *op_calloc(size_t num, size_t size);

void deviceSync();

#ifdef __cplusplus
}
#endif

#endif /* __OP_LIB_CORE_H */
