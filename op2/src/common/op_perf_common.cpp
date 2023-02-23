#include <op_lib_c.h>
#include <op_lib_mpi.h>
#include <op_util.h>

/*******************************************************************************
* Data structure to hold communications of an op_dat
*******************************************************************************/
#define NAMESIZE 20
typedef struct {
  // name of this op_dat
  char name[NAMESIZE];
  // size of this op_dat
  int size;
  // index of this op_dat
  int index;
  // total number of times this op_dat was halo exported
  int count;
  // total number of bytes halo exported for this op_dat in this kernel
  int bytes;
} op_dat_comm_info_core;

typedef op_dat_comm_info_core *op_dat_comm_info;


/*******************************************************************************
* Data Type to hold performance measures for communication
*******************************************************************************/

typedef struct {

  UT_hash_handle hh; // with this variable uthash makes this structure hashable

  // name of kernel
  char name[NAMESIZE];
  // total time spent in this kernel (compute + comm - overlap)
  double time;
  // number of times this kernel is called
  int count;
  // number of op_dat indices in this kernel
  int num_indices;
  // array to hold all the op_dat_mpi_comm_info structs for this kernel
  op_dat_comm_info *comm_info;
  // capacity of comm_info array
  int cap;
} op_comm_kernel;


/* table holding communication performance of each loop
   (accessed via a hash of loop name) */
op_comm_kernel *op_comm_kernel_tab = NULL;

/*******************************************************************************
 * Routine to measure timing for an op_par_loop / kernel
 *******************************************************************************/
void *op_comm_perf_time(const char *name, double time) {
  op_comm_kernel *kernel_entry;

  HASH_FIND_STR(op_comm_kernel_tab, name, kernel_entry);
  if (kernel_entry == NULL) {
    kernel_entry = (op_comm_kernel *)xmalloc(sizeof(op_comm_kernel));
    kernel_entry->num_indices = 0;
    kernel_entry->time = 0.0;
    kernel_entry->count = 0;
    strncpy((char *)kernel_entry->name, name, NAMESIZE);
    HASH_ADD_STR(op_comm_kernel_tab, name, kernel_entry);
  }

  kernel_entry->count += 1;
  kernel_entry->time += time;

  return (void *)kernel_entry;
}

#ifdef COMM_PERF // not entirely sure what this bit is for, but mpi references have been removed

/*******************************************************************************
 * Routine to linear search comm_info array in an op_comm_kernel for an op_dat
 *******************************************************************************/
int search_op_comm_kernel(op_dat dat, op_comm_kernel *kernal, int num_indices) {
  for (int i = 0; i < num_indices; i++) {
    if (strcmp((kernal->comm_info[i])->name, dat->name) == 0 &&
        (kernal->comm_info[i])->size == dat->size) {
      return i;
    }
  }

  return -1;
}

/*******************************************************************************
 * Routine to measure message sizes exchanged in an op_par_loop / kernel
 *******************************************************************************/
void op__perf_comm(void *k_i, op_dat dat) {
  halo_list exp_exec_list = OP_export_exec_list[dat->set->index];
  halo_list exp_nonexec_list = OP_export_nonexec_list[dat->set->index];
  int tot_halo_size =
      (exp_exec_list->size + exp_nonexec_list->size) * (size_t)dat->size;

  op_comm_kernel *kernel_entry = (op_comm_kernel *)k_i;
  int num_indices = kernel_entry->num_indices;

  if (num_indices == 0) {
    // set capcity of comm_info array
    kernel_entry->cap = 20;
    op_dat_comm_info dat_comm =
        (op_dat_comm_info)xmalloc(sizeof(op_dat_comm_info_core));
    kernel_entry->comm_info = (op_dat_comm_info *)xmalloc(
        sizeof(op_dat_comm_info *) * (kernel_entry->cap));
    strncpy((char *)dat_comm->name, dat->name, 20);
    dat_comm->size = dat->size;
    dat_comm->index = dat->index;
    dat_comm->count = 0;
    dat_comm->bytes = 0;

    // add first values
    dat_comm->count += 1;
    dat_comm->bytes += tot_halo_size;

    kernel_entry->comm_info[num_indices] = dat_comm;
    kernel_entry->num_indices++;
  } else {
    int index = search_op_comm_kernel(dat, kernel_entry, num_indices);
    if (index < 0) {
      // increase capacity of comm_info array
      if (num_indices >= kernel_entry->cap) {
        kernel_entry->cap = kernel_entry->cap * 2;
        kernel_entry->comm_info = (op_dat_comm_info *)xrealloc(
            kernel_entry->comm_info,
            sizeof(op_dat_comm_info *) * (kernel_entry->cap));
      }

      op_dat_comm_info dat_comm =
          (op_dat_comm_info)xmalloc(sizeof(op_dat_comm_info_core));

      strncpy((char *)dat_comm->name, dat->name, 20);
      dat_comm->size = dat->size;
      dat_comm->index = dat->index;
      dat_comm->count = 0;
      dat_comm->bytes = 0;

      // add first values
      dat_comm->count += 1;
      dat_comm->bytes += tot_halo_size;

      kernel_entry->comm_info[num_indices] = dat_comm;
      kernel_entry->num_indices++;
    } else {
      kernel_entry->comm_info[index]->count += 1;
      kernel_entry->comm_info[index]->bytes += tot_halo_size;
    }
  }
}
#endif

#ifdef COMM_PERF
void op_perf_comms(void *k_i, int nargs, op_arg *args) {

  for (int n = 0; n < nargs; n++) {
    if (args[n].argtype == OP_ARG_DAT && args[n].sent == 2) {
      op_perf_comm(k_i, (&args[n])->dat);
    }
  }
}
#endif