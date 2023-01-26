
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