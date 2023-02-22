#include "gpi_utils.h"

#include <op_lib_c.h>
#include <op_lib_mpi.h>
#include <op_util.h>

#include <op_gpi_core.h>

/* PROBLEM: op_mpi_core stores all of the global variables that are all of the halo types and dat list.
 * Ideally would be best to move that around a bit so it's common rather than just MPI 
 */


double t1, t2, c1, c2;


/* Segment pointers */
char *eeh_segment_ptr;
char *ieh_segment_ptr;
char *enh_segment_ptr;
char *inh_segment_ptr;

char *msc_segment_ptr;


/* IS_COMMON 
 * Bascially a near perfect copy of op_mpi_halo_exchanges, only 2/3 lines changed. */
int op_gpi_halo_exchanges(op_set set, int nargs, op_arg *args){
  int size = set->size;
  int direct_flag = 1;

  if (OP_diags > 0) {
    int dummy;
    for (int n = 0; n < nargs; n++)
      op_arg_check(set, n, args[n], &dummy, "halo_exchange mpi");
  }

  if (OP_hybrid_gpu) {
    for (int n = 0; n < nargs; n++)
      if (args[n].opt && args[n].argtype == OP_ARG_DAT &&
          args[n].dat->dirty_hd == 2) {
        op_download_dat(args[n].dat);
        args[n].dat->dirty_hd = 0;
      }
  }

  // check if this is a direct loop
  for (int n = 0; n < nargs; n++)
    if (args[n].opt && args[n].argtype == OP_ARG_DAT && args[n].idx != -1)
      direct_flag = 0;

  if (direct_flag == 1)
    return size;

  // not a direct loop ...
  int exec_flag = 0;
  for (int n = 0; n < nargs; n++) {
    if (args[n].opt && args[n].idx != -1 && args[n].acc != OP_READ) {
      size = set->size + set->exec_size;
      exec_flag = 1;
    }
  }
  op_timers_core(&c1, &t1);
  for (int n = 0; n < nargs; n++) {
    if (args[n].opt && args[n].argtype == OP_ARG_DAT) {
      if (args[n].map == OP_ID) {
        op_gpi_exchange_halo(&args[n], exec_flag);
      } else {
        // Check if dat-map combination was already done or if there is a
        // mismatch (same dat, diff map)
        int found = 0;
        int fallback = 0;
        for (int m = 0; m < nargs; m++) {
          if (m < n && args[n].dat == args[m].dat && args[n].map == args[m].map)
            found = 1;
          else if (args[n].dat == args[m].dat && args[n].map != args[m].map)
            fallback = 1;
        }
        // If there was a map mismatch with other argument, do full halo
        // exchange
        if (fallback)
          op_gpi_exchange_halo(&args[n], exec_flag);
        else if (!found) { // Otherwise, if partial halo exchange is enabled for
                           // this map, do it
          if (OP_map_partial_exchange[args[n].map->index])
            op_gpi_exchange_halo_partial(&args[n], exec_flag);
          else
            op_gpi_exchange_halo(&args[n], exec_flag);
        }
      }
    }
  }
  op_timers_core(&c2, &t2);
  if (OP_kern_max > 0)
    OP_kernels[OP_kern_curr].gpi_time += t2 - t1;
  return size;
}



/* Wait for all args.
 * IS_COMMON
 * GPI replacement for op_mpi_wait_all.
 * MPI has a bad naming convention for this stuff, thus tried to clarify with _args */
void op_gpi_waitall_args(int nargs, op_arg *args){
    op_timers_core(&c1, &t1);
    for (int n = 0; n < nargs; n++) {
        op_gpi_waitall(&args[n]);
    }
    op_timers_core(&c2, &t2);
    if (OP_kern_max > 0)
        OP_kernels[OP_kern_curr].gpi_time += t2 - t1;
}


void op_gpi_reduce_combined(op_arg *args, int nargs){

}

void op_gpi_reduce_float(op_arg *arg, float *data){

}

void op_gpi_reduce_double(op_arg *arg, double *data){

}

void op_gpi_reduce_int(op_arg *arg, int *data){

}

void op_gpi_reduce_bool(op_arg *arg, bool *data){

}