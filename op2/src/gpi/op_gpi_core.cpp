#include "gpi_utils.h"

#include <op_lib_c.h>
#include <op_lib_mpi.h>
#include <op_util.h>

/* PROBLEM: op_mpi_core stores all of the global variables that are all of the halo types and dat list.
 * Ideally would be best to move that around a bit so it's common rather than just MPI 
 */



int op_mpi_halo_exchanges(op_set set, int nargs, op_arg *args){}



/* Wait for all args */
void op_gpi_waitall_args(int nargs, op_arg *args){

}