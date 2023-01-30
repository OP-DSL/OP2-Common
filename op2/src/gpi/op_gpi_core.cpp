#include "gpi_utils.h"

#include <op_lib_c.h>
#include <op_lib_mpi.h>
#include <op_util.h>

/* PROBLEM: op_mpi_core stores all of the global variables that are all of the halo types and dat list.
 * Ideally would be best to move that around a bit so it's common rather than just MPI 
 */



int op_mpi_halo_exchanges(op_set set, int nargs, op_arg *args){}


void op_gpi_wait_all(int nargs, op_args *args){

}