
/* 
 * This file implements the user-level OP2 functions for the case
 * of the mpi back-end
 */

#include <mpi.h>
#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_rt_support.h>
#include <op_mpi_core.h>

/* 
 * Routines called by user code and kernels
 * these wrappers are used by non-CUDA versions
 * op_lib.cu provides wrappers for CUDA version
 */

void op_init ( int argc, char ** argv, int diags )
{
	op_init_core ( argc, argv, diags );
}

op_dat op_decl_dat ( op_set set, int dim, char const * type, int size, char * data, char const *name )
{
	return op_decl_dat_core ( set, dim, type, size, data, name );
}

void op_fetch_data ( op_dat dat )
{
  (void)dat;
}

/*
 * No specific action is required for constants in OpenMP
 */

void op_decl_const_char ( int dim, char const * type, int typeSize, char * data, char const * name )
{
  (void)dim;
  (void)type;
  (void)typeSize;
  (void)data;
  (void)name;
}

op_plan *
op_plan_get ( char const * name, op_set set, int part_size,
              int nargs, op_arg * args, int ninds, int *inds )
{
	return op_plan_get_offset ( name, set, 0, part_size,
		nargs, args, ninds, inds );
}

op_plan *
op_plan_get_offset ( char const * name, op_set set, int set_offset, int part_size,
              int nargs, op_arg * args, int ninds, int *inds )
{   
    return op_plan_core ( name, set, set_offset, part_size, nargs, args, ninds, inds );
}

void op_exit()
{
    op_mpi_exit();
    op_rt_exit();
    op_exit_core();
}

/* 
 * Wrappers of core lib
 */

op_set op_decl_set(int size, char const *name )
{
	return op_decl_set_core ( size, name );
}


op_map op_decl_map(op_set from, op_set to, int dim, int * imap, char const * name )
{
	return op_decl_map_core ( from, to, dim, imap, name );
}


op_arg op_arg_dat( op_dat dat, int idx, op_map map, int dim, char const * type, op_access acc )
{
	return op_arg_dat_core ( dat, idx, map, dim, type, acc );
}


op_arg op_arg_gbl( char * data, int dim, const char * type, op_access acc )
{
	return op_arg_gbl ( data, dim, type, acc );
}
