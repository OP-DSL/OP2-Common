#include "op_lib_core.h"

/*
 * This file implements thw wrappers of core library routines for
 * the OP2 reference implementation. These functions are also used
 * by the Fortran OP2 reference implementation.
 */


void op_init ( int argc, char ** argv, int diags )
{
  op_init_core ( argc, argv, diags );
}


op_set op_decl_set ( int size, char const * name )
{
  return op_decl_set_core ( size, name );
}


op_map op_decl_map ( op_set from, op_set to, int dim, int * imap, char const * name )
{
  return op_decl_map_core ( from, to, dim, imap, name );
}


op_dat op_decl_dat ( op_set set, int dim, char const * type, int size, char * data, char const * name )
{
  return op_decl_dat_core ( set, dim, type, size, data, name );
}

//
// The following function is empty for the reference implementation
// and is not present in the core library: it is only needed
// in OP2 main program to signal OP2 compilers which are the constant
// names in the program
//

void op_decl_const_char ( int dim, char const * type, int typeSize, char * data, char const * name ) {}


op_arg op_arg_dat ( op_dat dat, int idx, op_map map, int dim, char const * type, op_access acc )
{
  return op_arg_dat_core ( dat, idx, map, dim, type, acc );
}


op_arg op_arg_gbl ( char * data, int dim, const char * type, op_access acc )
{
  return op_arg_gbl ( data, dim, type, acc );
}


void op_fetch_data ( op_dat a ) {}


void op_exit ()
{
  op_exit_core ();
}
