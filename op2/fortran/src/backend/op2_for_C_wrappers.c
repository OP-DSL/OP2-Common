/*
 * This source file implements all the functions needed
 * to call C core library functions from Fortran
 *
 * It also provides implementation for other
 * Fortran specific core library functions
 */

#include <string.h>

#include <op_lib_c.h>
#include "../../include/op2_for_C_wrappers.h"

op_set op_decl_set_f ( int size, char const * name )
{

  char * heapName = (char *) calloc ( strlen ( name ), sizeof ( char ) );

  strncpy ( heapName, name, strlen ( name ) );

  return op_decl_set ( size, heapName );
}

op_map op_decl_map_f ( op_set_core * from, op_set_core * to, int dim, int ** imap, char const *name )
{

  char * heapName = (char *) calloc ( strlen ( name ), sizeof ( char ) );

  strncpy ( heapName, name, strlen ( name ) );

  return op_decl_map ( from, to, dim, *imap, heapName );
}

op_dat op_decl_dat_f ( op_set set, int dim, char const *type,
                       int size, char ** data, char const *name )
{
  char * heapName = (char *) calloc ( strlen ( name ), sizeof ( char ) );
  char * typeName = (char *) calloc ( strlen ( type ), sizeof ( char ) );

  strncpy ( heapName, name, strlen ( name ) );
  strncpy ( typeName, type, strlen ( type ) );

  return op_decl_dat ( set, dim, typeName, size, *data, heapName );
}

op_map_core * op_decl_null_map ( )
{
  /* must allocate op_set_core instead of op_set, because the latter is actually a pointer to the former */
  op_set nullSet = NULL;
  op_map map = NULL;

  nullSet = (op_set) calloc ( 1, sizeof ( op_set_core ) );
  map = (op_map) malloc(sizeof(op_map_core));

  nullSet->size = 0;
  nullSet->name = NULL;

  map->from = nullSet;
  map->to = nullSet;
  map->dim = 0; /* set to the proper value in Fortran */
  map->map = NULL;

  return map;
}

void op_decl_const_f ( int dim, void **dat, char const *name )
{
  (void)dat;

  if ( dim <= 0 )
  {
    printf ( "op_decl_const error -- negative/zero dimension for const: %s\n", name );
    exit ( -1 );
  }
}

op_dat op_decl_gbl_f ( char ** dataIn, int dim, int size, const char * type )
{
  op_dat_core * dataOut = calloc ( 1, sizeof ( op_dat_core ) );

  char * typeName = (char *) calloc ( strlen ( type ), sizeof ( char ) );

  strncpy ( typeName, type, strlen ( type ) );

  dataOut->index = -1;
  dataOut->set = NULL;
  dataOut->dim = dim;
  dataOut->size = size * dim;
  dataOut->data = *dataIn;
  dataOut->data_d = NULL;
  dataOut->type = typeName;
  dataOut->name = NULL;

  return dataOut;
}

