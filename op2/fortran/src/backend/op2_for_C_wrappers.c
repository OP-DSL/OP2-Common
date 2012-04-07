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
  op_dat tmp;

  char * heapName = (char *) calloc ( strlen ( name ), sizeof ( char ) );
  char * typeName = (char *) calloc ( strlen ( type ), sizeof ( char ) );

  strncpy ( heapName, name, strlen ( name ) );
  strncpy ( typeName, type, strlen ( type ) );

  tmp = op_decl_dat ( set, dim, typeName, size, *data, heapName );

  if (tmp->set == NULL)
    {
      printf ("%s->set is NULL\n", heapName);
      exit (0);
    }

  return tmp;
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


/* 
 * Utility functions: 
 * since op_set/map/dat have become pointers, then from fortran I can't
 * access their fields directly.
 * These routines permit to avoid c_f_pointers in the declaration routines.
 */
int get_set_size (op_set_core * set)
{
  if (set == NULL)
    {
      printf ("Set is NULL\n");
      exit (0);
    }

  return set->size;
}

int get_associated_set_size (op_dat_core * dat)
{
  if (dat == NULL)
    {
      printf ("Dat is NULL\n");
      exit (0);
    }

  if (dat->set == NULL)
    {
      printf ("Set of dat is NULL\n");
      exit (0);
    }

  return dat->set->size;
}


/*
 * For now implementation only for OpenMP
 * We will then need to fix it in the C branch of the lib
*/
void op_get_dat ( op_dat_core * opdat )
{
  (void)opdat;
}

void op_put_dat ( op_dat_core * opdat )
{
  (void)opdat;
}

void dumpOpDat (op_dat_core * data, const char * fileName)
{
  int i;

  FILE * outfile = fopen (fileName, "w+");

  if (outfile == NULL) exit (0);

  if ( data != NULL )
    {
      if ( strncmp ( "real", data->type, 4 ) == 0 )
        for ( i = 0; i < data->dim * data->set->size; i++ )
          fprintf (outfile, "%.10lf\n", ((double *) data->data)[i] );

      else if ( strncmp ( "integer", data->type, 7 ) == 0 )
        for ( i = 0; i < data->dim * data->set->size; i++ )
          fprintf (outfile, "%d\n", ((int *) data->data)[i] );

      else
        {
          printf ( "Unsupported type for dumping %s\n", data->type );
          exit ( 0 );
        }
    }

  fclose (outfile);
}

/* This function does not specialises w.r.t. a sequence number
 * because of the intrinsic complexity of modifying the
 * LOOP macro
 */
void dumpOpDatSequential(char * kernelName, op_dat_core * dat, op_access access, op_map_core * map)
{
  // OP_GBL or read only
  if (access == OP_READ || map->dim == -1) return;

  char * fileName = calloc (strlen(kernelName) + strlen(dat->name), sizeof (char));
  sprintf (fileName, "%s_%s", kernelName, dat->name);

  dumpOpDat (dat, fileName);
}

void dumpOpDatFromDevice (op_dat_core * data, const char * label, int * sequenceNumber)
{
  op_get_dat (data);

  char * fileName = calloc (strlen(label) + log10(*sequenceNumber) + 1, sizeof (char));

  sprintf (fileName, "%s_%d", label, *sequenceNumber);

  printf ("Dumping %s\n", fileName);

  dumpOpDat (data, fileName);
}

void dumpOpGbl (op_dat_core * data)
{
  int i;
  if ( data != NULL )
    {
      if ( strncmp ( "real", data->type, 4 ) == 0 )
        for ( i = 0; i < data->dim * data->set->size; i++ )
          printf ( "%lf\n", ((double *) data->data)[i] );

      else if ( strncmp ( "integer", data->type, 7 ) == 0 )
        for ( i = 0; i < data->dim * data->set->size; i++ )
          printf ( "%d\n", data->data[i] );
      else
        { 
          printf ( "Unsupported type for dumping %s\n", data->type );
          exit ( 0 );
        }
    }    
}



void dumpOpMap (op_map_core * map, const char * fileName)
{
  int i;

  FILE * outfile = fopen (fileName, "w+");

  if (outfile == NULL) exit (0);

  if ( map != NULL )
    {
      for ( i = 0; i < map->dim * map->from->size; i++ )
	fprintf (outfile, "%d\n", ((int *) map->map)[i] );
    }

  fclose (outfile);
}
