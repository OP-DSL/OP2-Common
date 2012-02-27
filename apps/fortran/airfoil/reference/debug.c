// for copying results in file
#include <stdlib.h>
#include <stdio.h>

#include "debug.h"

// warning: only one file at time!!!
static FILE * myfile;

int openfile ( const char filename[] )
{
  myfile = fopen ( filename, "w" );

  printf ( "file opened" );

  return 0;
}

int closefile ( )
{
  fclose ( myfile );

  printf ( "file closed" );

  return 0;
}

int writerealtofile ( double * value )
{
  fprintf ( myfile, "%lf\n", *value );

  return 0;
}

int writeinttofile ( int * value )
{
  fprintf ( myfile, "%d\n", *value );

  return 0;
}

