#ifndef __OP2_FORTRAN_AIRFOIL_DEBUG_H
#define __OP2_FORTRAN_AIRFOIL_DEBUG_H

int openfile ( const char filename[] );
int closefile ( );
int writerealtofile ( double * value );
int writeinttofile ( int * value );

#endif
