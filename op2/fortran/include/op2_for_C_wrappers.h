#ifndef __OP2_FOR_C_WRAPPERS_H
#define __OP2_FOR_C_WRAPPERS_H

/*
 * This file declares the C functions declaring OP2 data types in the
 * Fortran OP2 reference library
 */

#include <op_lib_core.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * In all Fortran callers we build name and type strings with the '\0' character
 * at the end. Here we copy them, because in the callers they are allocated onto
 * the stack. An alternative to this is to use dynamic memory allocation of F90
 * to guarantee persistence of name and type strings in the callers.
 */
op_set op_decl_set_f ( int size, char const * name );

op_map op_decl_map_f ( op_set_core * from, op_set_core * to, int dim, int ** imap, char const *name );

op_dat op_decl_dat_f ( op_set set, int dim, char const *type,
                       int size, char ** data, char const *name );

op_map_core * op_decl_null_map ( );

void op_decl_const_f ( int dim, void **dat, char const *name );

op_dat op_decl_gbl_f ( char ** dataIn, int dim, int size, const char * type );

int get_set_size (op_set_core * set);
int get_associated_set_size (op_dat_core * dat);

void op_get_dat ( op_dat_core * opdat );
void op_put_dat ( op_dat_core * opdat );

void dumpOpDat (op_dat_core * data, const char * fileName);
void dumpOpDatSequential(char * kernelName, op_dat_core * dat, op_access access, op_map_core * map);
void dumpOpDatFromDevice (op_dat_core * data, const char * label, int * sequenceNumber);
void dumpOpGbl (op_dat_core * data);
void dumpOpMap (op_map_core * map, const char * fileName);

#ifdef __cplusplus
}
#endif

#endif

