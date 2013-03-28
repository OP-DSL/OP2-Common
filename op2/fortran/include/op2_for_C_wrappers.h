#ifndef __OP2_FOR_C_WRAPPERS_H
#define __OP2_FOR_C_WRAPPERS_H

/*
 * This file declares the C functions declaring OP2 data types in the
 * Fortran OP2 reference library
 */

#include <stdbool.h>

#include <op_lib_core.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Access codes: must have the same values here and in op2_for_declarations.F90 file */
#define FOP_READ 1
#define FOP_WRITE 2
#define FOP_INC 3
#define FOP_RW 4
#define FOP_MIN 5
#define FOP_MAX 6

/* HYDRA feature: in some cases the op_dat
 * passed to an op_arg is NULL, and this index notifies it
 * to the parallel loop implementation
 * The following IDX value is for this case
 */
#define OP_ARG_NULL -3
#define OP_IDX_NULL -3
#define OP_ACC_NULL -3

op_access getAccFromIntCode (int accCode);

op_map_core * op_decl_null_map ( );

op_dat op_decl_gbl_f ( char ** dataIn, int dim, int size, const char * type );

op_arg op_arg_gbl_copy ( char * data, int dim, const char * typ, int size, op_access acc );
//op_arg op_arg_dat_null (op_dat dat, int idx, op_map map, int dim, const char * typ, op_access acc);
void op_dump_arg (op_arg * arg);
void print_type (op_arg * arg);
int op_mpi_size ();
void op_mpi_rank (int * rank);
void op_barrier ();
bool isCNullPointer (void * ptr);
void printFirstDatPosition (op_dat dat);
int setKernelTime (int id, char name[], double kernelTime, float transfer, float transfer2);
void decrement_all_mappings ();
void increment_all_mappings ();

void op_get_dat (op_dat dat);
void op_put_dat (op_dat dat);
void op_get_dat_mpi (op_dat dat);
void op_put_dat_mpi (op_dat dat);

int getSetSizeFromOpArg (op_arg * arg);
int getMapDimFromOpArg (op_arg * arg);

int get_set_size (op_set_core * set);
int get_associated_set_size (op_dat_core * dat);

void op_get_dat ( op_dat_core * opdat );
void op_put_dat ( op_dat_core * opdat );

void dumpOpDat (op_dat_core * data, const char * fileName);
void dumpOpDatSequential(char * kernelName, op_dat_core * dat, op_access access, op_map_core * map);
void dumpOpDatFromDevice (op_dat_core * data, const char * label, int * sequenceNumber);
void dumpOpGbl (op_dat_core * data);
void dumpOpMap (op_map_core * map, const char * fileName);

op_arg
op_arg_gbl_fortran (char * dat, int dim, char * type, int acc);


#ifdef NO_MPI

#else
int op_mpi_size ();

void op_mpi_rank (int * rank);

void op_barrier ();

void printDat_noGather (op_dat dat);
#endif

#ifdef __cplusplus
}
#endif

#endif

