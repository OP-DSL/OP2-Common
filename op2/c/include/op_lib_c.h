/*
 * This header file defines the user-level OP2 library for
 * the case of C programs.
 * To optimise software engineering activities it is also
 * used by the fortran backends
 */

#ifndef __OP_LIB_C_H
#define __OP_LIB_C_H

#include <op_lib_core.h>

/* identity mapping and global identifier */

#define OP_ID  (op_map) NULL
#define OP_GBL (op_map) NULL

/*
 * external variables declared in op_lib_core.cpp
 */

extern int OP_diags, OP_part_size, OP_block_size;

extern int OP_set_index, OP_set_max,
           OP_map_index, OP_map_max,
           OP_dat_index, OP_dat_max,
           OP_plan_index, OP_plan_max,
           OP_kern_max;

extern op_set * OP_set_list;
extern op_map * OP_map_list;
extern op_dat * OP_dat_list;
extern op_kernel * OP_kernels;

/*
 * declaration of C routines wrapping lower layer implementations (e.g. CUDA, reference, etc..)
 */

#ifdef __cplusplus
extern "C" {
#endif

void op_init ( int, char **, int);

op_set op_decl_set ( int, char const * );

op_map op_decl_map ( op_set, op_set, int, int *, char const * );

op_dat op_decl_dat ( op_set, int, char const *, int, char *, char const * );

void op_decl_const_char ( int, char const *, int, char *, char const * );

op_arg op_arg_dat ( op_dat, int, op_map, int, char const *, op_access );

op_arg op_arg_gbl ( char *, int, char const *, op_access );

void op_fetch_data ( op_dat );

void op_exit (  );

#ifdef __cplusplus
}
#endif

#endif /* __OP_LIB_C_H */

#include "op_rt_support.h"
