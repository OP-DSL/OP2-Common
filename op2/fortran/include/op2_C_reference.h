#ifndef __OP2_C_REFERENCE_H
#define __OP2_C_REFERENCE_H

/*
 * This file declares the C functions for invocation of parallel loops
 * in the Fortran OP2 reference library
 */

#include <op_lib_core.h>
#include "op2_reference_macros.h"
#include "op2_for_C_wrappers.h"

#ifdef __cplusplus
extern "C" {
#endif

void op_arg_set(int n, op_arg arg, char **p_arg, int halo);

void op_arg_copy_in(int n, op_arg arg, char **p_arg);

void op_args_check(op_set set, int nargs, op_arg *args,
  int *ninds);

#define CHARP_LIST(N) COMMA_LIST(N,CHARP)
#define CHARP(x) char*

#define ARG_LIST(N) COMMA_LIST(N,ARGS)
#define ARGS(x) op_arg * arg##x

#define OP_LOOP_DEC(N) \
  void op_par_loop_##N(void (*kernel)(CHARP_LIST(N)), op_set_core * set, ARG_LIST(N));

OP_LOOP_DEC(1)  OP_LOOP_DEC(2)  OP_LOOP_DEC(3)  OP_LOOP_DEC(4)  OP_LOOP_DEC(5)  OP_LOOP_DEC(6)  OP_LOOP_DEC(7)  OP_LOOP_DEC(8)  OP_LOOP_DEC(9)  OP_LOOP_DEC(10)
OP_LOOP_DEC(11) OP_LOOP_DEC(12) OP_LOOP_DEC(13) OP_LOOP_DEC(14) OP_LOOP_DEC(15) OP_LOOP_DEC(16) OP_LOOP_DEC(17) OP_LOOP_DEC(18) OP_LOOP_DEC(19) OP_LOOP_DEC(20)
OP_LOOP_DEC(21) OP_LOOP_DEC(22) OP_LOOP_DEC(23) OP_LOOP_DEC(24) OP_LOOP_DEC(25) OP_LOOP_DEC(26) OP_LOOP_DEC(27) OP_LOOP_DEC(28) OP_LOOP_DEC(29) OP_LOOP_DEC(30)
OP_LOOP_DEC(31) OP_LOOP_DEC(32) OP_LOOP_DEC(33) OP_LOOP_DEC(34) OP_LOOP_DEC(35) OP_LOOP_DEC(36) OP_LOOP_DEC(37) OP_LOOP_DEC(38) OP_LOOP_DEC(39) OP_LOOP_DEC(40) OP_LOOP_DEC(41) OP_LOOP_DEC(42) OP_LOOP_DEC(43)


#ifdef __cplusplus
}
#endif

#endif

