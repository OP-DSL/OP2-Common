#ifndef __OP_LIB_TUNER_H
#define __OP_LIB_TUNER_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <math.h>

/*
 * The core of the runtime op_tuner
 */

typedef enum {ANY, CPU, GPU, ACCELERATOR} arch;

typedef struct {
  int		block_size;
  int		part_size;
  int   cache_line_size;
  int   op_warpsize;
  arch  architecture;
  char const *name;
  int   loop_tuner;
  int   active;
} op_tuner;

struct node {
  op_tuner *OP_tuner;
  struct node * next;
};

/*
 * method declarations necessary for the op_tuner.
 * variables necessary for the tuners.
 * Also, we are externalizing OP_cache_line_size as it can be
 * manipulated by the runtime op_tuner.
 */

extern int OP_cache_line_size;
//extern node *head, *current;
//extern op_tuner* OP_global_tuner;


#ifdef __cplusplus
extern "C" {
#endif

op_tuner * op_tuner_core(char const *);

op_tuner * op_tuner_get(char const *);

op_tuner * op_create_tuner(char const *);

op_tuner * op_create_global_tuner();

op_tuner * op_get_global_tuner();
#ifdef __cplusplus
}
#endif

#endif
