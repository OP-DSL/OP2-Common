#ifndef __OP_OPENMP_RT_SUPPPORT_H
#define __OP_OPENMP_RT_SUPPPORT_H

#include "op_rt_support.h"

op_plan *op_plan_get ( char const *name, op_set set, int part_size,
                       int nargs, op_arg *args, int ninds, int *inds );


#endif
