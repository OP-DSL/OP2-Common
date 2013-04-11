/*
 * Open source copyright declaration based on BSD open source template:
 * http://www.opensource.org/licenses/bsd-license.php
 *
 * This file is part of the OP2 distribution.
 *
 * Copyright (c) 2011, Mike Giles and others. Please see the AUTHORS file in
 * the main source directory for a full list of copyright holders.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * The name of Mike Giles may not be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <op_lib_core.h>
#include <op_rt_support.h>
#include <op_lib_c.h>

#include "../include/op2_for_C_wrappers.h"
#include "../include/op2_for_rt_wrappers.h"

extern int OP_plan_index, OP_plan_max;
extern op_plan * OP_plans;

#define ERR_INDEX -1

void op_partition_wrapper (const char* lib_name, const char* lib_routine,
  op_set prime_set, op_map prime_map, op_dat coords) {
  op_partition (lib_name, lib_routine, prime_set, prime_map, coords);
}

void FortranToCMapping (op_arg * arg) {
  op_map newMapStruct = (op_map) calloc (1, sizeof(op_map_core));
  int * newMap = (int *) calloc (arg->map->from->size * arg->map->dim, sizeof (int));

  for ( int i = 0; i < arg->map->from->size * arg->map->dim; i++ )
    newMap[i] = arg->map->map[i] -1;

  // do not deallocate the old map, because it could be used elsewhere
  // - only get a new map in this op_arg
  newMapStruct->map = newMap;
  newMapStruct->index = arg->map->index;
  newMapStruct->from = arg->map->from;
  newMapStruct->to = arg->map->to;
  newMapStruct->dim = arg->map->dim;
  newMapStruct->name = arg->map->name;
  newMapStruct->user_managed = arg->map->user_managed;
  arg->map = newMapStruct;
}

void checkCMapping (op_arg arg) {
  for ( int i = 0; i < arg.map->from->size * arg.map->dim; i++ ) {
    if ( arg.map->map[i] >= arg.map->to->size ) {
      printf ("Invalid mapping 1\n");
      exit (0);
    }
    if ( arg.map->map[i] < 0 ) {
      printf ("Invalid mapping 2, value is %d\n", arg.map->map[i]);
      exit (0);
    }
  }
}

op_plan * checkExistingPlan (char name[], op_set set,
  int partitionSize, int argsNumber, op_arg args[],
  int indsNumber, int inds[]) {

  (void)inds;
  (void)name;

  int match =0, ip = 0;

  while ( match == 0 && ip < OP_plan_index )
  {
    if ( //( strcmp ( name, OP_plans[ip].name ) == 0 )
        ( set == OP_plans[ip].set )
        && ( argsNumber == OP_plans[ip].nargs )
        && ( indsNumber == OP_plans[ip].ninds )
        && ( partitionSize == OP_plans[ip].part_size ) )
    {
      match = 1;
      for ( int m = 0; m < argsNumber; m++ )
      {
        match = match && ( args[m].dat == OP_plans[ip].dats[m] )
//          && ( args[m].map->index == OP_plans[ip].maps[m]->index )
          && ( args[m].idx == OP_plans[ip].idxs[m] )
          && ( args[m].acc == OP_plans[ip].accs[m] );
      }
    }
    ip++;
  }

  if ( match )
  {
    ip--;
    if ( OP_diags > 3 )
      printf ( " old execution plan #%d\n", ip );
    OP_plans[ip].count++;
    return &( OP_plans[ip] );
  } else
    return NULL;
}


op_plan * FortranPlanCaller (char name[], op_set set,
  int partitionSize, int argsNumber, op_arg args[],
  int indsNumber, int inds[]) {

  op_plan * generatedPlan = NULL;

  generatedPlan = checkExistingPlan (name, set,
    partitionSize, argsNumber, args,
    indsNumber, inds);

  if ( generatedPlan != NULL ) return generatedPlan;

  /* copy the name because FORTRAN doesn't allow allocating
     strings */
  int nameLen = strlen (name);
  char * heapName = (char *) calloc (nameLen, sizeof(char));
  strncpy (heapName, name, nameLen);

  /* call the C OP2 function */
  generatedPlan = op_plan_get (heapName, set, partitionSize,
    argsNumber, args, indsNumber, inds);

  return generatedPlan;
}


int getSetSizeFromOpArg (op_arg * arg)
{
  return arg->opt ? arg->dat->set->size : 0;
}

int getMapDimFromOpArg (op_arg * arg)
{
  return (arg->opt && arg->map!=NULL) ? arg->map->dim : 0;
}

int reductionSize (op_arg *args, int nargs)
{
  int max_size = 0;
  for (int i = 0; i < nargs; i++) {
    if (args[i].argtype == OP_ARG_GBL && (args[i].acc == OP_INC || args[i].acc == OP_MAX || args[i].acc == OP_MIN))
      max_size = max_size > args[i].size ? max_size : args[i].size;
  }
  return max_size;
}
