#ifndef __OP2_FOR_RT_WRAPPERS__
#define __OP2_FOR_RT_WRAPPERS__

#ifdef __cplusplus
extern "C" {
#endif

void op_partition_wrapper (const char* lib_name, const char* lib_routine,
  op_set prime_set, op_map prime_map, op_dat coords);

void FortranToCMapping (op_arg * arg);

void checkCMapping (op_arg arg);

op_plan * checkExistingPlan (char name[], op_set set,
  int partitionSize, int argsNumber, op_arg args[],
  int indsNumber, int inds[]);

/*
 *  These functions scan all declared mappings and decrement/increment their
 *  values by one (used for FORTRAN <--> C conversions)
 */

void decrement_all_mappings ();

void increment_all_mappings ();

op_plan * FortranPlanCaller (char name[], op_set set,
  int partitionSize, int argsNumber, op_arg args[],
  int indsNumber, int inds[]);

#ifdef __cplusplus
}
#endif

#endif

