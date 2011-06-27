#include <op_lib_core.h>

/*
 * The following include is needed to be able to call the
 * op_plan_get function
*/
#include <op_lib_core.h>
#include <op_rt_support.h>
#include <op_lib_c.h>
#include <op_cuda_rt_support.h>

/*
 * Wrapper for Fortran to plan function for OP2 --> CUDA
 */





op_plan * FortranPlanCallerCUDA ( char name[],
                                  int setId,
                                  int argsNumber,
                                  int args[],
                                  int idxs[],
                                  int maps[],
                                  int accs[],
                                  int indsNumber,
                                  int inds[],
                                  int argsType[],
                                  int partitionSize
                                )
{

  op_plan * generatedPlan = NULL;
  op_set_core * iterationSet =  OP_set_list[setId];

  if ( iterationSet == NULL )
  {
     /* TO DO: treat this as an error to be returned to the caller */
    printf ( "bad set index\n" );
    exit ( -1 );
  }

  /* generate the input arguments for the plan function */
  op_arg * planArguments = generatePlanInputData ( name, setId, argsNumber, args, idxs, maps, accs, indsNumber, inds, argsType );

  /* call the C OP2 function including CUDA movement of data */
  generatedPlan = op_plan_get ( name,
                                iterationSet,
                                partitionSize,
                                argsNumber,
                                planArguments,
                                indsNumber,
                                inds
                              );

  return generatedPlan;
}