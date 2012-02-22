#ifndef __OP2_FOR_RT_WRAPPERS__
#define __OP2_FOR_RT_WRAPPERS__

#ifdef __cplusplus
extern "C" {
#endif

op_arg * generatePlanInputData ( char name[],
                                 int setId,
                                 int argsNumber,
                                 int args[],
                                 int idxs[],
                                 int maps[],
                                 int accs[],
                                 int indsNumber,
                                 int inds[],
                                 int argsType[]
                               );

op_plan * FortranPlanCallerOpenMP ( char name[],
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
                                  );

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
                                );

#ifdef __cplusplus
}
#endif

#endif

