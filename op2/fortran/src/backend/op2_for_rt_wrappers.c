#include <op_lib_core.h>
#include <op_rt_support.h>
#include <op_lib_c.h>

#include "../../include/op2_for_rt_wrappers.h"

/* These numbers must corresponds to those declared in op2_for_rt_support.f90 */
#define F_OP_ARG_DAT 0
#define F_OP_ARG_GBL 1

#define ERR_INDEX -1

/* Access codes: must have the same values here and in op2_for_declarations.F90 file */
#define FOP_READ 1
#define FOP_WRITE 2
#define FOP_INC 3
#define FOP_RW 4
#define FOP_MAX 5
#define FOP_MIN 6


/*
 * Small utility for transforming Fortran OP2 access codes into C OP2 access codes
 */
static op_access getAccFromIntCode ( int accCode )
{
  switch ( accCode ) {
  case FOP_READ:
    return OP_READ;
  case FOP_WRITE:
    return OP_WRITE;
  case FOP_RW:
    return OP_RW;
  case FOP_INC:
    return OP_INC;
  default:
    return OP_READ; //default case is treated as READ
  }
}

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
                               )
{
  (void)name;
  (void)setId;
  (void)indsNumber;

  int i;

  op_dat_core * planDatArgs = calloc ( argsNumber, sizeof ( op_dat_core ) );
  op_map_core * planMaps = calloc ( argsNumber, sizeof ( op_map_core ) );
  int * planDims = calloc ( argsNumber, sizeof ( int ) );
  char ** planTypes = calloc ( argsNumber, sizeof ( char * ) );
  op_access * planAccs = calloc ( argsNumber, sizeof ( op_access ) );
  op_arg * planArguments;

  planArguments = calloc ( argsNumber, sizeof ( op_arg ) );


  /* build planDatArgs variable by accessing OP_dat_list with indexes(=positions) in args */
  for ( i = 0; i < argsNumber; i++ )
  {
    op_dat_core * tmp = OP_dat_list[args[i]];
    planDatArgs[i] = *tmp;
  }

  /* build planMaps variables by accessing OP_map_list with indexes(=positions) in args */
  for ( i = 0; i < argsNumber; i++ )
  {
    op_map_core * tmp;

    if ( inds[i] >= 0 ) /* another magic number !!! */
    {
      tmp = OP_map_list[maps[i]];
      planMaps[i] = *tmp;
    }
    else
    {
      /* build false map with index = -1 ... */
      op_map_core * falseMap = (op_map_core *) calloc ( 1, sizeof ( op_map_core ) ); //OP_ID;
      falseMap->index = -1;
      planMaps[i] = *falseMap;

    }
  }

  /* build dimensions of data using op_dat */
  for ( i = 0; i < argsNumber; i++ )
  {
    planDims[i] = planDatArgs[i].dim;
  }

  /* build op_dat data type names (allocate precise space for name and copy it) */
  for ( i = 0; i < argsNumber; i++ )
  {
    /* obtain reference to next op_dat */
    op_dat_core * tmpDat = OP_dat_list[args[i]];

    /* allocate space and copy strings */
    int typeNameLen = strlen ( tmpDat->type );

    planTypes[i] = (char * ) calloc ( typeNameLen, sizeof ( char ) );
    strncpy ( planTypes[i], tmpDat->type, typeNameLen );
  }

  /* build op_access array needed to build the plan */
  for ( i = 0; i < argsNumber; i++ )
  {
    planAccs[i] = getAccFromIntCode ( accs[i] );
  }

  /* now builds op_arg array */
  for ( i = 0; i < argsNumber; i++ )
  {
    planArguments[i].index = -1; //index is not specified nor used for now..
    planArguments[i].dat = &(planDatArgs[i]);

    if ( inds[i] >= 0  ) /* another magic number !!! */
      planArguments[i].map = &(planMaps[i]);
    else
      planArguments[i].map = NULL;

    planArguments[i].dim = planDims[i];
    planArguments[i].idx = idxs[i];
    planArguments[i].size = planDatArgs[i].size;
    planArguments[i].data = planDatArgs[i].data;
    planArguments[i].data_d = planDatArgs[i].data_d;
    planArguments[i].type = planDatArgs[i].type;
    planArguments[i].acc = planAccs[i];

    switch ( argsType[i] )
    {
    case F_OP_ARG_DAT :
      planArguments[i].argtype = OP_ARG_DAT;
      break;
    case F_OP_ARG_GBL :
      planArguments[i].argtype = OP_ARG_GBL;
      break;
    default :
      printf ( "Error while setting argument type\n" );
      exit ( 0 );
    }
  }

  return planArguments;
}


/*
 * Wrapper for Fortran to plan function for OP2 --> OpenMP
 */
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

  /* call the C OP2 core function (we don't need anything else for openmp
   * FIXME: We're passing an offset of 0 since we're running on a single node.
   * Is that always going to be correct? */
  generatedPlan = op_plan_core ( name,
                                 iterationSet,
                                 partitionSize,
                                 0,
                                 argsNumber,
                                 planArguments,
                                 indsNumber,
                                 inds
                               );

  return generatedPlan;
}

