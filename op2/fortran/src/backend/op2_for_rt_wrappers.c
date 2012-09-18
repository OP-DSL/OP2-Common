#include <op_lib_core.h>
#include <op_rt_support.h>
#include <op_lib_c.h>

#include "../../include/op2_for_rt_wrappers.h"

extern op_plan * OP_plans;

/* These numbers must corresponds to those declared in op2_for_rt_support.f90 */
#define F_OP_ARG_DAT 0
#define F_OP_ARG_GBL 1

#define ERR_INDEX -1

/* Access codes: must have the same values here and in op2_for_declarations.F90 file */
#define FOP_READ 1
#define FOP_WRITE 2
#define FOP_INC 3
#define FOP_RW 4
#define FOP_MIN 5
#define FOP_MAX 6

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
  case FOP_MIN:
    return OP_MIN;
  case FOP_MAX:
    return OP_MAX;
  default:
    return OP_READ; //default case is treated as READ
  }
}

static op_dat find_dat_by_index(int idx)
{
  op_dat_entry *item;
  for (item = TAILQ_FIRST(&OP_dat_list); item != NULL; )
  {
    if (item->dat->index == idx)
    {
      return item->dat;
    }
    item = TAILQ_NEXT(item, entries);
  }
  return NULL;
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
    if ( argsType[i] != F_OP_ARG_GBL )
    {
      op_dat_core * tmp = find_dat_by_index(args[i]);
      planDatArgs[i] = *tmp;
    }
    //      else
    //  planDatArgs[i] = NULL;
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
    if ( argsType[i] != F_OP_ARG_GBL )
      planDims[i] = planDatArgs[i].dim;
    else
      planDims[i] = -1; //difference with C side!
  }

  /* build op_dat data type names (allocate precise space for name and copy it) */
  for ( i = 0; i < argsNumber; i++ )
  {
    if ( argsType[i] != F_OP_ARG_GBL )
    {
      /* obtain reference to next op_dat */
      op_dat_core * tmpDat = find_dat_by_index(args[i]);

      /* allocate space and copy strings */
      int typeNameLen = strlen ( tmpDat->type );

      planTypes[i] = (char * ) calloc ( typeNameLen, sizeof ( char ) );
      strncpy ( planTypes[i], tmpDat->type, typeNameLen );
    }
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
    if ( argsType[i] != F_OP_ARG_GBL )
      planArguments[i].dat = &(planDatArgs[i]);
    else
      planArguments[i].dat = NULL;

    if ( inds[i] >= 0 ) /* Mapping exists only if not in OP_ID or OP_GBL cases */
      planArguments[i].map = &(planMaps[i]);
    else
      planArguments[i].map = NULL;

    planArguments[i].dim = planDims[i];
    planArguments[i].idx = idxs[i];
    if ( argsType[i] != F_OP_ARG_GBL )
    {
      planArguments[i].size = planDatArgs[i].size;
      planArguments[i].data = planDatArgs[i].data;
      planArguments[i].data_d = planDatArgs[i].data_d;
      planArguments[i].type = planDatArgs[i].type;
    }
    else
    {
      planArguments[i].size = 0;
      planArguments[i].data = NULL;
      planArguments[i].data_d = NULL;
      planArguments[i].type = NULL;
    }

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
  int i, nameLength;
  char * heapKernelName;
  op_plan * generatedPlan = NULL;
  op_set_core * iterationSet =  OP_set_list[setId];

  if ( iterationSet == NULL )
  {
     /* TO DO: treat this as an error to be returned to the caller */
    printf ( "bad set index\n" );
    exit ( -1 );
  }


  /* first look for an existing execution plan */

  int ip = 0, match = 0;
  //  printf ("On fortran side kernel name = %s, plan index = %d\n", name, OP_plan_index);
  while ( match == 0 && ip < OP_plan_index )
    {
      //      printf ("Plan: %s, %d, %d, %d, %d\n", OP_plans[ip].name, OP_plans[ip].set->index, OP_plans[ip].nargs, OP_plans[ip].ninds, OP_plans[ip].part_size);
      if ( ( strcmp ( name, OP_plans[ip].name ) == 0 )
     && ( setId == OP_plans[ip].set->index )
     && ( argsNumber == OP_plans[ip].nargs )
     && ( indsNumber == OP_plans[ip].ninds )
     && ( partitionSize == OP_plans[ip].part_size ) )
  {
    match = 1;
    /* for ( int m = 0; m < argsNumber; m++ ) */
    /*   {         */
    /*     /\* Fortran only supports an op_dat: for OP_GBL there is no associated data in the plan and for OP_ID no map *\/ */
    /*     if ( argsType[m] != F_OP_ARG_GBL && inds[m] != -1 ) */
    /*  { */
    /*    match = match  */
    /*      && ( args[m] == OP_plans[ip].dats[m]->index ) */
    /*      && ( maps[m] == OP_plans[ip].maps[m]->index ) */
    /*      && ( idxs[m] == OP_plans[ip].idxs[m] ) */
    /*      && ( accs[m] == OP_plans[ip].accs[m] ); */
    /*  } */
    /*   } */
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
    }


  /* generate the input arguments for the plan function */
  op_arg * planArguments = generatePlanInputData ( name, setId, argsNumber, args, idxs, maps, accs, indsNumber, inds, argsType );


  /*
   * warning: for Hydra, we need to copy the whole mapping because we need to decrement
   * map data read from file, but we want to allocate this memory only if we actually
   * need a new plan
   */
  for ( i = 0; i < argsNumber; i++ ) {
    op_map_core * original;
    int j;

    if ( inds[i] >= 0 ) { // indirect access: there is a map

      original = OP_map_list[maps[i]];

      /* now decrementing */
      (planArguments[i].map)->map = (int *) calloc ( original->dim * original->from->size, sizeof ( int ) );
      for ( j = 0; j < original->dim * original->from->size; j++ )
        (planArguments[i].map)->map[j] = original->map[j] - 1;
    }
  }

  /* store the kernel name on the heap, otherwise will be lost when exiting the caller loop */
  nameLength = strlen(name);
  if ( nameLength <= 0 )
    {
      printf ("Plan caller: bad kernel name\n");
      exit (0);
    }
  heapKernelName = (char *) calloc ( nameLength+1, sizeof(char) );
  strncpy (heapKernelName, name, nameLength);

  //  printf ("Copied name: original<%s>, newone<%s> length %d\n", name, heapKernelName, nameLength);

  /* call the C OP2 core function (we don't need anything else for openmp */
  generatedPlan = op_plan_core ( heapKernelName,
                                 iterationSet,
                                 partitionSize,
                                 argsNumber,
                                 planArguments,
                                 indsNumber,
                                 inds
                               );

  return generatedPlan;
}

