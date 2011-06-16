
#include <op_lib_core.h>
#include <op_rt_support.h>
#include <op_lib_c.h>

/* These numbers must corresponds to those declared in op2_for_rt_support.f90 */
#define F_OP_ARG_DAT 0
#define F_OP_ARG_GBL 1

#define ERR_INDEX -1

/*
 * Wrapper for Fortran to plan function
 */

op_plan * FortranPlanCaller ( char name[],
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

	int i, generatedPlanIndex = ERR_INDEX;

	op_plan * generatedPlan = NULL;

	op_dat_core planDatArgs[argsNumber];
	op_map_core planMaps[argsNumber];
	int planDims[argsNumber];
	char * planTypes[argsNumber];
	op_access planAccs[argsNumber];
  op_arg planArguments[argsNumber];
	op_set_core * iterationSet =  OP_set_list[setId];

	if ( iterationSet == NULL )
	{
		printf ( "bad set index\n" );
		exit ( -1 );
	}

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
		int j;

		if ( maps[i] != -1 ) /* another magic number !!! */
		{
			int iter;
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
		int typeNameLen =	strlen ( tmpDat->type );

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
    planArguments[i].map = &(planMaps[i]);
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

//name,set,part_size,nargs,args,ninds,inds
  generatedPlan = op_plan_core ( name,
						   iterationSet,
						   partitionSize,
               argsNumber,
               planArguments,
						   indsNumber,
						   inds
						 );

	return generatedPlan;

}
