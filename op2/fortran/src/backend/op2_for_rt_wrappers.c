
#include <op_core_lib.h>
#include <op_rt_support.h>


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
							  int inds[]
							)
{

	int i, generatedPlanIndex = ERR;

	op_plan * generatedPlan = NULL;

	op_dat planArgs[argsNumber];
	op_map planMaps[argsNumber];
	int planDims[argsNumber];
	char * planTypes[argsNumber];
	op_access planAccs[argsNumber];
	op_set * iterationSet =  OP_set_list[setId];

	if ( iterationSet == NULL )
	{
		printf ( "bad set index\n" );
		exit ( -1 );
	}

	/* build planArgs variable by accessing OP_dat_list with indexes(=positions) in args */
	for ( i = 0; i < argsNumber; i++ )
	{
		op_dat * tmp = OP_dat_list[args[i]];
		planArgs[i] = *tmp;
	}

	/* build planMaps variables by accessing OP_map_list with indexes(=positions) in args */
	for ( i = 0; i < argsNumber; i++ )
	{
		op_map * tmp;
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
			planMaps[i] = OP_ID;
		}
	}

	/* build dimensions of data using op_dat */
	for ( i = 0; i < argsNumber; i++ )
	{
		planDims[i] = planArgs[i].dim;
	}

	/* build op_dat data type names (allocate precise space for name and copy it) */
	for ( i = 0; i < argsNumber; i++ )
	{
		/* obtain reference to next op_dat */
		op_dat * tmpDat = OP_dat_list[args[i]];

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

	generatedPlan = plan ( name,
						   *iterationSet,
						   argsNumber,
						   planArgs,
						   idxs,
						   planMaps,
						   planDims,
						   (const char **) planTypes,
						   planAccs,
						   indsNumber,
						   inds
						 );

	return generatedPlan;

}
