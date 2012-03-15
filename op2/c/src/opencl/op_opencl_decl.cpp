#include "op_opencl_rt_support.h"

int OP_plan_index = 0;
op_plan * OP_plans;

//
// OpenCL-specific OP2 functions
//

void op_init( int argc, char **argv, int diags ){
  op_init_core( argc, argv, diags );

  OpenCLDeviceInit( argc, argv );

}

op_dat op_decl_dat_char( op_set set, int dim, char const *type,
                        int size, char *data, char const *name ){
  op_dat dat = op_decl_dat_core( set, dim, type, size, data, name );

  op_cpHostToDevice( (cl_mem*) &(dat->data_d ), (void **)&(dat->data), dat->size*set->size);
  return dat;
}

op_plan *op_plan_get( char const *name, op_set set, int part_size,
                     int nargs, op_arg *args, int ninds, int *inds ){
  return op_plan_get_offset( name, set, 0, part_size, nargs, args, ninds, inds);
}

op_plan *op_plan_get_offset( char const * name, op_set set, int set_offset, int part_size,
                            int nargs, op_arg * args, int ninds, int *inds) {
  op_plan *plan = op_plan_core( name, set, set_offset, part_size, nargs, args, ninds, inds );

  // move plan arrays to device if first time

  int ip = 0, match = 0;

  while(!match && ip < OP_plan_index) {
    if(!strcmp(name,     OP_plans[ip].name)
      && set          == OP_plans[ip].set
      && nargs        == OP_plans[ip].nargs
      && ninds        == OP_plans[ip].ninds
      && part_size    == OP_plans[ip].part_size) {
      match = 1;
      for(int arg = 0; arg < nargs; ++arg) {
        match = match && (args[arg].dat == OP_plans[ip].dats[arg])
                      && (args[arg].map == OP_plans[ip].maps[arg])
                      && (args[arg].idx == OP_plans[ip].idxs[arg])
                      && (args[arg].acc == OP_plans[ip].accs[arg]);
      }
    }
    ++ip;
  }

  if(!match) {
    ++OP_plan_index;
    OP_plans = (op_plan *) realloc(OP_plans, OP_plan_index*sizeof(op_plan));
    if(OP_plans == NULL) {
      printf(" op_plan error - error reallocating memory for OP_plans\n");
      exit(-1);
    }
    OP_plans[OP_plan_index-1] = *plan;
  }

  if ( plan->count == 1 ) {
    for ( int m=0; m<ninds; m++ )
      op_mvHostToDevice( (void ** )&(plan->ind_maps[m]), sizeof(int)*plan->nindirect[m]);

    for ( int m=0; m<nargs; m++ )
      if ( plan->loc_maps[m] != NULL )
        op_mvHostToDevice( (void ** )&(plan->loc_maps[m]), sizeof(short)*plan->set->size);



    op_mvHostToDevice( (void ** )&(plan->ind_sizes),sizeof(int)*plan->nblocks *plan->ninds);
    op_mvHostToDevice( (void ** )&(plan->ind_offs), sizeof(int)*plan->nblocks *plan->ninds);
    op_mvHostToDevice( (void ** )&(plan->nthrcol),sizeof(int)*plan->nblocks);
    op_mvHostToDevice( (void ** )&(plan->thrcol ),sizeof(int)*plan->set->size);
    op_mvHostToDevice( (void ** )&(plan->offset ),sizeof(int)*plan->nblocks);
    op_mvHostToDevice( (void ** )&(plan->nelems ),sizeof(int)*plan->nblocks);
    op_mvHostToDevice( (void ** )&(plan->blkmap ),sizeof(int)*plan->nblocks);

  }
  return plan;
}

void op_exit( ){
  cl_int ciErrNum;

  ciErrNum = 0;

  for( int ip=0; ip<OP_plan_index; ip++ ) {
    for ( int m=0; m<OP_plans[ip].ninds; m++ )
      ciErrNum |= clReleaseMemObject( (cl_mem) OP_plans[ip].ind_maps[m] );
    for ( int m=0; m<OP_plans[ip].nargs; m++ )
      if ( OP_plans[ip].loc_maps[m] != NULL )
        ciErrNum |= clReleaseMemObject( (cl_mem) OP_plans[ip].loc_maps[m] );
    ciErrNum |= clReleaseMemObject( (cl_mem) OP_plans[ip].ind_offs );
    ciErrNum |= clReleaseMemObject( (cl_mem) OP_plans[ip].ind_sizes );
    ciErrNum |= clReleaseMemObject( (cl_mem) OP_plans[ip].nthrcol );
    ciErrNum |= clReleaseMemObject( (cl_mem) OP_plans[ip].thrcol );
    ciErrNum |= clReleaseMemObject( (cl_mem) OP_plans[ip].offset );
    ciErrNum |= clReleaseMemObject( (cl_mem) OP_plans[ip].nelems );
    ciErrNum |= clReleaseMemObject( (cl_mem) OP_plans[ip].blkmap );
  }

  for( int i=0; i<OP_dat_index; i++ ) {
    ciErrNum |= clReleaseMemObject( (cl_mem) OP_dat_list[i]->data_d );
  }

  assert_m( ciErrNum == CL_SUCCESS, "error releasing device memory" );

  op_exit_core( );

}



