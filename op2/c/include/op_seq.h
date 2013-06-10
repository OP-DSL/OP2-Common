
//
// header for sequential and MPI+sequentional execution
//

#include "op_lib_cpp.h"

static int op2_stride = 1;
#define OP2_STRIDE(arr, idx) arr[idx]

// scratch space to use for double counting in indirect reduction
static int blank_args_size = 512;
static char* blank_args = (char *)malloc(blank_args_size);

inline void op_arg_set(int n, op_arg arg, char **p_arg, int halo){
  *p_arg = arg.data;

  if (arg.argtype==OP_ARG_GBL) {
    if (halo && (arg.acc != OP_READ)) *p_arg = blank_args;
  }
  else {
    if (arg.map==NULL)         // identity mapping
      *p_arg += arg.size*n;
    else                       // standard pointers
      *p_arg += arg.size*arg.map->map[arg.idx+n*arg.map->dim];
  }
}

inline void op_arg_copy_in(int n, op_arg arg, char **p_arg) {
  for (int i = 0; i < -1*arg.idx; ++i)
    p_arg[i] = arg.data + arg.map->map[i+n*arg.map->dim]*arg.size;
}

inline void op_args_check(op_set set, int nargs, op_arg *args,
                                      int *ninds, const char *name) {
  for (int n=0; n<nargs; n++)
    op_arg_check(set,n,args[n],ninds,name);
}

//
//op_par_loop routine for 1 arguments
//
template <class T0>
void op_par_loop(void (*kernel)(T0*),
    char const * name, op_set set,
    op_arg arg0){

  char *p_a[1] = {0};
  op_arg args[1] = {arg0};
  if(arg0.idx < -1) {
    p_a[0] = (char *)malloc(-1*args[0].idx*sizeof(T0));
  }

  //allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i<1;i++)
    if(args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size )
    {
      blank_args_size = args[i].size;
      blank_args = (char *)malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags>0) op_args_check(set,1,args,&ninds,name);

  if (OP_diags>2) {
  if (ninds==0)
    printf(" kernel routine w/o indirection:  %s\n",name);
  else
    printf(" kernel routine with indirection: %s\n",name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 1, args);

  // loop over set elements
  int halo = 0;

  for (int n=0; n<n_upper; n++) {
    if (n==set->core_size) op_mpi_wait_all(1,args);
    if (n==set->size) halo = 1;
    if (args[0].idx < -1) op_arg_copy_in(n,args[0], (char **)p_a[0]);
    else op_arg_set(n,args[0], &p_a[0],halo);

    kernel( (T0 *)p_a[0]);
  }
  if ( n_upper == set->core_size || n_upper == 0 )
    op_mpi_wait_all (1,args);

  //set dirty bit on datasets touched
  op_mpi_set_dirtybit(1, args);

  //global reduction for MPI execution, if needed
  //p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0,(T0 *)p_a[0]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 1, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if(arg0.idx < -1) {
    free(p_a[0]);
  }
}
//
//op_par_loop routine for 2 arguments
//
template <class T0,class T1>
void op_par_loop(void (*kernel)(T0*, T1*),
    char const * name, op_set set,
    op_arg arg0, op_arg arg1){

  char *p_a[2] = {0,0};
  op_arg args[2] = {arg0, arg1};
  if(arg0.idx < -1) {
    p_a[0] = (char *)malloc(-1*args[0].idx*sizeof(T0));
  }
  if(arg1.idx < -1) {
    p_a[1] = (char *)malloc(-1*args[1].idx*sizeof(T1));
  }

  //allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i<2;i++)
    if(args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size )
    {
      blank_args_size = args[i].size;
      blank_args = (char *)malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags>0) op_args_check(set,2,args,&ninds,name);

  if (OP_diags>2) {
  if (ninds==0)
    printf(" kernel routine w/o indirection:  %s\n",name);
  else
    printf(" kernel routine with indirection: %s\n",name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 2, args);

  // loop over set elements
  int halo = 0;

  for (int n=0; n<n_upper; n++) {
    if (n==set->core_size) op_mpi_wait_all(2,args);
    if (n==set->size) halo = 1;
    if (args[0].idx < -1) op_arg_copy_in(n,args[0], (char **)p_a[0]);
    else op_arg_set(n,args[0], &p_a[0],halo);
    if (args[1].idx < -1) op_arg_copy_in(n,args[1], (char **)p_a[1]);
    else op_arg_set(n,args[1], &p_a[1],halo);

    kernel( (T0 *)p_a[0], (T1 *)p_a[1]);
  }
  if ( n_upper == set->core_size || n_upper == 0 )
    op_mpi_wait_all (2,args);

  //set dirty bit on datasets touched
  op_mpi_set_dirtybit(2, args);

  //global reduction for MPI execution, if needed
  //p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0,(T0 *)p_a[0]);
  op_mpi_reduce(&arg1,(T1 *)p_a[1]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 2, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if(arg0.idx < -1) {
    free(p_a[0]);
  }
  if(arg1.idx < -1) {
    free(p_a[1]);
  }
}
//
//op_par_loop routine for 3 arguments
//
template <class T0,class T1,class T2>
void op_par_loop(void (*kernel)(T0*, T1*, T2*),
    char const * name, op_set set,
    op_arg arg0, op_arg arg1, op_arg arg2){

  char *p_a[3] = {0,0,0};
  op_arg args[3] = {arg0, arg1, arg2};
  if(arg0.idx < -1) {
    p_a[0] = (char *)malloc(-1*args[0].idx*sizeof(T0));
  }
  if(arg1.idx < -1) {
    p_a[1] = (char *)malloc(-1*args[1].idx*sizeof(T1));
  }
  if(arg2.idx < -1) {
    p_a[2] = (char *)malloc(-1*args[2].idx*sizeof(T2));
  }

  //allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i<3;i++)
    if(args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size )
    {
      blank_args_size = args[i].size;
      blank_args = (char *)malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags>0) op_args_check(set,3,args,&ninds,name);

  if (OP_diags>2) {
  if (ninds==0)
    printf(" kernel routine w/o indirection:  %s\n",name);
  else
    printf(" kernel routine with indirection: %s\n",name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 3, args);

  // loop over set elements
  int halo = 0;

  for (int n=0; n<n_upper; n++) {
    if (n==set->core_size) op_mpi_wait_all(3,args);
    if (n==set->size) halo = 1;
    if (args[0].idx < -1) op_arg_copy_in(n,args[0], (char **)p_a[0]);
    else op_arg_set(n,args[0], &p_a[0],halo);
    if (args[1].idx < -1) op_arg_copy_in(n,args[1], (char **)p_a[1]);
    else op_arg_set(n,args[1], &p_a[1],halo);
    if (args[2].idx < -1) op_arg_copy_in(n,args[2], (char **)p_a[2]);
    else op_arg_set(n,args[2], &p_a[2],halo);

    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2]);
  }
  if ( n_upper == set->core_size || n_upper == 0 )
    op_mpi_wait_all (3,args);

  //set dirty bit on datasets touched
  op_mpi_set_dirtybit(3, args);

  //global reduction for MPI execution, if needed
  //p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0,(T0 *)p_a[0]);
  op_mpi_reduce(&arg1,(T1 *)p_a[1]);
  op_mpi_reduce(&arg2,(T2 *)p_a[2]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = (void *)op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 3, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if(arg0.idx < -1) {
    free(p_a[0]);
  }
  if(arg1.idx < -1) {
    free(p_a[1]);
  }
  if(arg2.idx < -1) {
    free(p_a[2]);
  }
}
//
//op_par_loop routine for 4 arguments
//
template <class T0,class T1,class T2,class T3>
void op_par_loop(void (*kernel)(T0*, T1*, T2*, T3*),
    char const * name, op_set set,
    op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3){

  char *p_a[4] = {0,0,0,0};
  op_arg args[4] = {arg0, arg1, arg2, arg3};
  if(arg0.idx < -1) {
    p_a[0] = (char *)malloc(-1*args[0].idx*sizeof(T0));
  }
  if(arg1.idx < -1) {
    p_a[1] = (char *)malloc(-1*args[1].idx*sizeof(T1));
  }
  if(arg2.idx < -1) {
    p_a[2] = (char *)malloc(-1*args[2].idx*sizeof(T2));
  }
  if(arg3.idx < -1) {
    p_a[3] = (char *)malloc(-1*args[3].idx*sizeof(T3));
  }

  //allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i<4;i++)
    if(args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size )
    {
      blank_args_size = args[i].size;
      blank_args = (char *)malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags>0) op_args_check(set,4,args,&ninds,name);

  if (OP_diags>2) {
  if (ninds==0)
    printf(" kernel routine w/o indirection:  %s\n",name);
  else
    printf(" kernel routine with indirection: %s\n",name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 4, args);

  // loop over set elements
  int halo = 0;

  for (int n=0; n<n_upper; n++) {
    if (n==set->core_size) op_mpi_wait_all(4,args);
    if (n==set->size) halo = 1;
    if (args[0].idx < -1) op_arg_copy_in(n,args[0], (char **)p_a[0]);
    else op_arg_set(n,args[0], &p_a[0],halo);
    if (args[1].idx < -1) op_arg_copy_in(n,args[1], (char **)p_a[1]);
    else op_arg_set(n,args[1], &p_a[1],halo);
    if (args[2].idx < -1) op_arg_copy_in(n,args[2], (char **)p_a[2]);
    else op_arg_set(n,args[2], &p_a[2],halo);
    if (args[3].idx < -1) op_arg_copy_in(n,args[3], (char **)p_a[3]);
    else op_arg_set(n,args[3], &p_a[3],halo);

    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3]);
  }
  if ( n_upper == set->core_size || n_upper == 0 )
    op_mpi_wait_all (4,args);

  //set dirty bit on datasets touched
  op_mpi_set_dirtybit(4, args);

  //global reduction for MPI execution, if needed
  //p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0,(T0 *)p_a[0]);
  op_mpi_reduce(&arg1,(T1 *)p_a[1]);
  op_mpi_reduce(&arg2,(T2 *)p_a[2]);
  op_mpi_reduce(&arg3,(T3 *)p_a[3]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 4, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if(arg0.idx < -1) {
    free(p_a[0]);
  }
  if(arg1.idx < -1) {
    free(p_a[1]);
  }
  if(arg2.idx < -1) {
    free(p_a[2]);
  }
  if(arg3.idx < -1) {
    free(p_a[3]);
  }
}
//
//op_par_loop routine for 5 arguments
//
template <class T0,class T1,class T2,class T3,
          class T4>
void op_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                                T4*),
    char const * name, op_set set,
    op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3,
    op_arg arg4){

  char *p_a[5] = {0,0,0,0,0};
  op_arg args[5] = {arg0, arg1, arg2, arg3,
                    arg4};
  if(arg0.idx < -1) {
    p_a[0] = (char *)malloc(-1*args[0].idx*sizeof(T0));
  }
  if(arg1.idx < -1) {
    p_a[1] = (char *)malloc(-1*args[1].idx*sizeof(T1));
  }
  if(arg2.idx < -1) {
    p_a[2] = (char *)malloc(-1*args[2].idx*sizeof(T2));
  }
  if(arg3.idx < -1) {
    p_a[3] = (char *)malloc(-1*args[3].idx*sizeof(T3));
  }
  if(arg4.idx < -1) {
    p_a[4] = (char *)malloc(-1*args[4].idx*sizeof(T4));
  }

  //allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i<5;i++)
    if(args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size )
    {
      blank_args_size = args[i].size;
      blank_args = (char *)malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags>0) op_args_check(set,5,args,&ninds,name);

  if (OP_diags>2) {
  if (ninds==0)
    printf(" kernel routine w/o indirection:  %s\n",name);
  else
    printf(" kernel routine with indirection: %s\n",name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 5, args);

  // loop over set elements
  int halo = 0;

  for (int n=0; n<n_upper; n++) {
    if (n==set->core_size) op_mpi_wait_all(5,args);
    if (n==set->size) halo = 1;
    if (args[0].idx < -1) op_arg_copy_in(n,args[0], (char **)p_a[0]);
    else op_arg_set(n,args[0], &p_a[0],halo);
    if (args[1].idx < -1) op_arg_copy_in(n,args[1], (char **)p_a[1]);
    else op_arg_set(n,args[1], &p_a[1],halo);
    if (args[2].idx < -1) op_arg_copy_in(n,args[2], (char **)p_a[2]);
    else op_arg_set(n,args[2], &p_a[2],halo);
    if (args[3].idx < -1) op_arg_copy_in(n,args[3], (char **)p_a[3]);
    else op_arg_set(n,args[3], &p_a[3],halo);
    if (args[4].idx < -1) op_arg_copy_in(n,args[4], (char **)p_a[4]);
    else op_arg_set(n,args[4], &p_a[4],halo);

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
          (T4 *)p_a[4]);
  }
  if ( n_upper == set->core_size || n_upper == 0 )
    op_mpi_wait_all (5,args);

  //set dirty bit on datasets touched
  op_mpi_set_dirtybit(5, args);

  //global reduction for MPI execution, if needed
  //p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0,(T0 *)p_a[0]);
  op_mpi_reduce(&arg1,(T1 *)p_a[1]);
  op_mpi_reduce(&arg2,(T2 *)p_a[2]);
  op_mpi_reduce(&arg3,(T3 *)p_a[3]);
  op_mpi_reduce(&arg4,(T4 *)p_a[4]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 5, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if(arg0.idx < -1) {
    free(p_a[0]);
  }
  if(arg1.idx < -1) {
    free(p_a[1]);
  }
  if(arg2.idx < -1) {
    free(p_a[2]);
  }
  if(arg3.idx < -1) {
    free(p_a[3]);
  }
  if(arg4.idx < -1) {
    free(p_a[4]);
  }
}
//
//op_par_loop routine for 6 arguments
//
template <class T0,class T1,class T2,class T3,
          class T4,class T5>
void op_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                                T4*, T5*),
    char const * name, op_set set,
    op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3,
    op_arg arg4, op_arg arg5){

  char *p_a[6] = {0,0,0,0,0,0};
  op_arg args[6] = {arg0, arg1, arg2, arg3,
                    arg4, arg5};
  if(arg0.idx < -1) {
    p_a[0] = (char *)malloc(-1*args[0].idx*sizeof(T0));
  }
  if(arg1.idx < -1) {
    p_a[1] = (char *)malloc(-1*args[1].idx*sizeof(T1));
  }
  if(arg2.idx < -1) {
    p_a[2] = (char *)malloc(-1*args[2].idx*sizeof(T2));
  }
  if(arg3.idx < -1) {
    p_a[3] = (char *)malloc(-1*args[3].idx*sizeof(T3));
  }
  if(arg4.idx < -1) {
    p_a[4] = (char *)malloc(-1*args[4].idx*sizeof(T4));
  }
  if(arg5.idx < -1) {
    p_a[5] = (char *)malloc(-1*args[5].idx*sizeof(T5));
  }

  //allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i<6;i++)
    if(args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size )
    {
      blank_args_size = args[i].size;
      blank_args = (char *)malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags>0) op_args_check(set,6,args,&ninds,name);

  if (OP_diags>2) {
  if (ninds==0)
    printf(" kernel routine w/o indirection:  %s\n",name);
  else
    printf(" kernel routine with indirection: %s\n",name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 6, args);

  // loop over set elements
  int halo = 0;

  for (int n=0; n<n_upper; n++) {
    if (n==set->core_size) op_mpi_wait_all(6,args);
    if (n==set->size) halo = 1;
    if (args[0].idx < -1) op_arg_copy_in(n,args[0], (char **)p_a[0]);
    else op_arg_set(n,args[0], &p_a[0],halo);
    if (args[1].idx < -1) op_arg_copy_in(n,args[1], (char **)p_a[1]);
    else op_arg_set(n,args[1], &p_a[1],halo);
    if (args[2].idx < -1) op_arg_copy_in(n,args[2], (char **)p_a[2]);
    else op_arg_set(n,args[2], &p_a[2],halo);
    if (args[3].idx < -1) op_arg_copy_in(n,args[3], (char **)p_a[3]);
    else op_arg_set(n,args[3], &p_a[3],halo);
    if (args[4].idx < -1) op_arg_copy_in(n,args[4], (char **)p_a[4]);
    else op_arg_set(n,args[4], &p_a[4],halo);
    if (args[5].idx < -1) op_arg_copy_in(n,args[5], (char **)p_a[5]);
    else op_arg_set(n,args[5], &p_a[5],halo);

    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
          (T4 *)p_a[4], (T5 *)p_a[5]);
  }
  if ( n_upper == set->core_size || n_upper == 0 )
    op_mpi_wait_all (6,args);

  //set dirty bit on datasets touched
  op_mpi_set_dirtybit(6, args);

  //global reduction for MPI execution, if needed
  //p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0,(T0 *)p_a[0]);
  op_mpi_reduce(&arg1,(T1 *)p_a[1]);
  op_mpi_reduce(&arg2,(T2 *)p_a[2]);
  op_mpi_reduce(&arg3,(T3 *)p_a[3]);
  op_mpi_reduce(&arg4,(T4 *)p_a[4]);
  op_mpi_reduce(&arg5,(T5 *)p_a[5]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 6, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if(arg0.idx < -1) {
    free(p_a[0]);
  }
  if(arg1.idx < -1) {
    free(p_a[1]);
  }
  if(arg2.idx < -1) {
    free(p_a[2]);
  }
  if(arg3.idx < -1) {
    free(p_a[3]);
  }
  if(arg4.idx < -1) {
    free(p_a[4]);
  }
  if(arg5.idx < -1) {
    free(p_a[5]);
  }
}
//
//op_par_loop routine for 7 arguments
//
template <class T0,class T1,class T2,class T3,
          class T4,class T5,class T6>
void op_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                                T4*, T5*, T6*),
    char const * name, op_set set,
    op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3,
    op_arg arg4, op_arg arg5, op_arg arg6){

  char *p_a[7] = {0,0,0,0,0,0,0};
  op_arg args[7] = {arg0, arg1, arg2, arg3,
                    arg4, arg5, arg6};
  if(arg0.idx < -1) {
    p_a[0] = (char *)malloc(-1*args[0].idx*sizeof(T0));
  }
  if(arg1.idx < -1) {
    p_a[1] = (char *)malloc(-1*args[1].idx*sizeof(T1));
  }
  if(arg2.idx < -1) {
    p_a[2] = (char *)malloc(-1*args[2].idx*sizeof(T2));
  }
  if(arg3.idx < -1) {
    p_a[3] = (char *)malloc(-1*args[3].idx*sizeof(T3));
  }
  if(arg4.idx < -1) {
    p_a[4] = (char *)malloc(-1*args[4].idx*sizeof(T4));
  }
  if(arg5.idx < -1) {
    p_a[5] = (char *)malloc(-1*args[5].idx*sizeof(T5));
  }
  if(arg6.idx < -1) {
    p_a[6] = (char *)malloc(-1*args[6].idx*sizeof(T6));
  }

  //allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i<7;i++)
    if(args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size )
    {
      blank_args_size = args[i].size;
      blank_args = (char *)malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags>0) op_args_check(set,7,args,&ninds,name);

  if (OP_diags>2) {
  if (ninds==0)
    printf(" kernel routine w/o indirection:  %s\n",name);
  else
    printf(" kernel routine with indirection: %s\n",name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 7, args);

  // loop over set elements
  int halo = 0;

  for (int n=0; n<n_upper; n++) {
    if (n==set->core_size) op_mpi_wait_all(7,args);
    if (n==set->size) halo = 1;
    if (args[0].idx < -1) op_arg_copy_in(n,args[0], (char **)p_a[0]);
    else op_arg_set(n,args[0], &p_a[0],halo);
    if (args[1].idx < -1) op_arg_copy_in(n,args[1], (char **)p_a[1]);
    else op_arg_set(n,args[1], &p_a[1],halo);
    if (args[2].idx < -1) op_arg_copy_in(n,args[2], (char **)p_a[2]);
    else op_arg_set(n,args[2], &p_a[2],halo);
    if (args[3].idx < -1) op_arg_copy_in(n,args[3], (char **)p_a[3]);
    else op_arg_set(n,args[3], &p_a[3],halo);
    if (args[4].idx < -1) op_arg_copy_in(n,args[4], (char **)p_a[4]);
    else op_arg_set(n,args[4], &p_a[4],halo);
    if (args[5].idx < -1) op_arg_copy_in(n,args[5], (char **)p_a[5]);
    else op_arg_set(n,args[5], &p_a[5],halo);
    if (args[6].idx < -1) op_arg_copy_in(n,args[6], (char **)p_a[6]);
    else op_arg_set(n,args[6], &p_a[6],halo);

    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
          (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6]);
  }
  if ( n_upper == set->core_size || n_upper == 0 )
    op_mpi_wait_all (7,args);

  //set dirty bit on datasets touched
  op_mpi_set_dirtybit(7, args);

  //global reduction for MPI execution, if needed
  //p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0,(T0 *)p_a[0]);
  op_mpi_reduce(&arg1,(T1 *)p_a[1]);
  op_mpi_reduce(&arg2,(T2 *)p_a[2]);
  op_mpi_reduce(&arg3,(T3 *)p_a[3]);
  op_mpi_reduce(&arg4,(T4 *)p_a[4]);
  op_mpi_reduce(&arg5,(T5 *)p_a[5]);
  op_mpi_reduce(&arg6,(T6 *)p_a[6]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 7, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if(arg0.idx < -1) {
    free(p_a[0]);
  }
  if(arg1.idx < -1) {
    free(p_a[1]);
  }
  if(arg2.idx < -1) {
    free(p_a[2]);
  }
  if(arg3.idx < -1) {
    free(p_a[3]);
  }
  if(arg4.idx < -1) {
    free(p_a[4]);
  }
  if(arg5.idx < -1) {
    free(p_a[5]);
  }
  if(arg6.idx < -1) {
    free(p_a[6]);
  }
}
//
//op_par_loop routine for 8 arguments
//
template <class T0,class T1,class T2,class T3,
          class T4,class T5,class T6,class T7>
void op_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                                T4*, T5*, T6*, T7*),
    char const * name, op_set set,
    op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3,
    op_arg arg4, op_arg arg5, op_arg arg6, op_arg arg7){

  char *p_a[8] = {0,0,0,0,0,0,0,0};
  op_arg args[8] = {arg0, arg1, arg2, arg3,
                    arg4, arg5, arg6, arg7};
  if(arg0.idx < -1) {
    p_a[0] = (char *)malloc(-1*args[0].idx*sizeof(T0));
  }
  if(arg1.idx < -1) {
    p_a[1] = (char *)malloc(-1*args[1].idx*sizeof(T1));
  }
  if(arg2.idx < -1) {
    p_a[2] = (char *)malloc(-1*args[2].idx*sizeof(T2));
  }
  if(arg3.idx < -1) {
    p_a[3] = (char *)malloc(-1*args[3].idx*sizeof(T3));
  }
  if(arg4.idx < -1) {
    p_a[4] = (char *)malloc(-1*args[4].idx*sizeof(T4));
  }
  if(arg5.idx < -1) {
    p_a[5] = (char *)malloc(-1*args[5].idx*sizeof(T5));
  }
  if(arg6.idx < -1) {
    p_a[6] = (char *)malloc(-1*args[6].idx*sizeof(T6));
  }
  if(arg7.idx < -1) {
    p_a[7] = (char *)malloc(-1*args[7].idx*sizeof(T7));
  }

  //allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i<8;i++)
    if(args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size )
    {
      blank_args_size = args[i].size;
      blank_args = (char *)malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags>0) op_args_check(set,8,args,&ninds,name);

  if (OP_diags>2) {
  if (ninds==0)
    printf(" kernel routine w/o indirection:  %s\n",name);
  else
    printf(" kernel routine with indirection: %s\n",name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 8, args);

  // loop over set elements
  int halo = 0;

  for (int n=0; n<n_upper; n++) {
    if (n==set->core_size) op_mpi_wait_all(8,args);
    if (n==set->size) halo = 1;
    if (args[0].idx < -1) op_arg_copy_in(n,args[0], (char **)p_a[0]);
    else op_arg_set(n,args[0], &p_a[0],halo);
    if (args[1].idx < -1) op_arg_copy_in(n,args[1], (char **)p_a[1]);
    else op_arg_set(n,args[1], &p_a[1],halo);
    if (args[2].idx < -1) op_arg_copy_in(n,args[2], (char **)p_a[2]);
    else op_arg_set(n,args[2], &p_a[2],halo);
    if (args[3].idx < -1) op_arg_copy_in(n,args[3], (char **)p_a[3]);
    else op_arg_set(n,args[3], &p_a[3],halo);
    if (args[4].idx < -1) op_arg_copy_in(n,args[4], (char **)p_a[4]);
    else op_arg_set(n,args[4], &p_a[4],halo);
    if (args[5].idx < -1) op_arg_copy_in(n,args[5], (char **)p_a[5]);
    else op_arg_set(n,args[5], &p_a[5],halo);
    if (args[6].idx < -1) op_arg_copy_in(n,args[6], (char **)p_a[6]);
    else op_arg_set(n,args[6], &p_a[6],halo);
    if (args[7].idx < -1) op_arg_copy_in(n,args[7], (char **)p_a[7]);
    else op_arg_set(n,args[7], &p_a[7],halo);

    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
          (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7]);
  }
  if ( n_upper == set->core_size || n_upper == 0 )
    op_mpi_wait_all (8,args);

  //set dirty bit on datasets touched
  op_mpi_set_dirtybit(8, args);

  //global reduction for MPI execution, if needed
  //p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0,(T0 *)p_a[0]);
  op_mpi_reduce(&arg1,(T1 *)p_a[1]);
  op_mpi_reduce(&arg2,(T2 *)p_a[2]);
  op_mpi_reduce(&arg3,(T3 *)p_a[3]);
  op_mpi_reduce(&arg4,(T4 *)p_a[4]);
  op_mpi_reduce(&arg5,(T5 *)p_a[5]);
  op_mpi_reduce(&arg6,(T6 *)p_a[6]);
  op_mpi_reduce(&arg7,(T7 *)p_a[7]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 8, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if(arg0.idx < -1) {
    free(p_a[0]);
  }
  if(arg1.idx < -1) {
    free(p_a[1]);
  }
  if(arg2.idx < -1) {
    free(p_a[2]);
  }
  if(arg3.idx < -1) {
    free(p_a[3]);
  }
  if(arg4.idx < -1) {
    free(p_a[4]);
  }
  if(arg5.idx < -1) {
    free(p_a[5]);
  }
  if(arg6.idx < -1) {
    free(p_a[6]);
  }
  if(arg7.idx < -1) {
    free(p_a[7]);
  }
}
//
//op_par_loop routine for 9 arguments
//
template <class T0,class T1,class T2,class T3,
          class T4,class T5,class T6,class T7,
          class T8>
void op_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                                T4*, T5*, T6*, T7*,
                                T8*),
    char const * name, op_set set,
    op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3,
    op_arg arg4, op_arg arg5, op_arg arg6, op_arg arg7,
    op_arg arg8){

  char *p_a[9] = {0,0,0,0,0,0,0,0,0};
  op_arg args[9] = {arg0, arg1, arg2, arg3,
                    arg4, arg5, arg6, arg7,
                    arg8};
  if(arg0.idx < -1) {
    p_a[0] = (char *)malloc(-1*args[0].idx*sizeof(T0));
  }
  if(arg1.idx < -1) {
    p_a[1] = (char *)malloc(-1*args[1].idx*sizeof(T1));
  }
  if(arg2.idx < -1) {
    p_a[2] = (char *)malloc(-1*args[2].idx*sizeof(T2));
  }
  if(arg3.idx < -1) {
    p_a[3] = (char *)malloc(-1*args[3].idx*sizeof(T3));
  }
  if(arg4.idx < -1) {
    p_a[4] = (char *)malloc(-1*args[4].idx*sizeof(T4));
  }
  if(arg5.idx < -1) {
    p_a[5] = (char *)malloc(-1*args[5].idx*sizeof(T5));
  }
  if(arg6.idx < -1) {
    p_a[6] = (char *)malloc(-1*args[6].idx*sizeof(T6));
  }
  if(arg7.idx < -1) {
    p_a[7] = (char *)malloc(-1*args[7].idx*sizeof(T7));
  }
  if(arg8.idx < -1) {
    p_a[8] = (char *)malloc(-1*args[8].idx*sizeof(T8));
  }

  //allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i<9;i++)
    if(args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size )
    {
      blank_args_size = args[i].size;
      blank_args = (char *)malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags>0) op_args_check(set,9,args,&ninds,name);

  if (OP_diags>2) {
  if (ninds==0)
    printf(" kernel routine w/o indirection:  %s\n",name);
  else
    printf(" kernel routine with indirection: %s\n",name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 9, args);

  // loop over set elements
  int halo = 0;

  for (int n=0; n<n_upper; n++) {
    if (n==set->core_size) op_mpi_wait_all(9,args);
    if (n==set->size) halo = 1;
    if (args[0].idx < -1) op_arg_copy_in(n,args[0], (char **)p_a[0]);
    else op_arg_set(n,args[0], &p_a[0],halo);
    if (args[1].idx < -1) op_arg_copy_in(n,args[1], (char **)p_a[1]);
    else op_arg_set(n,args[1], &p_a[1],halo);
    if (args[2].idx < -1) op_arg_copy_in(n,args[2], (char **)p_a[2]);
    else op_arg_set(n,args[2], &p_a[2],halo);
    if (args[3].idx < -1) op_arg_copy_in(n,args[3], (char **)p_a[3]);
    else op_arg_set(n,args[3], &p_a[3],halo);
    if (args[4].idx < -1) op_arg_copy_in(n,args[4], (char **)p_a[4]);
    else op_arg_set(n,args[4], &p_a[4],halo);
    if (args[5].idx < -1) op_arg_copy_in(n,args[5], (char **)p_a[5]);
    else op_arg_set(n,args[5], &p_a[5],halo);
    if (args[6].idx < -1) op_arg_copy_in(n,args[6], (char **)p_a[6]);
    else op_arg_set(n,args[6], &p_a[6],halo);
    if (args[7].idx < -1) op_arg_copy_in(n,args[7], (char **)p_a[7]);
    else op_arg_set(n,args[7], &p_a[7],halo);
    if (args[8].idx < -1) op_arg_copy_in(n,args[8], (char **)p_a[8]);
    else op_arg_set(n,args[8], &p_a[8],halo);

    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
          (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
          (T8 *)p_a[8]);
  }
  if ( n_upper == set->core_size || n_upper == 0 )
    op_mpi_wait_all (9,args);

  //set dirty bit on datasets touched
  op_mpi_set_dirtybit(9, args);

  //global reduction for MPI execution, if needed
  //p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0,(T0 *)p_a[0]);
  op_mpi_reduce(&arg1,(T1 *)p_a[1]);
  op_mpi_reduce(&arg2,(T2 *)p_a[2]);
  op_mpi_reduce(&arg3,(T3 *)p_a[3]);
  op_mpi_reduce(&arg4,(T4 *)p_a[4]);
  op_mpi_reduce(&arg5,(T5 *)p_a[5]);
  op_mpi_reduce(&arg6,(T6 *)p_a[6]);
  op_mpi_reduce(&arg7,(T7 *)p_a[7]);
  op_mpi_reduce(&arg8,(T8 *)p_a[8]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 9, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if(arg0.idx < -1) {
    free(p_a[0]);
  }
  if(arg1.idx < -1) {
    free(p_a[1]);
  }
  if(arg2.idx < -1) {
    free(p_a[2]);
  }
  if(arg3.idx < -1) {
    free(p_a[3]);
  }
  if(arg4.idx < -1) {
    free(p_a[4]);
  }
  if(arg5.idx < -1) {
    free(p_a[5]);
  }
  if(arg6.idx < -1) {
    free(p_a[6]);
  }
  if(arg7.idx < -1) {
    free(p_a[7]);
  }
  if(arg8.idx < -1) {
    free(p_a[8]);
  }
}
//
//op_par_loop routine for 10 arguments
//
template <class T0,class T1,class T2,class T3,
          class T4,class T5,class T6,class T7,
          class T8,class T9>
void op_par_loop(void (*kernel)(T0*, T1*, T2*, T3*,
                                T4*, T5*, T6*, T7*,
                                T8*, T9*),
    char const * name, op_set set,
    op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3,
    op_arg arg4, op_arg arg5, op_arg arg6, op_arg arg7,
    op_arg arg8, op_arg arg9){

  char *p_a[10] = {0,0,0,0,0,0,0,0,0,0};
  op_arg args[10] = {arg0, arg1, arg2, arg3,
                    arg4, arg5, arg6, arg7,
                    arg8, arg9};
  if(arg0.idx < -1) {
    p_a[0] = (char *)malloc(-1*args[0].idx*sizeof(T0));
  }
  if(arg1.idx < -1) {
    p_a[1] = (char *)malloc(-1*args[1].idx*sizeof(T1));
  }
  if(arg2.idx < -1) {
    p_a[2] = (char *)malloc(-1*args[2].idx*sizeof(T2));
  }
  if(arg3.idx < -1) {
    p_a[3] = (char *)malloc(-1*args[3].idx*sizeof(T3));
  }
  if(arg4.idx < -1) {
    p_a[4] = (char *)malloc(-1*args[4].idx*sizeof(T4));
  }
  if(arg5.idx < -1) {
    p_a[5] = (char *)malloc(-1*args[5].idx*sizeof(T5));
  }
  if(arg6.idx < -1) {
    p_a[6] = (char *)malloc(-1*args[6].idx*sizeof(T6));
  }
  if(arg7.idx < -1) {
    p_a[7] = (char *)malloc(-1*args[7].idx*sizeof(T7));
  }
  if(arg8.idx < -1) {
    p_a[8] = (char *)malloc(-1*args[8].idx*sizeof(T8));
  }
  if(arg9.idx < -1) {
    p_a[9] = (char *)malloc(-1*args[9].idx*sizeof(T9));
  }

  //allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i<10;i++)
    if(args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size )
    {
      blank_args_size = args[i].size;
      blank_args = (char *)malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags>0) op_args_check(set,10,args,&ninds,name);

  if (OP_diags>2) {
  if (ninds==0)
    printf(" kernel routine w/o indirection:  %s\n",name);
  else
    printf(" kernel routine with indirection: %s\n",name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 10, args);

  // loop over set elements
  int halo = 0;

  for (int n=0; n<n_upper; n++) {
    if (n==set->core_size) op_mpi_wait_all(10,args);
    if (n==set->size) halo = 1;
    if (args[0].idx < -1) op_arg_copy_in(n,args[0], (char **)p_a[0]);
    else op_arg_set(n,args[0], &p_a[0],halo);
    if (args[1].idx < -1) op_arg_copy_in(n,args[1], (char **)p_a[1]);
    else op_arg_set(n,args[1], &p_a[1],halo);
    if (args[2].idx < -1) op_arg_copy_in(n,args[2], (char **)p_a[2]);
    else op_arg_set(n,args[2], &p_a[2],halo);
    if (args[3].idx < -1) op_arg_copy_in(n,args[3], (char **)p_a[3]);
    else op_arg_set(n,args[3], &p_a[3],halo);
    if (args[4].idx < -1) op_arg_copy_in(n,args[4], (char **)p_a[4]);
    else op_arg_set(n,args[4], &p_a[4],halo);
    if (args[5].idx < -1) op_arg_copy_in(n,args[5], (char **)p_a[5]);
    else op_arg_set(n,args[5], &p_a[5],halo);
    if (args[6].idx < -1) op_arg_copy_in(n,args[6], (char **)p_a[6]);
    else op_arg_set(n,args[6], &p_a[6],halo);
    if (args[7].idx < -1) op_arg_copy_in(n,args[7], (char **)p_a[7]);
    else op_arg_set(n,args[7], &p_a[7],halo);
    if (args[8].idx < -1) op_arg_copy_in(n,args[8], (char **)p_a[8]);
    else op_arg_set(n,args[8], &p_a[8],halo);
    if (args[9].idx < -1) op_arg_copy_in(n,args[9], (char **)p_a[9]);
    else op_arg_set(n,args[9], &p_a[9],halo);

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
          (T4 *)p_a[4], (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7],
          (T8 *)p_a[8], (T9 *)p_a[9]);
  }
  if ( n_upper == set->core_size || n_upper == 0 )
    op_mpi_wait_all (10,args);

  //set dirty bit on datasets touched
  op_mpi_set_dirtybit(10, args);

  //global reduction for MPI execution, if needed
  //p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0,(T0 *)p_a[0]);
  op_mpi_reduce(&arg1,(T1 *)p_a[1]);
  op_mpi_reduce(&arg2,(T2 *)p_a[2]);
  op_mpi_reduce(&arg3,(T3 *)p_a[3]);
  op_mpi_reduce(&arg4,(T4 *)p_a[4]);
  op_mpi_reduce(&arg5,(T5 *)p_a[5]);
  op_mpi_reduce(&arg6,(T6 *)p_a[6]);
  op_mpi_reduce(&arg7,(T7 *)p_a[7]);
  op_mpi_reduce(&arg8,(T8 *)p_a[8]);
  op_mpi_reduce(&arg9,(T9 *)p_a[9]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 10, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if(arg0.idx < -1) {
    free(p_a[0]);
  }
  if(arg1.idx < -1) {
    free(p_a[1]);
  }
  if(arg2.idx < -1) {
    free(p_a[2]);
  }
  if(arg3.idx < -1) {
    free(p_a[3]);
  }
  if(arg4.idx < -1) {
    free(p_a[4]);
  }
  if(arg5.idx < -1) {
    free(p_a[5]);
  }
  if(arg6.idx < -1) {
    free(p_a[6]);
  }
  if(arg7.idx < -1) {
    free(p_a[7]);
  }
  if(arg8.idx < -1) {
    free(p_a[8]);
  }
  if(arg9.idx < -1) {
    free(p_a[9]);
  }
}
