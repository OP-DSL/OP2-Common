
//
// header for sequential and MPI+sequentional execution
//

#include "op_lib_cpp.h"

static int op2_stride = 1;
#define OP2_STRIDE(arr, idx) arr[idx]

// scratch space to use for double counting in indirect reduction
static int blank_args_size = 512;
static char *blank_args = (char *)op_malloc(blank_args_size);

inline void op_arg_set(int n, op_arg arg, char **p_arg, int halo) {
  *p_arg = arg.data;

  if (arg.argtype == OP_ARG_GBL) {
    if (halo && (arg.acc != OP_READ))
      *p_arg = blank_args;
  } else {
    if (arg.map == NULL || arg.opt == 0) // identity mapping
      *p_arg += arg.size * n;
    else // standard pointers
      *p_arg += arg.size * arg.map->map[arg.idx + n * arg.map->dim];
  }
}

inline void op_arg_copy_in(int n, op_arg arg, char **p_arg) {
  for (int i = 0; i < -1 * arg.idx; ++i)
    p_arg[i] = arg.data + arg.map->map[i + n * arg.map->dim] * arg.size;
}

inline void op_args_check(op_set set, int nargs, op_arg *args, int *ninds,
                          const char *name) {
  for (int n = 0; n < nargs; n++)
    op_arg_check(set, n, args[n], ninds, name);
}

//
// op_par_loop routine for 1 arguments
//
template <class T0>
void op_par_loop(void (*kernel)(T0 *), char const *name, op_set set,
                 op_arg arg0) {

  char *p_a[1] = {0};
  op_arg args[1] = {arg0};
  if (arg0.idx < -1) {
    p_a[0] = (char *)op_malloc(-1 * args[0].idx * sizeof(T0));
  }

  // allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i < 1; i++)
    if (args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size) {
      blank_args_size = args[i].size;
      blank_args = (char *)op_malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags > 0)
    op_args_check(set, 1, args, &ninds, name);

  if (OP_diags > 2) {
    if (ninds == 0)
      printf(" kernel routine w/o indirection:  %s\n", name);
    else
      printf(" kernel routine with indirection: %s\n", name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 1, args);

  // loop over set elements
  int halo = 0;

  for (int n = 0; n < n_upper; n++) {
    if (n == set->core_size)
      op_mpi_wait_all(1, args);
    if (n == set->size)
      halo = 1;
    if (args[0].idx < -1)
      op_arg_copy_in(n, args[0], (char **)p_a[0]);
    else
      op_arg_set(n, args[0], &p_a[0], halo);

    kernel((T0 *)p_a[0]);
  }
  if (n_upper == set->core_size || n_upper == 0)
    op_mpi_wait_all(1, args);

  // set dirty bit on datasets touched
  op_mpi_set_dirtybit(1, args);

  // global reduction for MPI execution, if needed
  // p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0, (T0 *)p_a[0]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 1, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if (arg0.idx < -1) {
    free(p_a[0]);
  }
}
//
// op_par_loop routine for 2 arguments
//
template <class T0, class T1>
void op_par_loop(void (*kernel)(T0 *, T1 *), char const *name, op_set set,
                 op_arg arg0, op_arg arg1) {

  char *p_a[2] = {0, 0};
  op_arg args[2] = {arg0, arg1};
  if (arg0.idx < -1) {
    p_a[0] = (char *)op_malloc(-1 * args[0].idx * sizeof(T0));
  }
  if (arg1.idx < -1) {
    p_a[1] = (char *)op_malloc(-1 * args[1].idx * sizeof(T1));
  }

  // allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i < 2; i++)
    if (args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size) {
      blank_args_size = args[i].size;
      blank_args = (char *)op_malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags > 0)
    op_args_check(set, 2, args, &ninds, name);

  if (OP_diags > 2) {
    if (ninds == 0)
      printf(" kernel routine w/o indirection:  %s\n", name);
    else
      printf(" kernel routine with indirection: %s\n", name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 2, args);

  // loop over set elements
  int halo = 0;

  for (int n = 0; n < n_upper; n++) {
    if (n == set->core_size)
      op_mpi_wait_all(2, args);
    if (n == set->size)
      halo = 1;
    if (args[0].idx < -1)
      op_arg_copy_in(n, args[0], (char **)p_a[0]);
    else
      op_arg_set(n, args[0], &p_a[0], halo);
    if (args[1].idx < -1)
      op_arg_copy_in(n, args[1], (char **)p_a[1]);
    else
      op_arg_set(n, args[1], &p_a[1], halo);

    kernel((T0 *)p_a[0], (T1 *)p_a[1]);
  }
  if (n_upper == set->core_size || n_upper == 0)
    op_mpi_wait_all(2, args);

  // set dirty bit on datasets touched
  op_mpi_set_dirtybit(2, args);

  // global reduction for MPI execution, if needed
  // p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0, (T0 *)p_a[0]);
  op_mpi_reduce(&arg1, (T1 *)p_a[1]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 2, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if (arg0.idx < -1) {
    free(p_a[0]);
  }
  if (arg1.idx < -1) {
    free(p_a[1]);
  }
}
//
// op_par_loop routine for 3 arguments
//
template <class T0, class T1, class T2>
void op_par_loop(void (*kernel)(T0 *, T1 *, T2 *), char const *name, op_set set,
                 op_arg arg0, op_arg arg1, op_arg arg2) {

  char *p_a[3] = {0, 0, 0};
  op_arg args[3] = {arg0, arg1, arg2};
  if (arg0.idx < -1) {
    p_a[0] = (char *)op_malloc(-1 * args[0].idx * sizeof(T0));
  }
  if (arg1.idx < -1) {
    p_a[1] = (char *)op_malloc(-1 * args[1].idx * sizeof(T1));
  }
  if (arg2.idx < -1) {
    p_a[2] = (char *)op_malloc(-1 * args[2].idx * sizeof(T2));
  }

  // allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i < 3; i++)
    if (args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size) {
      blank_args_size = args[i].size;
      blank_args = (char *)op_malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags > 0)
    op_args_check(set, 3, args, &ninds, name);

  if (OP_diags > 2) {
    if (ninds == 0)
      printf(" kernel routine w/o indirection:  %s\n", name);
    else
      printf(" kernel routine with indirection: %s\n", name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 3, args);

  // loop over set elements
  int halo = 0;

  for (int n = 0; n < n_upper; n++) {
    if (n == set->core_size)
      op_mpi_wait_all(3, args);
    if (n == set->size)
      halo = 1;
    if (args[0].idx < -1)
      op_arg_copy_in(n, args[0], (char **)p_a[0]);
    else
      op_arg_set(n, args[0], &p_a[0], halo);
    if (args[1].idx < -1)
      op_arg_copy_in(n, args[1], (char **)p_a[1]);
    else
      op_arg_set(n, args[1], &p_a[1], halo);
    if (args[2].idx < -1)
      op_arg_copy_in(n, args[2], (char **)p_a[2]);
    else
      op_arg_set(n, args[2], &p_a[2], halo);

    kernel((T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2]);
  }
  if (n_upper == set->core_size || n_upper == 0)
    op_mpi_wait_all(3, args);

  // set dirty bit on datasets touched
  op_mpi_set_dirtybit(3, args);

  // global reduction for MPI execution, if needed
  // p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0, (T0 *)p_a[0]);
  op_mpi_reduce(&arg1, (T1 *)p_a[1]);
  op_mpi_reduce(&arg2, (T2 *)p_a[2]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 3, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if (arg0.idx < -1) {
    free(p_a[0]);
  }
  if (arg1.idx < -1) {
    free(p_a[1]);
  }
  if (arg2.idx < -1) {
    free(p_a[2]);
  }
}
//
// op_par_loop routine for 4 arguments
//
template <class T0, class T1, class T2, class T3>
void op_par_loop(void (*kernel)(T0 *, T1 *, T2 *, T3 *), char const *name,
                 op_set set, op_arg arg0, op_arg arg1, op_arg arg2,
                 op_arg arg3) {

  char *p_a[4] = {0, 0, 0, 0};
  op_arg args[4] = {arg0, arg1, arg2, arg3};
  if (arg0.idx < -1) {
    p_a[0] = (char *)op_malloc(-1 * args[0].idx * sizeof(T0));
  }
  if (arg1.idx < -1) {
    p_a[1] = (char *)op_malloc(-1 * args[1].idx * sizeof(T1));
  }
  if (arg2.idx < -1) {
    p_a[2] = (char *)op_malloc(-1 * args[2].idx * sizeof(T2));
  }
  if (arg3.idx < -1) {
    p_a[3] = (char *)op_malloc(-1 * args[3].idx * sizeof(T3));
  }

  // allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i < 4; i++)
    if (args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size) {
      blank_args_size = args[i].size;
      blank_args = (char *)op_malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags > 0)
    op_args_check(set, 4, args, &ninds, name);

  if (OP_diags > 2) {
    if (ninds == 0)
      printf(" kernel routine w/o indirection:  %s\n", name);
    else
      printf(" kernel routine with indirection: %s\n", name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 4, args);

  // loop over set elements
  int halo = 0;

  for (int n = 0; n < n_upper; n++) {
    if (n == set->core_size)
      op_mpi_wait_all(4, args);
    if (n == set->size)
      halo = 1;
    if (args[0].idx < -1)
      op_arg_copy_in(n, args[0], (char **)p_a[0]);
    else
      op_arg_set(n, args[0], &p_a[0], halo);
    if (args[1].idx < -1)
      op_arg_copy_in(n, args[1], (char **)p_a[1]);
    else
      op_arg_set(n, args[1], &p_a[1], halo);
    if (args[2].idx < -1)
      op_arg_copy_in(n, args[2], (char **)p_a[2]);
    else
      op_arg_set(n, args[2], &p_a[2], halo);
    if (args[3].idx < -1)
      op_arg_copy_in(n, args[3], (char **)p_a[3]);
    else
      op_arg_set(n, args[3], &p_a[3], halo);

    kernel((T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3]);
  }
  if (n_upper == set->core_size || n_upper == 0)
    op_mpi_wait_all(4, args);

  // set dirty bit on datasets touched
  op_mpi_set_dirtybit(4, args);

  // global reduction for MPI execution, if needed
  // p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0, (T0 *)p_a[0]);
  op_mpi_reduce(&arg1, (T1 *)p_a[1]);
  op_mpi_reduce(&arg2, (T2 *)p_a[2]);
  op_mpi_reduce(&arg3, (T3 *)p_a[3]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 4, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if (arg0.idx < -1) {
    free(p_a[0]);
  }
  if (arg1.idx < -1) {
    free(p_a[1]);
  }
  if (arg2.idx < -1) {
    free(p_a[2]);
  }
  if (arg3.idx < -1) {
    free(p_a[3]);
  }
}
//
// op_par_loop routine for 5 arguments
//
template <class T0, class T1, class T2, class T3, class T4>
void op_par_loop(void (*kernel)(T0 *, T1 *, T2 *, T3 *, T4 *), char const *name,
                 op_set set, op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3,
                 op_arg arg4) {

  char *p_a[5] = {0, 0, 0, 0, 0};
  op_arg args[5] = {arg0, arg1, arg2, arg3, arg4};
  if (arg0.idx < -1) {
    p_a[0] = (char *)op_malloc(-1 * args[0].idx * sizeof(T0));
  }
  if (arg1.idx < -1) {
    p_a[1] = (char *)op_malloc(-1 * args[1].idx * sizeof(T1));
  }
  if (arg2.idx < -1) {
    p_a[2] = (char *)op_malloc(-1 * args[2].idx * sizeof(T2));
  }
  if (arg3.idx < -1) {
    p_a[3] = (char *)op_malloc(-1 * args[3].idx * sizeof(T3));
  }
  if (arg4.idx < -1) {
    p_a[4] = (char *)op_malloc(-1 * args[4].idx * sizeof(T4));
  }

  // allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i < 5; i++)
    if (args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size) {
      blank_args_size = args[i].size;
      blank_args = (char *)op_malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags > 0)
    op_args_check(set, 5, args, &ninds, name);

  if (OP_diags > 2) {
    if (ninds == 0)
      printf(" kernel routine w/o indirection:  %s\n", name);
    else
      printf(" kernel routine with indirection: %s\n", name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 5, args);

  // loop over set elements
  int halo = 0;

  for (int n = 0; n < n_upper; n++) {
    if (n == set->core_size)
      op_mpi_wait_all(5, args);
    if (n == set->size)
      halo = 1;
    if (args[0].idx < -1)
      op_arg_copy_in(n, args[0], (char **)p_a[0]);
    else
      op_arg_set(n, args[0], &p_a[0], halo);
    if (args[1].idx < -1)
      op_arg_copy_in(n, args[1], (char **)p_a[1]);
    else
      op_arg_set(n, args[1], &p_a[1], halo);
    if (args[2].idx < -1)
      op_arg_copy_in(n, args[2], (char **)p_a[2]);
    else
      op_arg_set(n, args[2], &p_a[2], halo);
    if (args[3].idx < -1)
      op_arg_copy_in(n, args[3], (char **)p_a[3]);
    else
      op_arg_set(n, args[3], &p_a[3], halo);
    if (args[4].idx < -1)
      op_arg_copy_in(n, args[4], (char **)p_a[4]);
    else
      op_arg_set(n, args[4], &p_a[4], halo);

    kernel((T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3],
           (T4 *)p_a[4]);
  }
  if (n_upper == set->core_size || n_upper == 0)
    op_mpi_wait_all(5, args);

  // set dirty bit on datasets touched
  op_mpi_set_dirtybit(5, args);

  // global reduction for MPI execution, if needed
  // p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0, (T0 *)p_a[0]);
  op_mpi_reduce(&arg1, (T1 *)p_a[1]);
  op_mpi_reduce(&arg2, (T2 *)p_a[2]);
  op_mpi_reduce(&arg3, (T3 *)p_a[3]);
  op_mpi_reduce(&arg4, (T4 *)p_a[4]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 5, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if (arg0.idx < -1) {
    free(p_a[0]);
  }
  if (arg1.idx < -1) {
    free(p_a[1]);
  }
  if (arg2.idx < -1) {
    free(p_a[2]);
  }
  if (arg3.idx < -1) {
    free(p_a[3]);
  }
  if (arg4.idx < -1) {
    free(p_a[4]);
  }
}
//
// op_par_loop routine for 6 arguments
//
template <class T0, class T1, class T2, class T3, class T4, class T5>
void op_par_loop(void (*kernel)(T0 *, T1 *, T2 *, T3 *, T4 *, T5 *),
                 char const *name, op_set set, op_arg arg0, op_arg arg1,
                 op_arg arg2, op_arg arg3, op_arg arg4, op_arg arg5) {

  char *p_a[6] = {0, 0, 0, 0, 0, 0};
  op_arg args[6] = {arg0, arg1, arg2, arg3, arg4, arg5};
  if (arg0.idx < -1) {
    p_a[0] = (char *)op_malloc(-1 * args[0].idx * sizeof(T0));
  }
  if (arg1.idx < -1) {
    p_a[1] = (char *)op_malloc(-1 * args[1].idx * sizeof(T1));
  }
  if (arg2.idx < -1) {
    p_a[2] = (char *)op_malloc(-1 * args[2].idx * sizeof(T2));
  }
  if (arg3.idx < -1) {
    p_a[3] = (char *)op_malloc(-1 * args[3].idx * sizeof(T3));
  }
  if (arg4.idx < -1) {
    p_a[4] = (char *)op_malloc(-1 * args[4].idx * sizeof(T4));
  }
  if (arg5.idx < -1) {
    p_a[5] = (char *)op_malloc(-1 * args[5].idx * sizeof(T5));
  }

  // allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i < 6; i++)
    if (args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size) {
      blank_args_size = args[i].size;
      blank_args = (char *)op_malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags > 0)
    op_args_check(set, 6, args, &ninds, name);

  if (OP_diags > 2) {
    if (ninds == 0)
      printf(" kernel routine w/o indirection:  %s\n", name);
    else
      printf(" kernel routine with indirection: %s\n", name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 6, args);

  // loop over set elements
  int halo = 0;

  for (int n = 0; n < n_upper; n++) {
    if (n == set->core_size)
      op_mpi_wait_all(6, args);
    if (n == set->size)
      halo = 1;
    if (args[0].idx < -1)
      op_arg_copy_in(n, args[0], (char **)p_a[0]);
    else
      op_arg_set(n, args[0], &p_a[0], halo);
    if (args[1].idx < -1)
      op_arg_copy_in(n, args[1], (char **)p_a[1]);
    else
      op_arg_set(n, args[1], &p_a[1], halo);
    if (args[2].idx < -1)
      op_arg_copy_in(n, args[2], (char **)p_a[2]);
    else
      op_arg_set(n, args[2], &p_a[2], halo);
    if (args[3].idx < -1)
      op_arg_copy_in(n, args[3], (char **)p_a[3]);
    else
      op_arg_set(n, args[3], &p_a[3], halo);
    if (args[4].idx < -1)
      op_arg_copy_in(n, args[4], (char **)p_a[4]);
    else
      op_arg_set(n, args[4], &p_a[4], halo);
    if (args[5].idx < -1)
      op_arg_copy_in(n, args[5], (char **)p_a[5]);
    else
      op_arg_set(n, args[5], &p_a[5], halo);

    kernel((T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3], (T4 *)p_a[4],
           (T5 *)p_a[5]);
  }
  if (n_upper == set->core_size || n_upper == 0)
    op_mpi_wait_all(6, args);

  // set dirty bit on datasets touched
  op_mpi_set_dirtybit(6, args);

  // global reduction for MPI execution, if needed
  // p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0, (T0 *)p_a[0]);
  op_mpi_reduce(&arg1, (T1 *)p_a[1]);
  op_mpi_reduce(&arg2, (T2 *)p_a[2]);
  op_mpi_reduce(&arg3, (T3 *)p_a[3]);
  op_mpi_reduce(&arg4, (T4 *)p_a[4]);
  op_mpi_reduce(&arg5, (T5 *)p_a[5]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 6, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if (arg0.idx < -1) {
    free(p_a[0]);
  }
  if (arg1.idx < -1) {
    free(p_a[1]);
  }
  if (arg2.idx < -1) {
    free(p_a[2]);
  }
  if (arg3.idx < -1) {
    free(p_a[3]);
  }
  if (arg4.idx < -1) {
    free(p_a[4]);
  }
  if (arg5.idx < -1) {
    free(p_a[5]);
  }
}
//
// op_par_loop routine for 7 arguments
//
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6>
void op_par_loop(void (*kernel)(T0 *, T1 *, T2 *, T3 *, T4 *, T5 *, T6 *),
                 char const *name, op_set set, op_arg arg0, op_arg arg1,
                 op_arg arg2, op_arg arg3, op_arg arg4, op_arg arg5,
                 op_arg arg6) {

  char *p_a[7] = {0, 0, 0, 0, 0, 0, 0};
  op_arg args[7] = {arg0, arg1, arg2, arg3, arg4, arg5, arg6};
  if (arg0.idx < -1) {
    p_a[0] = (char *)op_malloc(-1 * args[0].idx * sizeof(T0));
  }
  if (arg1.idx < -1) {
    p_a[1] = (char *)op_malloc(-1 * args[1].idx * sizeof(T1));
  }
  if (arg2.idx < -1) {
    p_a[2] = (char *)op_malloc(-1 * args[2].idx * sizeof(T2));
  }
  if (arg3.idx < -1) {
    p_a[3] = (char *)op_malloc(-1 * args[3].idx * sizeof(T3));
  }
  if (arg4.idx < -1) {
    p_a[4] = (char *)op_malloc(-1 * args[4].idx * sizeof(T4));
  }
  if (arg5.idx < -1) {
    p_a[5] = (char *)op_malloc(-1 * args[5].idx * sizeof(T5));
  }
  if (arg6.idx < -1) {
    p_a[6] = (char *)op_malloc(-1 * args[6].idx * sizeof(T6));
  }

  // allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i < 7; i++)
    if (args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size) {
      blank_args_size = args[i].size;
      blank_args = (char *)op_malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags > 0)
    op_args_check(set, 7, args, &ninds, name);

  if (OP_diags > 2) {
    if (ninds == 0)
      printf(" kernel routine w/o indirection:  %s\n", name);
    else
      printf(" kernel routine with indirection: %s\n", name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 7, args);

  // loop over set elements
  int halo = 0;

  for (int n = 0; n < n_upper; n++) {
    if (n == set->core_size)
      op_mpi_wait_all(7, args);
    if (n == set->size)
      halo = 1;
    if (args[0].idx < -1)
      op_arg_copy_in(n, args[0], (char **)p_a[0]);
    else
      op_arg_set(n, args[0], &p_a[0], halo);
    if (args[1].idx < -1)
      op_arg_copy_in(n, args[1], (char **)p_a[1]);
    else
      op_arg_set(n, args[1], &p_a[1], halo);
    if (args[2].idx < -1)
      op_arg_copy_in(n, args[2], (char **)p_a[2]);
    else
      op_arg_set(n, args[2], &p_a[2], halo);
    if (args[3].idx < -1)
      op_arg_copy_in(n, args[3], (char **)p_a[3]);
    else
      op_arg_set(n, args[3], &p_a[3], halo);
    if (args[4].idx < -1)
      op_arg_copy_in(n, args[4], (char **)p_a[4]);
    else
      op_arg_set(n, args[4], &p_a[4], halo);
    if (args[5].idx < -1)
      op_arg_copy_in(n, args[5], (char **)p_a[5]);
    else
      op_arg_set(n, args[5], &p_a[5], halo);
    if (args[6].idx < -1)
      op_arg_copy_in(n, args[6], (char **)p_a[6]);
    else
      op_arg_set(n, args[6], &p_a[6], halo);

    kernel((T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3], (T4 *)p_a[4],
           (T5 *)p_a[5], (T6 *)p_a[6]);
  }
  if (n_upper == set->core_size || n_upper == 0)
    op_mpi_wait_all(7, args);

  // set dirty bit on datasets touched
  op_mpi_set_dirtybit(7, args);

  // global reduction for MPI execution, if needed
  // p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0, (T0 *)p_a[0]);
  op_mpi_reduce(&arg1, (T1 *)p_a[1]);
  op_mpi_reduce(&arg2, (T2 *)p_a[2]);
  op_mpi_reduce(&arg3, (T3 *)p_a[3]);
  op_mpi_reduce(&arg4, (T4 *)p_a[4]);
  op_mpi_reduce(&arg5, (T5 *)p_a[5]);
  op_mpi_reduce(&arg6, (T6 *)p_a[6]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 7, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if (arg0.idx < -1) {
    free(p_a[0]);
  }
  if (arg1.idx < -1) {
    free(p_a[1]);
  }
  if (arg2.idx < -1) {
    free(p_a[2]);
  }
  if (arg3.idx < -1) {
    free(p_a[3]);
  }
  if (arg4.idx < -1) {
    free(p_a[4]);
  }
  if (arg5.idx < -1) {
    free(p_a[5]);
  }
  if (arg6.idx < -1) {
    free(p_a[6]);
  }
}
//
// op_par_loop routine for 8 arguments
//
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6,
          class T7>
void op_par_loop(void (*kernel)(T0 *, T1 *, T2 *, T3 *, T4 *, T5 *, T6 *, T7 *),
                 char const *name, op_set set, op_arg arg0, op_arg arg1,
                 op_arg arg2, op_arg arg3, op_arg arg4, op_arg arg5,
                 op_arg arg6, op_arg arg7) {

  char *p_a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  op_arg args[8] = {arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7};
  if (arg0.idx < -1) {
    p_a[0] = (char *)op_malloc(-1 * args[0].idx * sizeof(T0));
  }
  if (arg1.idx < -1) {
    p_a[1] = (char *)op_malloc(-1 * args[1].idx * sizeof(T1));
  }
  if (arg2.idx < -1) {
    p_a[2] = (char *)op_malloc(-1 * args[2].idx * sizeof(T2));
  }
  if (arg3.idx < -1) {
    p_a[3] = (char *)op_malloc(-1 * args[3].idx * sizeof(T3));
  }
  if (arg4.idx < -1) {
    p_a[4] = (char *)op_malloc(-1 * args[4].idx * sizeof(T4));
  }
  if (arg5.idx < -1) {
    p_a[5] = (char *)op_malloc(-1 * args[5].idx * sizeof(T5));
  }
  if (arg6.idx < -1) {
    p_a[6] = (char *)op_malloc(-1 * args[6].idx * sizeof(T6));
  }
  if (arg7.idx < -1) {
    p_a[7] = (char *)op_malloc(-1 * args[7].idx * sizeof(T7));
  }

  // allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i < 8; i++)
    if (args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size) {
      blank_args_size = args[i].size;
      blank_args = (char *)op_malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags > 0)
    op_args_check(set, 8, args, &ninds, name);

  if (OP_diags > 2) {
    if (ninds == 0)
      printf(" kernel routine w/o indirection:  %s\n", name);
    else
      printf(" kernel routine with indirection: %s\n", name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 8, args);

  // loop over set elements
  int halo = 0;

  for (int n = 0; n < n_upper; n++) {
    if (n == set->core_size)
      op_mpi_wait_all(8, args);
    if (n == set->size)
      halo = 1;
    if (args[0].idx < -1)
      op_arg_copy_in(n, args[0], (char **)p_a[0]);
    else
      op_arg_set(n, args[0], &p_a[0], halo);
    if (args[1].idx < -1)
      op_arg_copy_in(n, args[1], (char **)p_a[1]);
    else
      op_arg_set(n, args[1], &p_a[1], halo);
    if (args[2].idx < -1)
      op_arg_copy_in(n, args[2], (char **)p_a[2]);
    else
      op_arg_set(n, args[2], &p_a[2], halo);
    if (args[3].idx < -1)
      op_arg_copy_in(n, args[3], (char **)p_a[3]);
    else
      op_arg_set(n, args[3], &p_a[3], halo);
    if (args[4].idx < -1)
      op_arg_copy_in(n, args[4], (char **)p_a[4]);
    else
      op_arg_set(n, args[4], &p_a[4], halo);
    if (args[5].idx < -1)
      op_arg_copy_in(n, args[5], (char **)p_a[5]);
    else
      op_arg_set(n, args[5], &p_a[5], halo);
    if (args[6].idx < -1)
      op_arg_copy_in(n, args[6], (char **)p_a[6]);
    else
      op_arg_set(n, args[6], &p_a[6], halo);
    if (args[7].idx < -1)
      op_arg_copy_in(n, args[7], (char **)p_a[7]);
    else
      op_arg_set(n, args[7], &p_a[7], halo);

    kernel((T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3], (T4 *)p_a[4],
           (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7]);
  }
  if (n_upper == set->core_size || n_upper == 0)
    op_mpi_wait_all(8, args);

  // set dirty bit on datasets touched
  op_mpi_set_dirtybit(8, args);

  // global reduction for MPI execution, if needed
  // p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0, (T0 *)p_a[0]);
  op_mpi_reduce(&arg1, (T1 *)p_a[1]);
  op_mpi_reduce(&arg2, (T2 *)p_a[2]);
  op_mpi_reduce(&arg3, (T3 *)p_a[3]);
  op_mpi_reduce(&arg4, (T4 *)p_a[4]);
  op_mpi_reduce(&arg5, (T5 *)p_a[5]);
  op_mpi_reduce(&arg6, (T6 *)p_a[6]);
  op_mpi_reduce(&arg7, (T7 *)p_a[7]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 8, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if (arg0.idx < -1) {
    free(p_a[0]);
  }
  if (arg1.idx < -1) {
    free(p_a[1]);
  }
  if (arg2.idx < -1) {
    free(p_a[2]);
  }
  if (arg3.idx < -1) {
    free(p_a[3]);
  }
  if (arg4.idx < -1) {
    free(p_a[4]);
  }
  if (arg5.idx < -1) {
    free(p_a[5]);
  }
  if (arg6.idx < -1) {
    free(p_a[6]);
  }
  if (arg7.idx < -1) {
    free(p_a[7]);
  }
}
//
// op_par_loop routine for 9 arguments
//
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6,
          class T7, class T8>
void op_par_loop(void (*kernel)(T0 *, T1 *, T2 *, T3 *, T4 *, T5 *, T6 *, T7 *,
                                T8 *),
                 char const *name, op_set set, op_arg arg0, op_arg arg1,
                 op_arg arg2, op_arg arg3, op_arg arg4, op_arg arg5,
                 op_arg arg6, op_arg arg7, op_arg arg8) {

  char *p_a[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  op_arg args[9] = {arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8};
  if (arg0.idx < -1) {
    p_a[0] = (char *)op_malloc(-1 * args[0].idx * sizeof(T0));
  }
  if (arg1.idx < -1) {
    p_a[1] = (char *)op_malloc(-1 * args[1].idx * sizeof(T1));
  }
  if (arg2.idx < -1) {
    p_a[2] = (char *)op_malloc(-1 * args[2].idx * sizeof(T2));
  }
  if (arg3.idx < -1) {
    p_a[3] = (char *)op_malloc(-1 * args[3].idx * sizeof(T3));
  }
  if (arg4.idx < -1) {
    p_a[4] = (char *)op_malloc(-1 * args[4].idx * sizeof(T4));
  }
  if (arg5.idx < -1) {
    p_a[5] = (char *)op_malloc(-1 * args[5].idx * sizeof(T5));
  }
  if (arg6.idx < -1) {
    p_a[6] = (char *)op_malloc(-1 * args[6].idx * sizeof(T6));
  }
  if (arg7.idx < -1) {
    p_a[7] = (char *)op_malloc(-1 * args[7].idx * sizeof(T7));
  }
  if (arg8.idx < -1) {
    p_a[8] = (char *)op_malloc(-1 * args[8].idx * sizeof(T8));
  }

  // allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i < 9; i++)
    if (args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size) {
      blank_args_size = args[i].size;
      blank_args = (char *)op_malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags > 0)
    op_args_check(set, 9, args, &ninds, name);

  if (OP_diags > 2) {
    if (ninds == 0)
      printf(" kernel routine w/o indirection:  %s\n", name);
    else
      printf(" kernel routine with indirection: %s\n", name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 9, args);

  // loop over set elements
  int halo = 0;

  for (int n = 0; n < n_upper; n++) {
    if (n == set->core_size)
      op_mpi_wait_all(9, args);
    if (n == set->size)
      halo = 1;
    if (args[0].idx < -1)
      op_arg_copy_in(n, args[0], (char **)p_a[0]);
    else
      op_arg_set(n, args[0], &p_a[0], halo);
    if (args[1].idx < -1)
      op_arg_copy_in(n, args[1], (char **)p_a[1]);
    else
      op_arg_set(n, args[1], &p_a[1], halo);
    if (args[2].idx < -1)
      op_arg_copy_in(n, args[2], (char **)p_a[2]);
    else
      op_arg_set(n, args[2], &p_a[2], halo);
    if (args[3].idx < -1)
      op_arg_copy_in(n, args[3], (char **)p_a[3]);
    else
      op_arg_set(n, args[3], &p_a[3], halo);
    if (args[4].idx < -1)
      op_arg_copy_in(n, args[4], (char **)p_a[4]);
    else
      op_arg_set(n, args[4], &p_a[4], halo);
    if (args[5].idx < -1)
      op_arg_copy_in(n, args[5], (char **)p_a[5]);
    else
      op_arg_set(n, args[5], &p_a[5], halo);
    if (args[6].idx < -1)
      op_arg_copy_in(n, args[6], (char **)p_a[6]);
    else
      op_arg_set(n, args[6], &p_a[6], halo);
    if (args[7].idx < -1)
      op_arg_copy_in(n, args[7], (char **)p_a[7]);
    else
      op_arg_set(n, args[7], &p_a[7], halo);
    if (args[8].idx < -1)
      op_arg_copy_in(n, args[8], (char **)p_a[8]);
    else
      op_arg_set(n, args[8], &p_a[8], halo);

    kernel((T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3], (T4 *)p_a[4],
           (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7], (T8 *)p_a[8]);
  }
  if (n_upper == set->core_size || n_upper == 0)
    op_mpi_wait_all(9, args);

  // set dirty bit on datasets touched
  op_mpi_set_dirtybit(9, args);

  // global reduction for MPI execution, if needed
  // p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0, (T0 *)p_a[0]);
  op_mpi_reduce(&arg1, (T1 *)p_a[1]);
  op_mpi_reduce(&arg2, (T2 *)p_a[2]);
  op_mpi_reduce(&arg3, (T3 *)p_a[3]);
  op_mpi_reduce(&arg4, (T4 *)p_a[4]);
  op_mpi_reduce(&arg5, (T5 *)p_a[5]);
  op_mpi_reduce(&arg6, (T6 *)p_a[6]);
  op_mpi_reduce(&arg7, (T7 *)p_a[7]);
  op_mpi_reduce(&arg8, (T8 *)p_a[8]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 9, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if (arg0.idx < -1) {
    free(p_a[0]);
  }
  if (arg1.idx < -1) {
    free(p_a[1]);
  }
  if (arg2.idx < -1) {
    free(p_a[2]);
  }
  if (arg3.idx < -1) {
    free(p_a[3]);
  }
  if (arg4.idx < -1) {
    free(p_a[4]);
  }
  if (arg5.idx < -1) {
    free(p_a[5]);
  }
  if (arg6.idx < -1) {
    free(p_a[6]);
  }
  if (arg7.idx < -1) {
    free(p_a[7]);
  }
  if (arg8.idx < -1) {
    free(p_a[8]);
  }
}
//
// op_par_loop routine for 10 arguments
//
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6,
          class T7, class T8, class T9>
void op_par_loop(void (*kernel)(T0 *, T1 *, T2 *, T3 *, T4 *, T5 *, T6 *, T7 *,
                                T8 *, T9 *),
                 char const *name, op_set set, op_arg arg0, op_arg arg1,
                 op_arg arg2, op_arg arg3, op_arg arg4, op_arg arg5,
                 op_arg arg6, op_arg arg7, op_arg arg8, op_arg arg9) {

  char *p_a[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  op_arg args[10] = {arg0, arg1, arg2, arg3, arg4,
                     arg5, arg6, arg7, arg8, arg9};
  if (arg0.idx < -1) {
    p_a[0] = (char *)op_malloc(-1 * args[0].idx * sizeof(T0));
  }
  if (arg1.idx < -1) {
    p_a[1] = (char *)op_malloc(-1 * args[1].idx * sizeof(T1));
  }
  if (arg2.idx < -1) {
    p_a[2] = (char *)op_malloc(-1 * args[2].idx * sizeof(T2));
  }
  if (arg3.idx < -1) {
    p_a[3] = (char *)op_malloc(-1 * args[3].idx * sizeof(T3));
  }
  if (arg4.idx < -1) {
    p_a[4] = (char *)op_malloc(-1 * args[4].idx * sizeof(T4));
  }
  if (arg5.idx < -1) {
    p_a[5] = (char *)op_malloc(-1 * args[5].idx * sizeof(T5));
  }
  if (arg6.idx < -1) {
    p_a[6] = (char *)op_malloc(-1 * args[6].idx * sizeof(T6));
  }
  if (arg7.idx < -1) {
    p_a[7] = (char *)op_malloc(-1 * args[7].idx * sizeof(T7));
  }
  if (arg8.idx < -1) {
    p_a[8] = (char *)op_malloc(-1 * args[8].idx * sizeof(T8));
  }
  if (arg9.idx < -1) {
    p_a[9] = (char *)op_malloc(-1 * args[9].idx * sizeof(T9));
  }

  // allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i < 10; i++)
    if (args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size) {
      blank_args_size = args[i].size;
      blank_args = (char *)op_malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags > 0)
    op_args_check(set, 10, args, &ninds, name);

  if (OP_diags > 2) {
    if (ninds == 0)
      printf(" kernel routine w/o indirection:  %s\n", name);
    else
      printf(" kernel routine with indirection: %s\n", name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 10, args);

  // loop over set elements
  int halo = 0;

  for (int n = 0; n < n_upper; n++) {
    if (n == set->core_size)
      op_mpi_wait_all(10, args);
    if (n == set->size)
      halo = 1;
    if (args[0].idx < -1)
      op_arg_copy_in(n, args[0], (char **)p_a[0]);
    else
      op_arg_set(n, args[0], &p_a[0], halo);
    if (args[1].idx < -1)
      op_arg_copy_in(n, args[1], (char **)p_a[1]);
    else
      op_arg_set(n, args[1], &p_a[1], halo);
    if (args[2].idx < -1)
      op_arg_copy_in(n, args[2], (char **)p_a[2]);
    else
      op_arg_set(n, args[2], &p_a[2], halo);
    if (args[3].idx < -1)
      op_arg_copy_in(n, args[3], (char **)p_a[3]);
    else
      op_arg_set(n, args[3], &p_a[3], halo);
    if (args[4].idx < -1)
      op_arg_copy_in(n, args[4], (char **)p_a[4]);
    else
      op_arg_set(n, args[4], &p_a[4], halo);
    if (args[5].idx < -1)
      op_arg_copy_in(n, args[5], (char **)p_a[5]);
    else
      op_arg_set(n, args[5], &p_a[5], halo);
    if (args[6].idx < -1)
      op_arg_copy_in(n, args[6], (char **)p_a[6]);
    else
      op_arg_set(n, args[6], &p_a[6], halo);
    if (args[7].idx < -1)
      op_arg_copy_in(n, args[7], (char **)p_a[7]);
    else
      op_arg_set(n, args[7], &p_a[7], halo);
    if (args[8].idx < -1)
      op_arg_copy_in(n, args[8], (char **)p_a[8]);
    else
      op_arg_set(n, args[8], &p_a[8], halo);
    if (args[9].idx < -1)
      op_arg_copy_in(n, args[9], (char **)p_a[9]);
    else
      op_arg_set(n, args[9], &p_a[9], halo);

    kernel((T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3], (T4 *)p_a[4],
           (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7], (T8 *)p_a[8],
           (T9 *)p_a[9]);
  }
  if (n_upper == set->core_size || n_upper == 0)
    op_mpi_wait_all(10, args);

  // set dirty bit on datasets touched
  op_mpi_set_dirtybit(10, args);

  // global reduction for MPI execution, if needed
  // p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0, (T0 *)p_a[0]);
  op_mpi_reduce(&arg1, (T1 *)p_a[1]);
  op_mpi_reduce(&arg2, (T2 *)p_a[2]);
  op_mpi_reduce(&arg3, (T3 *)p_a[3]);
  op_mpi_reduce(&arg4, (T4 *)p_a[4]);
  op_mpi_reduce(&arg5, (T5 *)p_a[5]);
  op_mpi_reduce(&arg6, (T6 *)p_a[6]);
  op_mpi_reduce(&arg7, (T7 *)p_a[7]);
  op_mpi_reduce(&arg8, (T8 *)p_a[8]);
  op_mpi_reduce(&arg9, (T9 *)p_a[9]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 10, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if (arg0.idx < -1) {
    free(p_a[0]);
  }
  if (arg1.idx < -1) {
    free(p_a[1]);
  }
  if (arg2.idx < -1) {
    free(p_a[2]);
  }
  if (arg3.idx < -1) {
    free(p_a[3]);
  }
  if (arg4.idx < -1) {
    free(p_a[4]);
  }
  if (arg5.idx < -1) {
    free(p_a[5]);
  }
  if (arg6.idx < -1) {
    free(p_a[6]);
  }
  if (arg7.idx < -1) {
    free(p_a[7]);
  }
  if (arg8.idx < -1) {
    free(p_a[8]);
  }
  if (arg9.idx < -1) {
    free(p_a[9]);
  }
}
//
// op_par_loop routine for 11 arguments
//
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6,
          class T7, class T8, class T9, class T10>
void op_par_loop(void (*kernel)(T0 *, T1 *, T2 *, T3 *, T4 *, T5 *, T6 *, T7 *,
                                T8 *, T9 *, T10 *),
                 char const *name, op_set set, op_arg arg0, op_arg arg1,
                 op_arg arg2, op_arg arg3, op_arg arg4, op_arg arg5,
                 op_arg arg6, op_arg arg7, op_arg arg8, op_arg arg9,
                 op_arg arg10) {

  char *p_a[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  op_arg args[11] = {arg0, arg1, arg2, arg3, arg4, arg5,
                     arg6, arg7, arg8, arg9, arg10};
  if (arg0.idx < -1) {
    p_a[0] = (char *)op_malloc(-1 * args[0].idx * sizeof(T0));
  }
  if (arg1.idx < -1) {
    p_a[1] = (char *)op_malloc(-1 * args[1].idx * sizeof(T1));
  }
  if (arg2.idx < -1) {
    p_a[2] = (char *)op_malloc(-1 * args[2].idx * sizeof(T2));
  }
  if (arg3.idx < -1) {
    p_a[3] = (char *)op_malloc(-1 * args[3].idx * sizeof(T3));
  }
  if (arg4.idx < -1) {
    p_a[4] = (char *)op_malloc(-1 * args[4].idx * sizeof(T4));
  }
  if (arg5.idx < -1) {
    p_a[5] = (char *)op_malloc(-1 * args[5].idx * sizeof(T5));
  }
  if (arg6.idx < -1) {
    p_a[6] = (char *)op_malloc(-1 * args[6].idx * sizeof(T6));
  }
  if (arg7.idx < -1) {
    p_a[7] = (char *)op_malloc(-1 * args[7].idx * sizeof(T7));
  }
  if (arg8.idx < -1) {
    p_a[8] = (char *)op_malloc(-1 * args[8].idx * sizeof(T8));
  }
  if (arg9.idx < -1) {
    p_a[9] = (char *)op_malloc(-1 * args[9].idx * sizeof(T9));
  }
  if (arg10.idx < -1) {
    p_a[10] = (char *)op_malloc(-1 * args[10].idx * sizeof(T10));
  }

  // allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i < 11; i++)
    if (args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size) {
      blank_args_size = args[i].size;
      blank_args = (char *)op_malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags > 0)
    op_args_check(set, 11, args, &ninds, name);

  if (OP_diags > 2) {
    if (ninds == 0)
      printf(" kernel routine w/o indirection:  %s\n", name);
    else
      printf(" kernel routine with indirection: %s\n", name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 11, args);

  // loop over set elements
  int halo = 0;

  for (int n = 0; n < n_upper; n++) {
    if (n == set->core_size)
      op_mpi_wait_all(11, args);
    if (n == set->size)
      halo = 1;
    if (args[0].idx < -1)
      op_arg_copy_in(n, args[0], (char **)p_a[0]);
    else
      op_arg_set(n, args[0], &p_a[0], halo);
    if (args[1].idx < -1)
      op_arg_copy_in(n, args[1], (char **)p_a[1]);
    else
      op_arg_set(n, args[1], &p_a[1], halo);
    if (args[2].idx < -1)
      op_arg_copy_in(n, args[2], (char **)p_a[2]);
    else
      op_arg_set(n, args[2], &p_a[2], halo);
    if (args[3].idx < -1)
      op_arg_copy_in(n, args[3], (char **)p_a[3]);
    else
      op_arg_set(n, args[3], &p_a[3], halo);
    if (args[4].idx < -1)
      op_arg_copy_in(n, args[4], (char **)p_a[4]);
    else
      op_arg_set(n, args[4], &p_a[4], halo);
    if (args[5].idx < -1)
      op_arg_copy_in(n, args[5], (char **)p_a[5]);
    else
      op_arg_set(n, args[5], &p_a[5], halo);
    if (args[6].idx < -1)
      op_arg_copy_in(n, args[6], (char **)p_a[6]);
    else
      op_arg_set(n, args[6], &p_a[6], halo);
    if (args[7].idx < -1)
      op_arg_copy_in(n, args[7], (char **)p_a[7]);
    else
      op_arg_set(n, args[7], &p_a[7], halo);
    if (args[8].idx < -1)
      op_arg_copy_in(n, args[8], (char **)p_a[8]);
    else
      op_arg_set(n, args[8], &p_a[8], halo);
    if (args[9].idx < -1)
      op_arg_copy_in(n, args[9], (char **)p_a[9]);
    else
      op_arg_set(n, args[9], &p_a[9], halo);
    if (args[10].idx < -1)
      op_arg_copy_in(n, args[10], (char **)p_a[10]);
    else
      op_arg_set(n, args[10], &p_a[10], halo);

    kernel((T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3], (T4 *)p_a[4],
           (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7], (T8 *)p_a[8], (T9 *)p_a[9],
           (T10 *)p_a[10]);
  }
  if (n_upper == set->core_size || n_upper == 0)
    op_mpi_wait_all(11, args);

  // set dirty bit on datasets touched
  op_mpi_set_dirtybit(11, args);

  // global reduction for MPI execution, if needed
  // p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0, (T0 *)p_a[0]);
  op_mpi_reduce(&arg1, (T1 *)p_a[1]);
  op_mpi_reduce(&arg2, (T2 *)p_a[2]);
  op_mpi_reduce(&arg3, (T3 *)p_a[3]);
  op_mpi_reduce(&arg4, (T4 *)p_a[4]);
  op_mpi_reduce(&arg5, (T5 *)p_a[5]);
  op_mpi_reduce(&arg6, (T6 *)p_a[6]);
  op_mpi_reduce(&arg7, (T7 *)p_a[7]);
  op_mpi_reduce(&arg8, (T8 *)p_a[8]);
  op_mpi_reduce(&arg9, (T9 *)p_a[9]);
  op_mpi_reduce(&arg10, (T10 *)p_a[10]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 11, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if (arg0.idx < -1) {
    free(p_a[0]);
  }
  if (arg1.idx < -1) {
    free(p_a[1]);
  }
  if (arg2.idx < -1) {
    free(p_a[2]);
  }
  if (arg3.idx < -1) {
    free(p_a[3]);
  }
  if (arg4.idx < -1) {
    free(p_a[4]);
  }
  if (arg5.idx < -1) {
    free(p_a[5]);
  }
  if (arg6.idx < -1) {
    free(p_a[6]);
  }
  if (arg7.idx < -1) {
    free(p_a[7]);
  }
  if (arg8.idx < -1) {
    free(p_a[8]);
  }
  if (arg9.idx < -1) {
    free(p_a[9]);
  }
  if (arg10.idx < -1) {
    free(p_a[10]);
  }
}
//
// op_par_loop routine for 12 arguments
//
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6,
          class T7, class T8, class T9, class T10, class T11>
void op_par_loop(void (*kernel)(T0 *, T1 *, T2 *, T3 *, T4 *, T5 *, T6 *, T7 *,
                                T8 *, T9 *, T10 *, T11 *),
                 char const *name, op_set set, op_arg arg0, op_arg arg1,
                 op_arg arg2, op_arg arg3, op_arg arg4, op_arg arg5,
                 op_arg arg6, op_arg arg7, op_arg arg8, op_arg arg9,
                 op_arg arg10, op_arg arg11) {

  char *p_a[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  op_arg args[12] = {arg0, arg1, arg2, arg3, arg4,  arg5,
                     arg6, arg7, arg8, arg9, arg10, arg11};
  if (arg0.idx < -1) {
    p_a[0] = (char *)op_malloc(-1 * args[0].idx * sizeof(T0));
  }
  if (arg1.idx < -1) {
    p_a[1] = (char *)op_malloc(-1 * args[1].idx * sizeof(T1));
  }
  if (arg2.idx < -1) {
    p_a[2] = (char *)op_malloc(-1 * args[2].idx * sizeof(T2));
  }
  if (arg3.idx < -1) {
    p_a[3] = (char *)op_malloc(-1 * args[3].idx * sizeof(T3));
  }
  if (arg4.idx < -1) {
    p_a[4] = (char *)op_malloc(-1 * args[4].idx * sizeof(T4));
  }
  if (arg5.idx < -1) {
    p_a[5] = (char *)op_malloc(-1 * args[5].idx * sizeof(T5));
  }
  if (arg6.idx < -1) {
    p_a[6] = (char *)op_malloc(-1 * args[6].idx * sizeof(T6));
  }
  if (arg7.idx < -1) {
    p_a[7] = (char *)op_malloc(-1 * args[7].idx * sizeof(T7));
  }
  if (arg8.idx < -1) {
    p_a[8] = (char *)op_malloc(-1 * args[8].idx * sizeof(T8));
  }
  if (arg9.idx < -1) {
    p_a[9] = (char *)op_malloc(-1 * args[9].idx * sizeof(T9));
  }
  if (arg10.idx < -1) {
    p_a[10] = (char *)op_malloc(-1 * args[10].idx * sizeof(T10));
  }
  if (arg11.idx < -1) {
    p_a[11] = (char *)op_malloc(-1 * args[11].idx * sizeof(T11));
  }

  // allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i < 12; i++)
    if (args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size) {
      blank_args_size = args[i].size;
      blank_args = (char *)op_malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags > 0)
    op_args_check(set, 12, args, &ninds, name);

  if (OP_diags > 2) {
    if (ninds == 0)
      printf(" kernel routine w/o indirection:  %s\n", name);
    else
      printf(" kernel routine with indirection: %s\n", name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 12, args);

  // loop over set elements
  int halo = 0;

  for (int n = 0; n < n_upper; n++) {
    if (n == set->core_size)
      op_mpi_wait_all(12, args);
    if (n == set->size)
      halo = 1;
    if (args[0].idx < -1)
      op_arg_copy_in(n, args[0], (char **)p_a[0]);
    else
      op_arg_set(n, args[0], &p_a[0], halo);
    if (args[1].idx < -1)
      op_arg_copy_in(n, args[1], (char **)p_a[1]);
    else
      op_arg_set(n, args[1], &p_a[1], halo);
    if (args[2].idx < -1)
      op_arg_copy_in(n, args[2], (char **)p_a[2]);
    else
      op_arg_set(n, args[2], &p_a[2], halo);
    if (args[3].idx < -1)
      op_arg_copy_in(n, args[3], (char **)p_a[3]);
    else
      op_arg_set(n, args[3], &p_a[3], halo);
    if (args[4].idx < -1)
      op_arg_copy_in(n, args[4], (char **)p_a[4]);
    else
      op_arg_set(n, args[4], &p_a[4], halo);
    if (args[5].idx < -1)
      op_arg_copy_in(n, args[5], (char **)p_a[5]);
    else
      op_arg_set(n, args[5], &p_a[5], halo);
    if (args[6].idx < -1)
      op_arg_copy_in(n, args[6], (char **)p_a[6]);
    else
      op_arg_set(n, args[6], &p_a[6], halo);
    if (args[7].idx < -1)
      op_arg_copy_in(n, args[7], (char **)p_a[7]);
    else
      op_arg_set(n, args[7], &p_a[7], halo);
    if (args[8].idx < -1)
      op_arg_copy_in(n, args[8], (char **)p_a[8]);
    else
      op_arg_set(n, args[8], &p_a[8], halo);
    if (args[9].idx < -1)
      op_arg_copy_in(n, args[9], (char **)p_a[9]);
    else
      op_arg_set(n, args[9], &p_a[9], halo);
    if (args[10].idx < -1)
      op_arg_copy_in(n, args[10], (char **)p_a[10]);
    else
      op_arg_set(n, args[10], &p_a[10], halo);
    if (args[11].idx < -1)
      op_arg_copy_in(n, args[11], (char **)p_a[11]);
    else
      op_arg_set(n, args[11], &p_a[11], halo);

    kernel((T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3], (T4 *)p_a[4],
           (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7], (T8 *)p_a[8], (T9 *)p_a[9],
           (T10 *)p_a[10], (T11 *)p_a[11]);
  }
  if (n_upper == set->core_size || n_upper == 0)
    op_mpi_wait_all(12, args);

  // set dirty bit on datasets touched
  op_mpi_set_dirtybit(12, args);

  // global reduction for MPI execution, if needed
  // p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0, (T0 *)p_a[0]);
  op_mpi_reduce(&arg1, (T1 *)p_a[1]);
  op_mpi_reduce(&arg2, (T2 *)p_a[2]);
  op_mpi_reduce(&arg3, (T3 *)p_a[3]);
  op_mpi_reduce(&arg4, (T4 *)p_a[4]);
  op_mpi_reduce(&arg5, (T5 *)p_a[5]);
  op_mpi_reduce(&arg6, (T6 *)p_a[6]);
  op_mpi_reduce(&arg7, (T7 *)p_a[7]);
  op_mpi_reduce(&arg8, (T8 *)p_a[8]);
  op_mpi_reduce(&arg9, (T9 *)p_a[9]);
  op_mpi_reduce(&arg10, (T10 *)p_a[10]);
  op_mpi_reduce(&arg11, (T11 *)p_a[11]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 12, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if (arg0.idx < -1) {
    free(p_a[0]);
  }
  if (arg1.idx < -1) {
    free(p_a[1]);
  }
  if (arg2.idx < -1) {
    free(p_a[2]);
  }
  if (arg3.idx < -1) {
    free(p_a[3]);
  }
  if (arg4.idx < -1) {
    free(p_a[4]);
  }
  if (arg5.idx < -1) {
    free(p_a[5]);
  }
  if (arg6.idx < -1) {
    free(p_a[6]);
  }
  if (arg7.idx < -1) {
    free(p_a[7]);
  }
  if (arg8.idx < -1) {
    free(p_a[8]);
  }
  if (arg9.idx < -1) {
    free(p_a[9]);
  }
  if (arg10.idx < -1) {
    free(p_a[10]);
  }
  if (arg11.idx < -1) {
    free(p_a[11]);
  }
}
//
// op_par_loop routine for 13 arguments
//
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6,
          class T7, class T8, class T9, class T10, class T11, class T12>
void op_par_loop(void (*kernel)(T0 *, T1 *, T2 *, T3 *, T4 *, T5 *, T6 *, T7 *,
                                T8 *, T9 *, T10 *, T11 *, T12 *),
                 char const *name, op_set set, op_arg arg0, op_arg arg1,
                 op_arg arg2, op_arg arg3, op_arg arg4, op_arg arg5,
                 op_arg arg6, op_arg arg7, op_arg arg8, op_arg arg9,
                 op_arg arg10, op_arg arg11, op_arg arg12) {

  char *p_a[13] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  op_arg args[13] = {arg0, arg1, arg2, arg3,  arg4,  arg5, arg6,
                     arg7, arg8, arg9, arg10, arg11, arg12};
  if (arg0.idx < -1) {
    p_a[0] = (char *)op_malloc(-1 * args[0].idx * sizeof(T0));
  }
  if (arg1.idx < -1) {
    p_a[1] = (char *)op_malloc(-1 * args[1].idx * sizeof(T1));
  }
  if (arg2.idx < -1) {
    p_a[2] = (char *)op_malloc(-1 * args[2].idx * sizeof(T2));
  }
  if (arg3.idx < -1) {
    p_a[3] = (char *)op_malloc(-1 * args[3].idx * sizeof(T3));
  }
  if (arg4.idx < -1) {
    p_a[4] = (char *)op_malloc(-1 * args[4].idx * sizeof(T4));
  }
  if (arg5.idx < -1) {
    p_a[5] = (char *)op_malloc(-1 * args[5].idx * sizeof(T5));
  }
  if (arg6.idx < -1) {
    p_a[6] = (char *)op_malloc(-1 * args[6].idx * sizeof(T6));
  }
  if (arg7.idx < -1) {
    p_a[7] = (char *)op_malloc(-1 * args[7].idx * sizeof(T7));
  }
  if (arg8.idx < -1) {
    p_a[8] = (char *)op_malloc(-1 * args[8].idx * sizeof(T8));
  }
  if (arg9.idx < -1) {
    p_a[9] = (char *)op_malloc(-1 * args[9].idx * sizeof(T9));
  }
  if (arg10.idx < -1) {
    p_a[10] = (char *)op_malloc(-1 * args[10].idx * sizeof(T10));
  }
  if (arg11.idx < -1) {
    p_a[11] = (char *)op_malloc(-1 * args[11].idx * sizeof(T11));
  }
  if (arg12.idx < -1) {
    p_a[12] = (char *)op_malloc(-1 * args[12].idx * sizeof(T12));
  }

  // allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i < 13; i++)
    if (args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size) {
      blank_args_size = args[i].size;
      blank_args = (char *)op_malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags > 0)
    op_args_check(set, 13, args, &ninds, name);

  if (OP_diags > 2) {
    if (ninds == 0)
      printf(" kernel routine w/o indirection:  %s\n", name);
    else
      printf(" kernel routine with indirection: %s\n", name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 13, args);

  // loop over set elements
  int halo = 0;

  for (int n = 0; n < n_upper; n++) {
    if (n == set->core_size)
      op_mpi_wait_all(13, args);
    if (n == set->size)
      halo = 1;
    if (args[0].idx < -1)
      op_arg_copy_in(n, args[0], (char **)p_a[0]);
    else
      op_arg_set(n, args[0], &p_a[0], halo);
    if (args[1].idx < -1)
      op_arg_copy_in(n, args[1], (char **)p_a[1]);
    else
      op_arg_set(n, args[1], &p_a[1], halo);
    if (args[2].idx < -1)
      op_arg_copy_in(n, args[2], (char **)p_a[2]);
    else
      op_arg_set(n, args[2], &p_a[2], halo);
    if (args[3].idx < -1)
      op_arg_copy_in(n, args[3], (char **)p_a[3]);
    else
      op_arg_set(n, args[3], &p_a[3], halo);
    if (args[4].idx < -1)
      op_arg_copy_in(n, args[4], (char **)p_a[4]);
    else
      op_arg_set(n, args[4], &p_a[4], halo);
    if (args[5].idx < -1)
      op_arg_copy_in(n, args[5], (char **)p_a[5]);
    else
      op_arg_set(n, args[5], &p_a[5], halo);
    if (args[6].idx < -1)
      op_arg_copy_in(n, args[6], (char **)p_a[6]);
    else
      op_arg_set(n, args[6], &p_a[6], halo);
    if (args[7].idx < -1)
      op_arg_copy_in(n, args[7], (char **)p_a[7]);
    else
      op_arg_set(n, args[7], &p_a[7], halo);
    if (args[8].idx < -1)
      op_arg_copy_in(n, args[8], (char **)p_a[8]);
    else
      op_arg_set(n, args[8], &p_a[8], halo);
    if (args[9].idx < -1)
      op_arg_copy_in(n, args[9], (char **)p_a[9]);
    else
      op_arg_set(n, args[9], &p_a[9], halo);
    if (args[10].idx < -1)
      op_arg_copy_in(n, args[10], (char **)p_a[10]);
    else
      op_arg_set(n, args[10], &p_a[10], halo);
    if (args[11].idx < -1)
      op_arg_copy_in(n, args[11], (char **)p_a[11]);
    else
      op_arg_set(n, args[11], &p_a[11], halo);
    if (args[12].idx < -1)
      op_arg_copy_in(n, args[12], (char **)p_a[12]);
    else
      op_arg_set(n, args[12], &p_a[12], halo);

    kernel((T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3], (T4 *)p_a[4],
           (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7], (T8 *)p_a[8], (T9 *)p_a[9],
           (T10 *)p_a[10], (T11 *)p_a[11], (T12 *)p_a[12]);
  }
  if (n_upper == set->core_size || n_upper == 0)
    op_mpi_wait_all(13, args);

  // set dirty bit on datasets touched
  op_mpi_set_dirtybit(13, args);

  // global reduction for MPI execution, if needed
  // p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0, (T0 *)p_a[0]);
  op_mpi_reduce(&arg1, (T1 *)p_a[1]);
  op_mpi_reduce(&arg2, (T2 *)p_a[2]);
  op_mpi_reduce(&arg3, (T3 *)p_a[3]);
  op_mpi_reduce(&arg4, (T4 *)p_a[4]);
  op_mpi_reduce(&arg5, (T5 *)p_a[5]);
  op_mpi_reduce(&arg6, (T6 *)p_a[6]);
  op_mpi_reduce(&arg7, (T7 *)p_a[7]);
  op_mpi_reduce(&arg8, (T8 *)p_a[8]);
  op_mpi_reduce(&arg9, (T9 *)p_a[9]);
  op_mpi_reduce(&arg10, (T10 *)p_a[10]);
  op_mpi_reduce(&arg11, (T11 *)p_a[11]);
  op_mpi_reduce(&arg12, (T12 *)p_a[12]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 13, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if (arg0.idx < -1) {
    free(p_a[0]);
  }
  if (arg1.idx < -1) {
    free(p_a[1]);
  }
  if (arg2.idx < -1) {
    free(p_a[2]);
  }
  if (arg3.idx < -1) {
    free(p_a[3]);
  }
  if (arg4.idx < -1) {
    free(p_a[4]);
  }
  if (arg5.idx < -1) {
    free(p_a[5]);
  }
  if (arg6.idx < -1) {
    free(p_a[6]);
  }
  if (arg7.idx < -1) {
    free(p_a[7]);
  }
  if (arg8.idx < -1) {
    free(p_a[8]);
  }
  if (arg9.idx < -1) {
    free(p_a[9]);
  }
  if (arg10.idx < -1) {
    free(p_a[10]);
  }
  if (arg11.idx < -1) {
    free(p_a[11]);
  }
  if (arg12.idx < -1) {
    free(p_a[12]);
  }
}
//
// op_par_loop routine for 14 arguments
//
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6,
          class T7, class T8, class T9, class T10, class T11, class T12,
          class T13>
void op_par_loop(void (*kernel)(T0 *, T1 *, T2 *, T3 *, T4 *, T5 *, T6 *, T7 *,
                                T8 *, T9 *, T10 *, T11 *, T12 *, T13 *),
                 char const *name, op_set set, op_arg arg0, op_arg arg1,
                 op_arg arg2, op_arg arg3, op_arg arg4, op_arg arg5,
                 op_arg arg6, op_arg arg7, op_arg arg8, op_arg arg9,
                 op_arg arg10, op_arg arg11, op_arg arg12, op_arg arg13) {

  char *p_a[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  op_arg args[14] = {arg0, arg1, arg2, arg3,  arg4,  arg5,  arg6,
                     arg7, arg8, arg9, arg10, arg11, arg12, arg13};
  if (arg0.idx < -1) {
    p_a[0] = (char *)op_malloc(-1 * args[0].idx * sizeof(T0));
  }
  if (arg1.idx < -1) {
    p_a[1] = (char *)op_malloc(-1 * args[1].idx * sizeof(T1));
  }
  if (arg2.idx < -1) {
    p_a[2] = (char *)op_malloc(-1 * args[2].idx * sizeof(T2));
  }
  if (arg3.idx < -1) {
    p_a[3] = (char *)op_malloc(-1 * args[3].idx * sizeof(T3));
  }
  if (arg4.idx < -1) {
    p_a[4] = (char *)op_malloc(-1 * args[4].idx * sizeof(T4));
  }
  if (arg5.idx < -1) {
    p_a[5] = (char *)op_malloc(-1 * args[5].idx * sizeof(T5));
  }
  if (arg6.idx < -1) {
    p_a[6] = (char *)op_malloc(-1 * args[6].idx * sizeof(T6));
  }
  if (arg7.idx < -1) {
    p_a[7] = (char *)op_malloc(-1 * args[7].idx * sizeof(T7));
  }
  if (arg8.idx < -1) {
    p_a[8] = (char *)op_malloc(-1 * args[8].idx * sizeof(T8));
  }
  if (arg9.idx < -1) {
    p_a[9] = (char *)op_malloc(-1 * args[9].idx * sizeof(T9));
  }
  if (arg10.idx < -1) {
    p_a[10] = (char *)op_malloc(-1 * args[10].idx * sizeof(T10));
  }
  if (arg11.idx < -1) {
    p_a[11] = (char *)op_malloc(-1 * args[11].idx * sizeof(T11));
  }
  if (arg12.idx < -1) {
    p_a[12] = (char *)op_malloc(-1 * args[12].idx * sizeof(T12));
  }
  if (arg13.idx < -1) {
    p_a[13] = (char *)op_malloc(-1 * args[13].idx * sizeof(T13));
  }

  // allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i < 14; i++)
    if (args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size) {
      blank_args_size = args[i].size;
      blank_args = (char *)op_malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags > 0)
    op_args_check(set, 14, args, &ninds, name);

  if (OP_diags > 2) {
    if (ninds == 0)
      printf(" kernel routine w/o indirection:  %s\n", name);
    else
      printf(" kernel routine with indirection: %s\n", name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 14, args);

  // loop over set elements
  int halo = 0;

  for (int n = 0; n < n_upper; n++) {
    if (n == set->core_size)
      op_mpi_wait_all(14, args);
    if (n == set->size)
      halo = 1;
    if (args[0].idx < -1)
      op_arg_copy_in(n, args[0], (char **)p_a[0]);
    else
      op_arg_set(n, args[0], &p_a[0], halo);
    if (args[1].idx < -1)
      op_arg_copy_in(n, args[1], (char **)p_a[1]);
    else
      op_arg_set(n, args[1], &p_a[1], halo);
    if (args[2].idx < -1)
      op_arg_copy_in(n, args[2], (char **)p_a[2]);
    else
      op_arg_set(n, args[2], &p_a[2], halo);
    if (args[3].idx < -1)
      op_arg_copy_in(n, args[3], (char **)p_a[3]);
    else
      op_arg_set(n, args[3], &p_a[3], halo);
    if (args[4].idx < -1)
      op_arg_copy_in(n, args[4], (char **)p_a[4]);
    else
      op_arg_set(n, args[4], &p_a[4], halo);
    if (args[5].idx < -1)
      op_arg_copy_in(n, args[5], (char **)p_a[5]);
    else
      op_arg_set(n, args[5], &p_a[5], halo);
    if (args[6].idx < -1)
      op_arg_copy_in(n, args[6], (char **)p_a[6]);
    else
      op_arg_set(n, args[6], &p_a[6], halo);
    if (args[7].idx < -1)
      op_arg_copy_in(n, args[7], (char **)p_a[7]);
    else
      op_arg_set(n, args[7], &p_a[7], halo);
    if (args[8].idx < -1)
      op_arg_copy_in(n, args[8], (char **)p_a[8]);
    else
      op_arg_set(n, args[8], &p_a[8], halo);
    if (args[9].idx < -1)
      op_arg_copy_in(n, args[9], (char **)p_a[9]);
    else
      op_arg_set(n, args[9], &p_a[9], halo);
    if (args[10].idx < -1)
      op_arg_copy_in(n, args[10], (char **)p_a[10]);
    else
      op_arg_set(n, args[10], &p_a[10], halo);
    if (args[11].idx < -1)
      op_arg_copy_in(n, args[11], (char **)p_a[11]);
    else
      op_arg_set(n, args[11], &p_a[11], halo);
    if (args[12].idx < -1)
      op_arg_copy_in(n, args[12], (char **)p_a[12]);
    else
      op_arg_set(n, args[12], &p_a[12], halo);
    if (args[13].idx < -1)
      op_arg_copy_in(n, args[13], (char **)p_a[13]);
    else
      op_arg_set(n, args[13], &p_a[13], halo);

    kernel((T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3], (T4 *)p_a[4],
           (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7], (T8 *)p_a[8], (T9 *)p_a[9],
           (T10 *)p_a[10], (T11 *)p_a[11], (T12 *)p_a[12], (T13 *)p_a[13]);
  }
  if (n_upper == set->core_size || n_upper == 0)
    op_mpi_wait_all(14, args);

  // set dirty bit on datasets touched
  op_mpi_set_dirtybit(14, args);

  // global reduction for MPI execution, if needed
  // p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0, (T0 *)p_a[0]);
  op_mpi_reduce(&arg1, (T1 *)p_a[1]);
  op_mpi_reduce(&arg2, (T2 *)p_a[2]);
  op_mpi_reduce(&arg3, (T3 *)p_a[3]);
  op_mpi_reduce(&arg4, (T4 *)p_a[4]);
  op_mpi_reduce(&arg5, (T5 *)p_a[5]);
  op_mpi_reduce(&arg6, (T6 *)p_a[6]);
  op_mpi_reduce(&arg7, (T7 *)p_a[7]);
  op_mpi_reduce(&arg8, (T8 *)p_a[8]);
  op_mpi_reduce(&arg9, (T9 *)p_a[9]);
  op_mpi_reduce(&arg10, (T10 *)p_a[10]);
  op_mpi_reduce(&arg11, (T11 *)p_a[11]);
  op_mpi_reduce(&arg12, (T12 *)p_a[12]);
  op_mpi_reduce(&arg13, (T13 *)p_a[13]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 14, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if (arg0.idx < -1) {
    free(p_a[0]);
  }
  if (arg1.idx < -1) {
    free(p_a[1]);
  }
  if (arg2.idx < -1) {
    free(p_a[2]);
  }
  if (arg3.idx < -1) {
    free(p_a[3]);
  }
  if (arg4.idx < -1) {
    free(p_a[4]);
  }
  if (arg5.idx < -1) {
    free(p_a[5]);
  }
  if (arg6.idx < -1) {
    free(p_a[6]);
  }
  if (arg7.idx < -1) {
    free(p_a[7]);
  }
  if (arg8.idx < -1) {
    free(p_a[8]);
  }
  if (arg9.idx < -1) {
    free(p_a[9]);
  }
  if (arg10.idx < -1) {
    free(p_a[10]);
  }
  if (arg11.idx < -1) {
    free(p_a[11]);
  }
  if (arg12.idx < -1) {
    free(p_a[12]);
  }
  if (arg13.idx < -1) {
    free(p_a[13]);
  }
}
//
// op_par_loop routine for 15 arguments
//
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6,
          class T7, class T8, class T9, class T10, class T11, class T12,
          class T13, class T14>
void op_par_loop(void (*kernel)(T0 *, T1 *, T2 *, T3 *, T4 *, T5 *, T6 *, T7 *,
                                T8 *, T9 *, T10 *, T11 *, T12 *, T13 *, T14 *),
                 char const *name, op_set set, op_arg arg0, op_arg arg1,
                 op_arg arg2, op_arg arg3, op_arg arg4, op_arg arg5,
                 op_arg arg6, op_arg arg7, op_arg arg8, op_arg arg9,
                 op_arg arg10, op_arg arg11, op_arg arg12, op_arg arg13,
                 op_arg arg14) {

  char *p_a[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  op_arg args[15] = {arg0, arg1, arg2,  arg3,  arg4,  arg5,  arg6, arg7,
                     arg8, arg9, arg10, arg11, arg12, arg13, arg14};
  if (arg0.idx < -1) {
    p_a[0] = (char *)op_malloc(-1 * args[0].idx * sizeof(T0));
  }
  if (arg1.idx < -1) {
    p_a[1] = (char *)op_malloc(-1 * args[1].idx * sizeof(T1));
  }
  if (arg2.idx < -1) {
    p_a[2] = (char *)op_malloc(-1 * args[2].idx * sizeof(T2));
  }
  if (arg3.idx < -1) {
    p_a[3] = (char *)op_malloc(-1 * args[3].idx * sizeof(T3));
  }
  if (arg4.idx < -1) {
    p_a[4] = (char *)op_malloc(-1 * args[4].idx * sizeof(T4));
  }
  if (arg5.idx < -1) {
    p_a[5] = (char *)op_malloc(-1 * args[5].idx * sizeof(T5));
  }
  if (arg6.idx < -1) {
    p_a[6] = (char *)op_malloc(-1 * args[6].idx * sizeof(T6));
  }
  if (arg7.idx < -1) {
    p_a[7] = (char *)op_malloc(-1 * args[7].idx * sizeof(T7));
  }
  if (arg8.idx < -1) {
    p_a[8] = (char *)op_malloc(-1 * args[8].idx * sizeof(T8));
  }
  if (arg9.idx < -1) {
    p_a[9] = (char *)op_malloc(-1 * args[9].idx * sizeof(T9));
  }
  if (arg10.idx < -1) {
    p_a[10] = (char *)op_malloc(-1 * args[10].idx * sizeof(T10));
  }
  if (arg11.idx < -1) {
    p_a[11] = (char *)op_malloc(-1 * args[11].idx * sizeof(T11));
  }
  if (arg12.idx < -1) {
    p_a[12] = (char *)op_malloc(-1 * args[12].idx * sizeof(T12));
  }
  if (arg13.idx < -1) {
    p_a[13] = (char *)op_malloc(-1 * args[13].idx * sizeof(T13));
  }
  if (arg14.idx < -1) {
    p_a[14] = (char *)op_malloc(-1 * args[14].idx * sizeof(T14));
  }

  // allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i < 15; i++)
    if (args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size) {
      blank_args_size = args[i].size;
      blank_args = (char *)op_malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags > 0)
    op_args_check(set, 15, args, &ninds, name);

  if (OP_diags > 2) {
    if (ninds == 0)
      printf(" kernel routine w/o indirection:  %s\n", name);
    else
      printf(" kernel routine with indirection: %s\n", name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 15, args);

  // loop over set elements
  int halo = 0;

  for (int n = 0; n < n_upper; n++) {
    if (n == set->core_size)
      op_mpi_wait_all(15, args);
    if (n == set->size)
      halo = 1;
    if (args[0].idx < -1)
      op_arg_copy_in(n, args[0], (char **)p_a[0]);
    else
      op_arg_set(n, args[0], &p_a[0], halo);
    if (args[1].idx < -1)
      op_arg_copy_in(n, args[1], (char **)p_a[1]);
    else
      op_arg_set(n, args[1], &p_a[1], halo);
    if (args[2].idx < -1)
      op_arg_copy_in(n, args[2], (char **)p_a[2]);
    else
      op_arg_set(n, args[2], &p_a[2], halo);
    if (args[3].idx < -1)
      op_arg_copy_in(n, args[3], (char **)p_a[3]);
    else
      op_arg_set(n, args[3], &p_a[3], halo);
    if (args[4].idx < -1)
      op_arg_copy_in(n, args[4], (char **)p_a[4]);
    else
      op_arg_set(n, args[4], &p_a[4], halo);
    if (args[5].idx < -1)
      op_arg_copy_in(n, args[5], (char **)p_a[5]);
    else
      op_arg_set(n, args[5], &p_a[5], halo);
    if (args[6].idx < -1)
      op_arg_copy_in(n, args[6], (char **)p_a[6]);
    else
      op_arg_set(n, args[6], &p_a[6], halo);
    if (args[7].idx < -1)
      op_arg_copy_in(n, args[7], (char **)p_a[7]);
    else
      op_arg_set(n, args[7], &p_a[7], halo);
    if (args[8].idx < -1)
      op_arg_copy_in(n, args[8], (char **)p_a[8]);
    else
      op_arg_set(n, args[8], &p_a[8], halo);
    if (args[9].idx < -1)
      op_arg_copy_in(n, args[9], (char **)p_a[9]);
    else
      op_arg_set(n, args[9], &p_a[9], halo);
    if (args[10].idx < -1)
      op_arg_copy_in(n, args[10], (char **)p_a[10]);
    else
      op_arg_set(n, args[10], &p_a[10], halo);
    if (args[11].idx < -1)
      op_arg_copy_in(n, args[11], (char **)p_a[11]);
    else
      op_arg_set(n, args[11], &p_a[11], halo);
    if (args[12].idx < -1)
      op_arg_copy_in(n, args[12], (char **)p_a[12]);
    else
      op_arg_set(n, args[12], &p_a[12], halo);
    if (args[13].idx < -1)
      op_arg_copy_in(n, args[13], (char **)p_a[13]);
    else
      op_arg_set(n, args[13], &p_a[13], halo);
    if (args[14].idx < -1)
      op_arg_copy_in(n, args[14], (char **)p_a[14]);
    else
      op_arg_set(n, args[14], &p_a[14], halo);

    kernel((T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3], (T4 *)p_a[4],
           (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7], (T8 *)p_a[8], (T9 *)p_a[9],
           (T10 *)p_a[10], (T11 *)p_a[11], (T12 *)p_a[12], (T13 *)p_a[13],
           (T14 *)p_a[14]);
  }
  if (n_upper == set->core_size || n_upper == 0)
    op_mpi_wait_all(15, args);

  // set dirty bit on datasets touched
  op_mpi_set_dirtybit(15, args);

  // global reduction for MPI execution, if needed
  // p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0, (T0 *)p_a[0]);
  op_mpi_reduce(&arg1, (T1 *)p_a[1]);
  op_mpi_reduce(&arg2, (T2 *)p_a[2]);
  op_mpi_reduce(&arg3, (T3 *)p_a[3]);
  op_mpi_reduce(&arg4, (T4 *)p_a[4]);
  op_mpi_reduce(&arg5, (T5 *)p_a[5]);
  op_mpi_reduce(&arg6, (T6 *)p_a[6]);
  op_mpi_reduce(&arg7, (T7 *)p_a[7]);
  op_mpi_reduce(&arg8, (T8 *)p_a[8]);
  op_mpi_reduce(&arg9, (T9 *)p_a[9]);
  op_mpi_reduce(&arg10, (T10 *)p_a[10]);
  op_mpi_reduce(&arg11, (T11 *)p_a[11]);
  op_mpi_reduce(&arg12, (T12 *)p_a[12]);
  op_mpi_reduce(&arg13, (T13 *)p_a[13]);
  op_mpi_reduce(&arg14, (T14 *)p_a[14]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 15, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if (arg0.idx < -1) {
    free(p_a[0]);
  }
  if (arg1.idx < -1) {
    free(p_a[1]);
  }
  if (arg2.idx < -1) {
    free(p_a[2]);
  }
  if (arg3.idx < -1) {
    free(p_a[3]);
  }
  if (arg4.idx < -1) {
    free(p_a[4]);
  }
  if (arg5.idx < -1) {
    free(p_a[5]);
  }
  if (arg6.idx < -1) {
    free(p_a[6]);
  }
  if (arg7.idx < -1) {
    free(p_a[7]);
  }
  if (arg8.idx < -1) {
    free(p_a[8]);
  }
  if (arg9.idx < -1) {
    free(p_a[9]);
  }
  if (arg10.idx < -1) {
    free(p_a[10]);
  }
  if (arg11.idx < -1) {
    free(p_a[11]);
  }
  if (arg12.idx < -1) {
    free(p_a[12]);
  }
  if (arg13.idx < -1) {
    free(p_a[13]);
  }
  if (arg14.idx < -1) {
    free(p_a[14]);
  }
}
//
// op_par_loop routine for 16 arguments
//
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6,
          class T7, class T8, class T9, class T10, class T11, class T12,
          class T13, class T14, class T15>
void op_par_loop(void (*kernel)(T0 *, T1 *, T2 *, T3 *, T4 *, T5 *, T6 *, T7 *,
                                T8 *, T9 *, T10 *, T11 *, T12 *, T13 *, T14 *,
                                T15 *),
                 char const *name, op_set set, op_arg arg0, op_arg arg1,
                 op_arg arg2, op_arg arg3, op_arg arg4, op_arg arg5,
                 op_arg arg6, op_arg arg7, op_arg arg8, op_arg arg9,
                 op_arg arg10, op_arg arg11, op_arg arg12, op_arg arg13,
                 op_arg arg14, op_arg arg15) {

  char *p_a[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  op_arg args[16] = {arg0, arg1, arg2,  arg3,  arg4,  arg5,  arg6,  arg7,
                     arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15};
  if (arg0.idx < -1) {
    p_a[0] = (char *)op_malloc(-1 * args[0].idx * sizeof(T0));
  }
  if (arg1.idx < -1) {
    p_a[1] = (char *)op_malloc(-1 * args[1].idx * sizeof(T1));
  }
  if (arg2.idx < -1) {
    p_a[2] = (char *)op_malloc(-1 * args[2].idx * sizeof(T2));
  }
  if (arg3.idx < -1) {
    p_a[3] = (char *)op_malloc(-1 * args[3].idx * sizeof(T3));
  }
  if (arg4.idx < -1) {
    p_a[4] = (char *)op_malloc(-1 * args[4].idx * sizeof(T4));
  }
  if (arg5.idx < -1) {
    p_a[5] = (char *)op_malloc(-1 * args[5].idx * sizeof(T5));
  }
  if (arg6.idx < -1) {
    p_a[6] = (char *)op_malloc(-1 * args[6].idx * sizeof(T6));
  }
  if (arg7.idx < -1) {
    p_a[7] = (char *)op_malloc(-1 * args[7].idx * sizeof(T7));
  }
  if (arg8.idx < -1) {
    p_a[8] = (char *)op_malloc(-1 * args[8].idx * sizeof(T8));
  }
  if (arg9.idx < -1) {
    p_a[9] = (char *)op_malloc(-1 * args[9].idx * sizeof(T9));
  }
  if (arg10.idx < -1) {
    p_a[10] = (char *)op_malloc(-1 * args[10].idx * sizeof(T10));
  }
  if (arg11.idx < -1) {
    p_a[11] = (char *)op_malloc(-1 * args[11].idx * sizeof(T11));
  }
  if (arg12.idx < -1) {
    p_a[12] = (char *)op_malloc(-1 * args[12].idx * sizeof(T12));
  }
  if (arg13.idx < -1) {
    p_a[13] = (char *)op_malloc(-1 * args[13].idx * sizeof(T13));
  }
  if (arg14.idx < -1) {
    p_a[14] = (char *)op_malloc(-1 * args[14].idx * sizeof(T14));
  }
  if (arg15.idx < -1) {
    p_a[15] = (char *)op_malloc(-1 * args[15].idx * sizeof(T15));
  }

  // allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i < 16; i++)
    if (args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size) {
      blank_args_size = args[i].size;
      blank_args = (char *)op_malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags > 0)
    op_args_check(set, 16, args, &ninds, name);

  if (OP_diags > 2) {
    if (ninds == 0)
      printf(" kernel routine w/o indirection:  %s\n", name);
    else
      printf(" kernel routine with indirection: %s\n", name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 16, args);

  // loop over set elements
  int halo = 0;

  for (int n = 0; n < n_upper; n++) {
    if (n == set->core_size)
      op_mpi_wait_all(16, args);
    if (n == set->size)
      halo = 1;
    if (args[0].idx < -1)
      op_arg_copy_in(n, args[0], (char **)p_a[0]);
    else
      op_arg_set(n, args[0], &p_a[0], halo);
    if (args[1].idx < -1)
      op_arg_copy_in(n, args[1], (char **)p_a[1]);
    else
      op_arg_set(n, args[1], &p_a[1], halo);
    if (args[2].idx < -1)
      op_arg_copy_in(n, args[2], (char **)p_a[2]);
    else
      op_arg_set(n, args[2], &p_a[2], halo);
    if (args[3].idx < -1)
      op_arg_copy_in(n, args[3], (char **)p_a[3]);
    else
      op_arg_set(n, args[3], &p_a[3], halo);
    if (args[4].idx < -1)
      op_arg_copy_in(n, args[4], (char **)p_a[4]);
    else
      op_arg_set(n, args[4], &p_a[4], halo);
    if (args[5].idx < -1)
      op_arg_copy_in(n, args[5], (char **)p_a[5]);
    else
      op_arg_set(n, args[5], &p_a[5], halo);
    if (args[6].idx < -1)
      op_arg_copy_in(n, args[6], (char **)p_a[6]);
    else
      op_arg_set(n, args[6], &p_a[6], halo);
    if (args[7].idx < -1)
      op_arg_copy_in(n, args[7], (char **)p_a[7]);
    else
      op_arg_set(n, args[7], &p_a[7], halo);
    if (args[8].idx < -1)
      op_arg_copy_in(n, args[8], (char **)p_a[8]);
    else
      op_arg_set(n, args[8], &p_a[8], halo);
    if (args[9].idx < -1)
      op_arg_copy_in(n, args[9], (char **)p_a[9]);
    else
      op_arg_set(n, args[9], &p_a[9], halo);
    if (args[10].idx < -1)
      op_arg_copy_in(n, args[10], (char **)p_a[10]);
    else
      op_arg_set(n, args[10], &p_a[10], halo);
    if (args[11].idx < -1)
      op_arg_copy_in(n, args[11], (char **)p_a[11]);
    else
      op_arg_set(n, args[11], &p_a[11], halo);
    if (args[12].idx < -1)
      op_arg_copy_in(n, args[12], (char **)p_a[12]);
    else
      op_arg_set(n, args[12], &p_a[12], halo);
    if (args[13].idx < -1)
      op_arg_copy_in(n, args[13], (char **)p_a[13]);
    else
      op_arg_set(n, args[13], &p_a[13], halo);
    if (args[14].idx < -1)
      op_arg_copy_in(n, args[14], (char **)p_a[14]);
    else
      op_arg_set(n, args[14], &p_a[14], halo);
    if (args[15].idx < -1)
      op_arg_copy_in(n, args[15], (char **)p_a[15]);
    else
      op_arg_set(n, args[15], &p_a[15], halo);

    kernel((T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3], (T4 *)p_a[4],
           (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7], (T8 *)p_a[8], (T9 *)p_a[9],
           (T10 *)p_a[10], (T11 *)p_a[11], (T12 *)p_a[12], (T13 *)p_a[13],
           (T14 *)p_a[14], (T15 *)p_a[15]);
  }
  if (n_upper == set->core_size || n_upper == 0)
    op_mpi_wait_all(16, args);

  // set dirty bit on datasets touched
  op_mpi_set_dirtybit(16, args);

  // global reduction for MPI execution, if needed
  // p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0, (T0 *)p_a[0]);
  op_mpi_reduce(&arg1, (T1 *)p_a[1]);
  op_mpi_reduce(&arg2, (T2 *)p_a[2]);
  op_mpi_reduce(&arg3, (T3 *)p_a[3]);
  op_mpi_reduce(&arg4, (T4 *)p_a[4]);
  op_mpi_reduce(&arg5, (T5 *)p_a[5]);
  op_mpi_reduce(&arg6, (T6 *)p_a[6]);
  op_mpi_reduce(&arg7, (T7 *)p_a[7]);
  op_mpi_reduce(&arg8, (T8 *)p_a[8]);
  op_mpi_reduce(&arg9, (T9 *)p_a[9]);
  op_mpi_reduce(&arg10, (T10 *)p_a[10]);
  op_mpi_reduce(&arg11, (T11 *)p_a[11]);
  op_mpi_reduce(&arg12, (T12 *)p_a[12]);
  op_mpi_reduce(&arg13, (T13 *)p_a[13]);
  op_mpi_reduce(&arg14, (T14 *)p_a[14]);
  op_mpi_reduce(&arg15, (T15 *)p_a[15]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 16, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if (arg0.idx < -1) {
    free(p_a[0]);
  }
  if (arg1.idx < -1) {
    free(p_a[1]);
  }
  if (arg2.idx < -1) {
    free(p_a[2]);
  }
  if (arg3.idx < -1) {
    free(p_a[3]);
  }
  if (arg4.idx < -1) {
    free(p_a[4]);
  }
  if (arg5.idx < -1) {
    free(p_a[5]);
  }
  if (arg6.idx < -1) {
    free(p_a[6]);
  }
  if (arg7.idx < -1) {
    free(p_a[7]);
  }
  if (arg8.idx < -1) {
    free(p_a[8]);
  }
  if (arg9.idx < -1) {
    free(p_a[9]);
  }
  if (arg10.idx < -1) {
    free(p_a[10]);
  }
  if (arg11.idx < -1) {
    free(p_a[11]);
  }
  if (arg12.idx < -1) {
    free(p_a[12]);
  }
  if (arg13.idx < -1) {
    free(p_a[13]);
  }
  if (arg14.idx < -1) {
    free(p_a[14]);
  }
  if (arg15.idx < -1) {
    free(p_a[15]);
  }
}
//
// op_par_loop routine for 17 arguments
//
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6,
          class T7, class T8, class T9, class T10, class T11, class T12,
          class T13, class T14, class T15, class T16>
void op_par_loop(void (*kernel)(T0 *, T1 *, T2 *, T3 *, T4 *, T5 *, T6 *, T7 *,
                                T8 *, T9 *, T10 *, T11 *, T12 *, T13 *, T14 *,
                                T15 *, T16 *),
                 char const *name, op_set set, op_arg arg0, op_arg arg1,
                 op_arg arg2, op_arg arg3, op_arg arg4, op_arg arg5,
                 op_arg arg6, op_arg arg7, op_arg arg8, op_arg arg9,
                 op_arg arg10, op_arg arg11, op_arg arg12, op_arg arg13,
                 op_arg arg14, op_arg arg15, op_arg arg16) {

  char *p_a[17] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  op_arg args[17] = {arg0, arg1,  arg2,  arg3,  arg4,  arg5,  arg6,  arg7, arg8,
                     arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16};
  if (arg0.idx < -1) {
    p_a[0] = (char *)op_malloc(-1 * args[0].idx * sizeof(T0));
  }
  if (arg1.idx < -1) {
    p_a[1] = (char *)op_malloc(-1 * args[1].idx * sizeof(T1));
  }
  if (arg2.idx < -1) {
    p_a[2] = (char *)op_malloc(-1 * args[2].idx * sizeof(T2));
  }
  if (arg3.idx < -1) {
    p_a[3] = (char *)op_malloc(-1 * args[3].idx * sizeof(T3));
  }
  if (arg4.idx < -1) {
    p_a[4] = (char *)op_malloc(-1 * args[4].idx * sizeof(T4));
  }
  if (arg5.idx < -1) {
    p_a[5] = (char *)op_malloc(-1 * args[5].idx * sizeof(T5));
  }
  if (arg6.idx < -1) {
    p_a[6] = (char *)op_malloc(-1 * args[6].idx * sizeof(T6));
  }
  if (arg7.idx < -1) {
    p_a[7] = (char *)op_malloc(-1 * args[7].idx * sizeof(T7));
  }
  if (arg8.idx < -1) {
    p_a[8] = (char *)op_malloc(-1 * args[8].idx * sizeof(T8));
  }
  if (arg9.idx < -1) {
    p_a[9] = (char *)op_malloc(-1 * args[9].idx * sizeof(T9));
  }
  if (arg10.idx < -1) {
    p_a[10] = (char *)op_malloc(-1 * args[10].idx * sizeof(T10));
  }
  if (arg11.idx < -1) {
    p_a[11] = (char *)op_malloc(-1 * args[11].idx * sizeof(T11));
  }
  if (arg12.idx < -1) {
    p_a[12] = (char *)op_malloc(-1 * args[12].idx * sizeof(T12));
  }
  if (arg13.idx < -1) {
    p_a[13] = (char *)op_malloc(-1 * args[13].idx * sizeof(T13));
  }
  if (arg14.idx < -1) {
    p_a[14] = (char *)op_malloc(-1 * args[14].idx * sizeof(T14));
  }
  if (arg15.idx < -1) {
    p_a[15] = (char *)op_malloc(-1 * args[15].idx * sizeof(T15));
  }
  if (arg16.idx < -1) {
    p_a[16] = (char *)op_malloc(-1 * args[16].idx * sizeof(T16));
  }

  // allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i < 17; i++)
    if (args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size) {
      blank_args_size = args[i].size;
      blank_args = (char *)op_malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags > 0)
    op_args_check(set, 17, args, &ninds, name);

  if (OP_diags > 2) {
    if (ninds == 0)
      printf(" kernel routine w/o indirection:  %s\n", name);
    else
      printf(" kernel routine with indirection: %s\n", name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 17, args);

  // loop over set elements
  int halo = 0;

  for (int n = 0; n < n_upper; n++) {
    if (n == set->core_size)
      op_mpi_wait_all(17, args);
    if (n == set->size)
      halo = 1;
    if (args[0].idx < -1)
      op_arg_copy_in(n, args[0], (char **)p_a[0]);
    else
      op_arg_set(n, args[0], &p_a[0], halo);
    if (args[1].idx < -1)
      op_arg_copy_in(n, args[1], (char **)p_a[1]);
    else
      op_arg_set(n, args[1], &p_a[1], halo);
    if (args[2].idx < -1)
      op_arg_copy_in(n, args[2], (char **)p_a[2]);
    else
      op_arg_set(n, args[2], &p_a[2], halo);
    if (args[3].idx < -1)
      op_arg_copy_in(n, args[3], (char **)p_a[3]);
    else
      op_arg_set(n, args[3], &p_a[3], halo);
    if (args[4].idx < -1)
      op_arg_copy_in(n, args[4], (char **)p_a[4]);
    else
      op_arg_set(n, args[4], &p_a[4], halo);
    if (args[5].idx < -1)
      op_arg_copy_in(n, args[5], (char **)p_a[5]);
    else
      op_arg_set(n, args[5], &p_a[5], halo);
    if (args[6].idx < -1)
      op_arg_copy_in(n, args[6], (char **)p_a[6]);
    else
      op_arg_set(n, args[6], &p_a[6], halo);
    if (args[7].idx < -1)
      op_arg_copy_in(n, args[7], (char **)p_a[7]);
    else
      op_arg_set(n, args[7], &p_a[7], halo);
    if (args[8].idx < -1)
      op_arg_copy_in(n, args[8], (char **)p_a[8]);
    else
      op_arg_set(n, args[8], &p_a[8], halo);
    if (args[9].idx < -1)
      op_arg_copy_in(n, args[9], (char **)p_a[9]);
    else
      op_arg_set(n, args[9], &p_a[9], halo);
    if (args[10].idx < -1)
      op_arg_copy_in(n, args[10], (char **)p_a[10]);
    else
      op_arg_set(n, args[10], &p_a[10], halo);
    if (args[11].idx < -1)
      op_arg_copy_in(n, args[11], (char **)p_a[11]);
    else
      op_arg_set(n, args[11], &p_a[11], halo);
    if (args[12].idx < -1)
      op_arg_copy_in(n, args[12], (char **)p_a[12]);
    else
      op_arg_set(n, args[12], &p_a[12], halo);
    if (args[13].idx < -1)
      op_arg_copy_in(n, args[13], (char **)p_a[13]);
    else
      op_arg_set(n, args[13], &p_a[13], halo);
    if (args[14].idx < -1)
      op_arg_copy_in(n, args[14], (char **)p_a[14]);
    else
      op_arg_set(n, args[14], &p_a[14], halo);
    if (args[15].idx < -1)
      op_arg_copy_in(n, args[15], (char **)p_a[15]);
    else
      op_arg_set(n, args[15], &p_a[15], halo);
    if (args[16].idx < -1)
      op_arg_copy_in(n, args[16], (char **)p_a[16]);
    else
      op_arg_set(n, args[16], &p_a[16], halo);

    kernel((T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3], (T4 *)p_a[4],
           (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7], (T8 *)p_a[8], (T9 *)p_a[9],
           (T10 *)p_a[10], (T11 *)p_a[11], (T12 *)p_a[12], (T13 *)p_a[13],
           (T14 *)p_a[14], (T15 *)p_a[15], (T16 *)p_a[16]);
  }
  if (n_upper == set->core_size || n_upper == 0)
    op_mpi_wait_all(17, args);

  // set dirty bit on datasets touched
  op_mpi_set_dirtybit(17, args);

  // global reduction for MPI execution, if needed
  // p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0, (T0 *)p_a[0]);
  op_mpi_reduce(&arg1, (T1 *)p_a[1]);
  op_mpi_reduce(&arg2, (T2 *)p_a[2]);
  op_mpi_reduce(&arg3, (T3 *)p_a[3]);
  op_mpi_reduce(&arg4, (T4 *)p_a[4]);
  op_mpi_reduce(&arg5, (T5 *)p_a[5]);
  op_mpi_reduce(&arg6, (T6 *)p_a[6]);
  op_mpi_reduce(&arg7, (T7 *)p_a[7]);
  op_mpi_reduce(&arg8, (T8 *)p_a[8]);
  op_mpi_reduce(&arg9, (T9 *)p_a[9]);
  op_mpi_reduce(&arg10, (T10 *)p_a[10]);
  op_mpi_reduce(&arg11, (T11 *)p_a[11]);
  op_mpi_reduce(&arg12, (T12 *)p_a[12]);
  op_mpi_reduce(&arg13, (T13 *)p_a[13]);
  op_mpi_reduce(&arg14, (T14 *)p_a[14]);
  op_mpi_reduce(&arg15, (T15 *)p_a[15]);
  op_mpi_reduce(&arg16, (T16 *)p_a[16]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 17, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if (arg0.idx < -1) {
    free(p_a[0]);
  }
  if (arg1.idx < -1) {
    free(p_a[1]);
  }
  if (arg2.idx < -1) {
    free(p_a[2]);
  }
  if (arg3.idx < -1) {
    free(p_a[3]);
  }
  if (arg4.idx < -1) {
    free(p_a[4]);
  }
  if (arg5.idx < -1) {
    free(p_a[5]);
  }
  if (arg6.idx < -1) {
    free(p_a[6]);
  }
  if (arg7.idx < -1) {
    free(p_a[7]);
  }
  if (arg8.idx < -1) {
    free(p_a[8]);
  }
  if (arg9.idx < -1) {
    free(p_a[9]);
  }
  if (arg10.idx < -1) {
    free(p_a[10]);
  }
  if (arg11.idx < -1) {
    free(p_a[11]);
  }
  if (arg12.idx < -1) {
    free(p_a[12]);
  }
  if (arg13.idx < -1) {
    free(p_a[13]);
  }
  if (arg14.idx < -1) {
    free(p_a[14]);
  }
  if (arg15.idx < -1) {
    free(p_a[15]);
  }
  if (arg16.idx < -1) {
    free(p_a[16]);
  }
}
//
// op_par_loop routine for 18 arguments
//
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6,
          class T7, class T8, class T9, class T10, class T11, class T12,
          class T13, class T14, class T15, class T16, class T17>
void op_par_loop(void (*kernel)(T0 *, T1 *, T2 *, T3 *, T4 *, T5 *, T6 *, T7 *,
                                T8 *, T9 *, T10 *, T11 *, T12 *, T13 *, T14 *,
                                T15 *, T16 *, T17 *),
                 char const *name, op_set set, op_arg arg0, op_arg arg1,
                 op_arg arg2, op_arg arg3, op_arg arg4, op_arg arg5,
                 op_arg arg6, op_arg arg7, op_arg arg8, op_arg arg9,
                 op_arg arg10, op_arg arg11, op_arg arg12, op_arg arg13,
                 op_arg arg14, op_arg arg15, op_arg arg16, op_arg arg17) {

  char *p_a[18] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  op_arg args[18] = {arg0,  arg1,  arg2,  arg3,  arg4,  arg5,
                     arg6,  arg7,  arg8,  arg9,  arg10, arg11,
                     arg12, arg13, arg14, arg15, arg16, arg17};
  if (arg0.idx < -1) {
    p_a[0] = (char *)op_malloc(-1 * args[0].idx * sizeof(T0));
  }
  if (arg1.idx < -1) {
    p_a[1] = (char *)op_malloc(-1 * args[1].idx * sizeof(T1));
  }
  if (arg2.idx < -1) {
    p_a[2] = (char *)op_malloc(-1 * args[2].idx * sizeof(T2));
  }
  if (arg3.idx < -1) {
    p_a[3] = (char *)op_malloc(-1 * args[3].idx * sizeof(T3));
  }
  if (arg4.idx < -1) {
    p_a[4] = (char *)op_malloc(-1 * args[4].idx * sizeof(T4));
  }
  if (arg5.idx < -1) {
    p_a[5] = (char *)op_malloc(-1 * args[5].idx * sizeof(T5));
  }
  if (arg6.idx < -1) {
    p_a[6] = (char *)op_malloc(-1 * args[6].idx * sizeof(T6));
  }
  if (arg7.idx < -1) {
    p_a[7] = (char *)op_malloc(-1 * args[7].idx * sizeof(T7));
  }
  if (arg8.idx < -1) {
    p_a[8] = (char *)op_malloc(-1 * args[8].idx * sizeof(T8));
  }
  if (arg9.idx < -1) {
    p_a[9] = (char *)op_malloc(-1 * args[9].idx * sizeof(T9));
  }
  if (arg10.idx < -1) {
    p_a[10] = (char *)op_malloc(-1 * args[10].idx * sizeof(T10));
  }
  if (arg11.idx < -1) {
    p_a[11] = (char *)op_malloc(-1 * args[11].idx * sizeof(T11));
  }
  if (arg12.idx < -1) {
    p_a[12] = (char *)op_malloc(-1 * args[12].idx * sizeof(T12));
  }
  if (arg13.idx < -1) {
    p_a[13] = (char *)op_malloc(-1 * args[13].idx * sizeof(T13));
  }
  if (arg14.idx < -1) {
    p_a[14] = (char *)op_malloc(-1 * args[14].idx * sizeof(T14));
  }
  if (arg15.idx < -1) {
    p_a[15] = (char *)op_malloc(-1 * args[15].idx * sizeof(T15));
  }
  if (arg16.idx < -1) {
    p_a[16] = (char *)op_malloc(-1 * args[16].idx * sizeof(T16));
  }
  if (arg17.idx < -1) {
    p_a[17] = (char *)op_malloc(-1 * args[17].idx * sizeof(T17));
  }

  // allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i < 18; i++)
    if (args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size) {
      blank_args_size = args[i].size;
      blank_args = (char *)op_malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags > 0)
    op_args_check(set, 18, args, &ninds, name);

  if (OP_diags > 2) {
    if (ninds == 0)
      printf(" kernel routine w/o indirection:  %s\n", name);
    else
      printf(" kernel routine with indirection: %s\n", name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 18, args);

  // loop over set elements
  int halo = 0;

  for (int n = 0; n < n_upper; n++) {
    if (n == set->core_size)
      op_mpi_wait_all(18, args);
    if (n == set->size)
      halo = 1;
    if (args[0].idx < -1)
      op_arg_copy_in(n, args[0], (char **)p_a[0]);
    else
      op_arg_set(n, args[0], &p_a[0], halo);
    if (args[1].idx < -1)
      op_arg_copy_in(n, args[1], (char **)p_a[1]);
    else
      op_arg_set(n, args[1], &p_a[1], halo);
    if (args[2].idx < -1)
      op_arg_copy_in(n, args[2], (char **)p_a[2]);
    else
      op_arg_set(n, args[2], &p_a[2], halo);
    if (args[3].idx < -1)
      op_arg_copy_in(n, args[3], (char **)p_a[3]);
    else
      op_arg_set(n, args[3], &p_a[3], halo);
    if (args[4].idx < -1)
      op_arg_copy_in(n, args[4], (char **)p_a[4]);
    else
      op_arg_set(n, args[4], &p_a[4], halo);
    if (args[5].idx < -1)
      op_arg_copy_in(n, args[5], (char **)p_a[5]);
    else
      op_arg_set(n, args[5], &p_a[5], halo);
    if (args[6].idx < -1)
      op_arg_copy_in(n, args[6], (char **)p_a[6]);
    else
      op_arg_set(n, args[6], &p_a[6], halo);
    if (args[7].idx < -1)
      op_arg_copy_in(n, args[7], (char **)p_a[7]);
    else
      op_arg_set(n, args[7], &p_a[7], halo);
    if (args[8].idx < -1)
      op_arg_copy_in(n, args[8], (char **)p_a[8]);
    else
      op_arg_set(n, args[8], &p_a[8], halo);
    if (args[9].idx < -1)
      op_arg_copy_in(n, args[9], (char **)p_a[9]);
    else
      op_arg_set(n, args[9], &p_a[9], halo);
    if (args[10].idx < -1)
      op_arg_copy_in(n, args[10], (char **)p_a[10]);
    else
      op_arg_set(n, args[10], &p_a[10], halo);
    if (args[11].idx < -1)
      op_arg_copy_in(n, args[11], (char **)p_a[11]);
    else
      op_arg_set(n, args[11], &p_a[11], halo);
    if (args[12].idx < -1)
      op_arg_copy_in(n, args[12], (char **)p_a[12]);
    else
      op_arg_set(n, args[12], &p_a[12], halo);
    if (args[13].idx < -1)
      op_arg_copy_in(n, args[13], (char **)p_a[13]);
    else
      op_arg_set(n, args[13], &p_a[13], halo);
    if (args[14].idx < -1)
      op_arg_copy_in(n, args[14], (char **)p_a[14]);
    else
      op_arg_set(n, args[14], &p_a[14], halo);
    if (args[15].idx < -1)
      op_arg_copy_in(n, args[15], (char **)p_a[15]);
    else
      op_arg_set(n, args[15], &p_a[15], halo);
    if (args[16].idx < -1)
      op_arg_copy_in(n, args[16], (char **)p_a[16]);
    else
      op_arg_set(n, args[16], &p_a[16], halo);
    if (args[17].idx < -1)
      op_arg_copy_in(n, args[17], (char **)p_a[17]);
    else
      op_arg_set(n, args[17], &p_a[17], halo);

    kernel((T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3], (T4 *)p_a[4],
           (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7], (T8 *)p_a[8], (T9 *)p_a[9],
           (T10 *)p_a[10], (T11 *)p_a[11], (T12 *)p_a[12], (T13 *)p_a[13],
           (T14 *)p_a[14], (T15 *)p_a[15], (T16 *)p_a[16], (T17 *)p_a[17]);
  }
  if (n_upper == set->core_size || n_upper == 0)
    op_mpi_wait_all(18, args);

  // set dirty bit on datasets touched
  op_mpi_set_dirtybit(18, args);

  // global reduction for MPI execution, if needed
  // p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0, (T0 *)p_a[0]);
  op_mpi_reduce(&arg1, (T1 *)p_a[1]);
  op_mpi_reduce(&arg2, (T2 *)p_a[2]);
  op_mpi_reduce(&arg3, (T3 *)p_a[3]);
  op_mpi_reduce(&arg4, (T4 *)p_a[4]);
  op_mpi_reduce(&arg5, (T5 *)p_a[5]);
  op_mpi_reduce(&arg6, (T6 *)p_a[6]);
  op_mpi_reduce(&arg7, (T7 *)p_a[7]);
  op_mpi_reduce(&arg8, (T8 *)p_a[8]);
  op_mpi_reduce(&arg9, (T9 *)p_a[9]);
  op_mpi_reduce(&arg10, (T10 *)p_a[10]);
  op_mpi_reduce(&arg11, (T11 *)p_a[11]);
  op_mpi_reduce(&arg12, (T12 *)p_a[12]);
  op_mpi_reduce(&arg13, (T13 *)p_a[13]);
  op_mpi_reduce(&arg14, (T14 *)p_a[14]);
  op_mpi_reduce(&arg15, (T15 *)p_a[15]);
  op_mpi_reduce(&arg16, (T16 *)p_a[16]);
  op_mpi_reduce(&arg17, (T17 *)p_a[17]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 18, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if (arg0.idx < -1) {
    free(p_a[0]);
  }
  if (arg1.idx < -1) {
    free(p_a[1]);
  }
  if (arg2.idx < -1) {
    free(p_a[2]);
  }
  if (arg3.idx < -1) {
    free(p_a[3]);
  }
  if (arg4.idx < -1) {
    free(p_a[4]);
  }
  if (arg5.idx < -1) {
    free(p_a[5]);
  }
  if (arg6.idx < -1) {
    free(p_a[6]);
  }
  if (arg7.idx < -1) {
    free(p_a[7]);
  }
  if (arg8.idx < -1) {
    free(p_a[8]);
  }
  if (arg9.idx < -1) {
    free(p_a[9]);
  }
  if (arg10.idx < -1) {
    free(p_a[10]);
  }
  if (arg11.idx < -1) {
    free(p_a[11]);
  }
  if (arg12.idx < -1) {
    free(p_a[12]);
  }
  if (arg13.idx < -1) {
    free(p_a[13]);
  }
  if (arg14.idx < -1) {
    free(p_a[14]);
  }
  if (arg15.idx < -1) {
    free(p_a[15]);
  }
  if (arg16.idx < -1) {
    free(p_a[16]);
  }
  if (arg17.idx < -1) {
    free(p_a[17]);
  }
}
//
// op_par_loop routine for 19 arguments
//
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6,
          class T7, class T8, class T9, class T10, class T11, class T12,
          class T13, class T14, class T15, class T16, class T17, class T18>
void op_par_loop(void (*kernel)(T0 *, T1 *, T2 *, T3 *, T4 *, T5 *, T6 *, T7 *,
                                T8 *, T9 *, T10 *, T11 *, T12 *, T13 *, T14 *,
                                T15 *, T16 *, T17 *, T18 *),
                 char const *name, op_set set, op_arg arg0, op_arg arg1,
                 op_arg arg2, op_arg arg3, op_arg arg4, op_arg arg5,
                 op_arg arg6, op_arg arg7, op_arg arg8, op_arg arg9,
                 op_arg arg10, op_arg arg11, op_arg arg12, op_arg arg13,
                 op_arg arg14, op_arg arg15, op_arg arg16, op_arg arg17,
                 op_arg arg18) {

  char *p_a[19] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  op_arg args[19] = {arg0,  arg1,  arg2,  arg3,  arg4,  arg5,  arg6,
                     arg7,  arg8,  arg9,  arg10, arg11, arg12, arg13,
                     arg14, arg15, arg16, arg17, arg18};
  if (arg0.idx < -1) {
    p_a[0] = (char *)op_malloc(-1 * args[0].idx * sizeof(T0));
  }
  if (arg1.idx < -1) {
    p_a[1] = (char *)op_malloc(-1 * args[1].idx * sizeof(T1));
  }
  if (arg2.idx < -1) {
    p_a[2] = (char *)op_malloc(-1 * args[2].idx * sizeof(T2));
  }
  if (arg3.idx < -1) {
    p_a[3] = (char *)op_malloc(-1 * args[3].idx * sizeof(T3));
  }
  if (arg4.idx < -1) {
    p_a[4] = (char *)op_malloc(-1 * args[4].idx * sizeof(T4));
  }
  if (arg5.idx < -1) {
    p_a[5] = (char *)op_malloc(-1 * args[5].idx * sizeof(T5));
  }
  if (arg6.idx < -1) {
    p_a[6] = (char *)op_malloc(-1 * args[6].idx * sizeof(T6));
  }
  if (arg7.idx < -1) {
    p_a[7] = (char *)op_malloc(-1 * args[7].idx * sizeof(T7));
  }
  if (arg8.idx < -1) {
    p_a[8] = (char *)op_malloc(-1 * args[8].idx * sizeof(T8));
  }
  if (arg9.idx < -1) {
    p_a[9] = (char *)op_malloc(-1 * args[9].idx * sizeof(T9));
  }
  if (arg10.idx < -1) {
    p_a[10] = (char *)op_malloc(-1 * args[10].idx * sizeof(T10));
  }
  if (arg11.idx < -1) {
    p_a[11] = (char *)op_malloc(-1 * args[11].idx * sizeof(T11));
  }
  if (arg12.idx < -1) {
    p_a[12] = (char *)op_malloc(-1 * args[12].idx * sizeof(T12));
  }
  if (arg13.idx < -1) {
    p_a[13] = (char *)op_malloc(-1 * args[13].idx * sizeof(T13));
  }
  if (arg14.idx < -1) {
    p_a[14] = (char *)op_malloc(-1 * args[14].idx * sizeof(T14));
  }
  if (arg15.idx < -1) {
    p_a[15] = (char *)op_malloc(-1 * args[15].idx * sizeof(T15));
  }
  if (arg16.idx < -1) {
    p_a[16] = (char *)op_malloc(-1 * args[16].idx * sizeof(T16));
  }
  if (arg17.idx < -1) {
    p_a[17] = (char *)op_malloc(-1 * args[17].idx * sizeof(T17));
  }
  if (arg18.idx < -1) {
    p_a[18] = (char *)op_malloc(-1 * args[18].idx * sizeof(T18));
  }

  // allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i < 19; i++)
    if (args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size) {
      blank_args_size = args[i].size;
      blank_args = (char *)op_malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags > 0)
    op_args_check(set, 19, args, &ninds, name);

  if (OP_diags > 2) {
    if (ninds == 0)
      printf(" kernel routine w/o indirection:  %s\n", name);
    else
      printf(" kernel routine with indirection: %s\n", name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 19, args);

  // loop over set elements
  int halo = 0;

  for (int n = 0; n < n_upper; n++) {
    if (n == set->core_size)
      op_mpi_wait_all(19, args);
    if (n == set->size)
      halo = 1;
    if (args[0].idx < -1)
      op_arg_copy_in(n, args[0], (char **)p_a[0]);
    else
      op_arg_set(n, args[0], &p_a[0], halo);
    if (args[1].idx < -1)
      op_arg_copy_in(n, args[1], (char **)p_a[1]);
    else
      op_arg_set(n, args[1], &p_a[1], halo);
    if (args[2].idx < -1)
      op_arg_copy_in(n, args[2], (char **)p_a[2]);
    else
      op_arg_set(n, args[2], &p_a[2], halo);
    if (args[3].idx < -1)
      op_arg_copy_in(n, args[3], (char **)p_a[3]);
    else
      op_arg_set(n, args[3], &p_a[3], halo);
    if (args[4].idx < -1)
      op_arg_copy_in(n, args[4], (char **)p_a[4]);
    else
      op_arg_set(n, args[4], &p_a[4], halo);
    if (args[5].idx < -1)
      op_arg_copy_in(n, args[5], (char **)p_a[5]);
    else
      op_arg_set(n, args[5], &p_a[5], halo);
    if (args[6].idx < -1)
      op_arg_copy_in(n, args[6], (char **)p_a[6]);
    else
      op_arg_set(n, args[6], &p_a[6], halo);
    if (args[7].idx < -1)
      op_arg_copy_in(n, args[7], (char **)p_a[7]);
    else
      op_arg_set(n, args[7], &p_a[7], halo);
    if (args[8].idx < -1)
      op_arg_copy_in(n, args[8], (char **)p_a[8]);
    else
      op_arg_set(n, args[8], &p_a[8], halo);
    if (args[9].idx < -1)
      op_arg_copy_in(n, args[9], (char **)p_a[9]);
    else
      op_arg_set(n, args[9], &p_a[9], halo);
    if (args[10].idx < -1)
      op_arg_copy_in(n, args[10], (char **)p_a[10]);
    else
      op_arg_set(n, args[10], &p_a[10], halo);
    if (args[11].idx < -1)
      op_arg_copy_in(n, args[11], (char **)p_a[11]);
    else
      op_arg_set(n, args[11], &p_a[11], halo);
    if (args[12].idx < -1)
      op_arg_copy_in(n, args[12], (char **)p_a[12]);
    else
      op_arg_set(n, args[12], &p_a[12], halo);
    if (args[13].idx < -1)
      op_arg_copy_in(n, args[13], (char **)p_a[13]);
    else
      op_arg_set(n, args[13], &p_a[13], halo);
    if (args[14].idx < -1)
      op_arg_copy_in(n, args[14], (char **)p_a[14]);
    else
      op_arg_set(n, args[14], &p_a[14], halo);
    if (args[15].idx < -1)
      op_arg_copy_in(n, args[15], (char **)p_a[15]);
    else
      op_arg_set(n, args[15], &p_a[15], halo);
    if (args[16].idx < -1)
      op_arg_copy_in(n, args[16], (char **)p_a[16]);
    else
      op_arg_set(n, args[16], &p_a[16], halo);
    if (args[17].idx < -1)
      op_arg_copy_in(n, args[17], (char **)p_a[17]);
    else
      op_arg_set(n, args[17], &p_a[17], halo);
    if (args[18].idx < -1)
      op_arg_copy_in(n, args[18], (char **)p_a[18]);
    else
      op_arg_set(n, args[18], &p_a[18], halo);

    kernel((T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3], (T4 *)p_a[4],
           (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7], (T8 *)p_a[8], (T9 *)p_a[9],
           (T10 *)p_a[10], (T11 *)p_a[11], (T12 *)p_a[12], (T13 *)p_a[13],
           (T14 *)p_a[14], (T15 *)p_a[15], (T16 *)p_a[16], (T17 *)p_a[17],
           (T18 *)p_a[18]);
  }
  if (n_upper == set->core_size || n_upper == 0)
    op_mpi_wait_all(19, args);

  // set dirty bit on datasets touched
  op_mpi_set_dirtybit(19, args);

  // global reduction for MPI execution, if needed
  // p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0, (T0 *)p_a[0]);
  op_mpi_reduce(&arg1, (T1 *)p_a[1]);
  op_mpi_reduce(&arg2, (T2 *)p_a[2]);
  op_mpi_reduce(&arg3, (T3 *)p_a[3]);
  op_mpi_reduce(&arg4, (T4 *)p_a[4]);
  op_mpi_reduce(&arg5, (T5 *)p_a[5]);
  op_mpi_reduce(&arg6, (T6 *)p_a[6]);
  op_mpi_reduce(&arg7, (T7 *)p_a[7]);
  op_mpi_reduce(&arg8, (T8 *)p_a[8]);
  op_mpi_reduce(&arg9, (T9 *)p_a[9]);
  op_mpi_reduce(&arg10, (T10 *)p_a[10]);
  op_mpi_reduce(&arg11, (T11 *)p_a[11]);
  op_mpi_reduce(&arg12, (T12 *)p_a[12]);
  op_mpi_reduce(&arg13, (T13 *)p_a[13]);
  op_mpi_reduce(&arg14, (T14 *)p_a[14]);
  op_mpi_reduce(&arg15, (T15 *)p_a[15]);
  op_mpi_reduce(&arg16, (T16 *)p_a[16]);
  op_mpi_reduce(&arg17, (T17 *)p_a[17]);
  op_mpi_reduce(&arg18, (T18 *)p_a[18]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 19, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if (arg0.idx < -1) {
    free(p_a[0]);
  }
  if (arg1.idx < -1) {
    free(p_a[1]);
  }
  if (arg2.idx < -1) {
    free(p_a[2]);
  }
  if (arg3.idx < -1) {
    free(p_a[3]);
  }
  if (arg4.idx < -1) {
    free(p_a[4]);
  }
  if (arg5.idx < -1) {
    free(p_a[5]);
  }
  if (arg6.idx < -1) {
    free(p_a[6]);
  }
  if (arg7.idx < -1) {
    free(p_a[7]);
  }
  if (arg8.idx < -1) {
    free(p_a[8]);
  }
  if (arg9.idx < -1) {
    free(p_a[9]);
  }
  if (arg10.idx < -1) {
    free(p_a[10]);
  }
  if (arg11.idx < -1) {
    free(p_a[11]);
  }
  if (arg12.idx < -1) {
    free(p_a[12]);
  }
  if (arg13.idx < -1) {
    free(p_a[13]);
  }
  if (arg14.idx < -1) {
    free(p_a[14]);
  }
  if (arg15.idx < -1) {
    free(p_a[15]);
  }
  if (arg16.idx < -1) {
    free(p_a[16]);
  }
  if (arg17.idx < -1) {
    free(p_a[17]);
  }
  if (arg18.idx < -1) {
    free(p_a[18]);
  }
}
//
// op_par_loop routine for 20 arguments
//
template <class T0, class T1, class T2, class T3, class T4, class T5, class T6,
          class T7, class T8, class T9, class T10, class T11, class T12,
          class T13, class T14, class T15, class T16, class T17, class T18,
          class T19>
void op_par_loop(void (*kernel)(T0 *, T1 *, T2 *, T3 *, T4 *, T5 *, T6 *, T7 *,
                                T8 *, T9 *, T10 *, T11 *, T12 *, T13 *, T14 *,
                                T15 *, T16 *, T17 *, T18 *, T19 *),
                 char const *name, op_set set, op_arg arg0, op_arg arg1,
                 op_arg arg2, op_arg arg3, op_arg arg4, op_arg arg5,
                 op_arg arg6, op_arg arg7, op_arg arg8, op_arg arg9,
                 op_arg arg10, op_arg arg11, op_arg arg12, op_arg arg13,
                 op_arg arg14, op_arg arg15, op_arg arg16, op_arg arg17,
                 op_arg arg18, op_arg arg19) {

  char *p_a[20] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  op_arg args[20] = {arg0,  arg1,  arg2,  arg3,  arg4,  arg5,  arg6,
                     arg7,  arg8,  arg9,  arg10, arg11, arg12, arg13,
                     arg14, arg15, arg16, arg17, arg18, arg19};
  if (arg0.idx < -1) {
    p_a[0] = (char *)op_malloc(-1 * args[0].idx * sizeof(T0));
  }
  if (arg1.idx < -1) {
    p_a[1] = (char *)op_malloc(-1 * args[1].idx * sizeof(T1));
  }
  if (arg2.idx < -1) {
    p_a[2] = (char *)op_malloc(-1 * args[2].idx * sizeof(T2));
  }
  if (arg3.idx < -1) {
    p_a[3] = (char *)op_malloc(-1 * args[3].idx * sizeof(T3));
  }
  if (arg4.idx < -1) {
    p_a[4] = (char *)op_malloc(-1 * args[4].idx * sizeof(T4));
  }
  if (arg5.idx < -1) {
    p_a[5] = (char *)op_malloc(-1 * args[5].idx * sizeof(T5));
  }
  if (arg6.idx < -1) {
    p_a[6] = (char *)op_malloc(-1 * args[6].idx * sizeof(T6));
  }
  if (arg7.idx < -1) {
    p_a[7] = (char *)op_malloc(-1 * args[7].idx * sizeof(T7));
  }
  if (arg8.idx < -1) {
    p_a[8] = (char *)op_malloc(-1 * args[8].idx * sizeof(T8));
  }
  if (arg9.idx < -1) {
    p_a[9] = (char *)op_malloc(-1 * args[9].idx * sizeof(T9));
  }
  if (arg10.idx < -1) {
    p_a[10] = (char *)op_malloc(-1 * args[10].idx * sizeof(T10));
  }
  if (arg11.idx < -1) {
    p_a[11] = (char *)op_malloc(-1 * args[11].idx * sizeof(T11));
  }
  if (arg12.idx < -1) {
    p_a[12] = (char *)op_malloc(-1 * args[12].idx * sizeof(T12));
  }
  if (arg13.idx < -1) {
    p_a[13] = (char *)op_malloc(-1 * args[13].idx * sizeof(T13));
  }
  if (arg14.idx < -1) {
    p_a[14] = (char *)op_malloc(-1 * args[14].idx * sizeof(T14));
  }
  if (arg15.idx < -1) {
    p_a[15] = (char *)op_malloc(-1 * args[15].idx * sizeof(T15));
  }
  if (arg16.idx < -1) {
    p_a[16] = (char *)op_malloc(-1 * args[16].idx * sizeof(T16));
  }
  if (arg17.idx < -1) {
    p_a[17] = (char *)op_malloc(-1 * args[17].idx * sizeof(T17));
  }
  if (arg18.idx < -1) {
    p_a[18] = (char *)op_malloc(-1 * args[18].idx * sizeof(T18));
  }
  if (arg19.idx < -1) {
    p_a[19] = (char *)op_malloc(-1 * args[19].idx * sizeof(T19));
  }

  // allocate scratch mememory to do double counting in indirect reduction
  for (int i = 0; i < 20; i++)
    if (args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size) {
      blank_args_size = args[i].size;
      blank_args = (char *)op_malloc(blank_args_size);
    }
  // consistency checks
  int ninds = 0;
  if (OP_diags > 0)
    op_args_check(set, 20, args, &ninds, name);

  if (OP_diags > 2) {
    if (ninds == 0)
      printf(" kernel routine w/o indirection:  %s\n", name);
    else
      printf(" kernel routine with indirection: %s\n", name);
  }
  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  // MPI halo exchange and dirty bit setting, if needed
  int n_upper = op_mpi_halo_exchanges(set, 20, args);

  // loop over set elements
  int halo = 0;

  for (int n = 0; n < n_upper; n++) {
    if (n == set->core_size)
      op_mpi_wait_all(20, args);
    if (n == set->size)
      halo = 1;
    if (args[0].idx < -1)
      op_arg_copy_in(n, args[0], (char **)p_a[0]);
    else
      op_arg_set(n, args[0], &p_a[0], halo);
    if (args[1].idx < -1)
      op_arg_copy_in(n, args[1], (char **)p_a[1]);
    else
      op_arg_set(n, args[1], &p_a[1], halo);
    if (args[2].idx < -1)
      op_arg_copy_in(n, args[2], (char **)p_a[2]);
    else
      op_arg_set(n, args[2], &p_a[2], halo);
    if (args[3].idx < -1)
      op_arg_copy_in(n, args[3], (char **)p_a[3]);
    else
      op_arg_set(n, args[3], &p_a[3], halo);
    if (args[4].idx < -1)
      op_arg_copy_in(n, args[4], (char **)p_a[4]);
    else
      op_arg_set(n, args[4], &p_a[4], halo);
    if (args[5].idx < -1)
      op_arg_copy_in(n, args[5], (char **)p_a[5]);
    else
      op_arg_set(n, args[5], &p_a[5], halo);
    if (args[6].idx < -1)
      op_arg_copy_in(n, args[6], (char **)p_a[6]);
    else
      op_arg_set(n, args[6], &p_a[6], halo);
    if (args[7].idx < -1)
      op_arg_copy_in(n, args[7], (char **)p_a[7]);
    else
      op_arg_set(n, args[7], &p_a[7], halo);
    if (args[8].idx < -1)
      op_arg_copy_in(n, args[8], (char **)p_a[8]);
    else
      op_arg_set(n, args[8], &p_a[8], halo);
    if (args[9].idx < -1)
      op_arg_copy_in(n, args[9], (char **)p_a[9]);
    else
      op_arg_set(n, args[9], &p_a[9], halo);
    if (args[10].idx < -1)
      op_arg_copy_in(n, args[10], (char **)p_a[10]);
    else
      op_arg_set(n, args[10], &p_a[10], halo);
    if (args[11].idx < -1)
      op_arg_copy_in(n, args[11], (char **)p_a[11]);
    else
      op_arg_set(n, args[11], &p_a[11], halo);
    if (args[12].idx < -1)
      op_arg_copy_in(n, args[12], (char **)p_a[12]);
    else
      op_arg_set(n, args[12], &p_a[12], halo);
    if (args[13].idx < -1)
      op_arg_copy_in(n, args[13], (char **)p_a[13]);
    else
      op_arg_set(n, args[13], &p_a[13], halo);
    if (args[14].idx < -1)
      op_arg_copy_in(n, args[14], (char **)p_a[14]);
    else
      op_arg_set(n, args[14], &p_a[14], halo);
    if (args[15].idx < -1)
      op_arg_copy_in(n, args[15], (char **)p_a[15]);
    else
      op_arg_set(n, args[15], &p_a[15], halo);
    if (args[16].idx < -1)
      op_arg_copy_in(n, args[16], (char **)p_a[16]);
    else
      op_arg_set(n, args[16], &p_a[16], halo);
    if (args[17].idx < -1)
      op_arg_copy_in(n, args[17], (char **)p_a[17]);
    else
      op_arg_set(n, args[17], &p_a[17], halo);
    if (args[18].idx < -1)
      op_arg_copy_in(n, args[18], (char **)p_a[18]);
    else
      op_arg_set(n, args[18], &p_a[18], halo);
    if (args[19].idx < -1)
      op_arg_copy_in(n, args[19], (char **)p_a[19]);
    else
      op_arg_set(n, args[19], &p_a[19], halo);

    kernel((T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3], (T4 *)p_a[4],
           (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7], (T8 *)p_a[8], (T9 *)p_a[9],
           (T10 *)p_a[10], (T11 *)p_a[11], (T12 *)p_a[12], (T13 *)p_a[13],
           (T14 *)p_a[14], (T15 *)p_a[15], (T16 *)p_a[16], (T17 *)p_a[17],
           (T18 *)p_a[18], (T19 *)p_a[19]);
  }
  if (n_upper == set->core_size || n_upper == 0)
    op_mpi_wait_all(20, args);

  // set dirty bit on datasets touched
  op_mpi_set_dirtybit(20, args);

  // global reduction for MPI execution, if needed
  // p_a simply used to determine type for MPI reduction
  op_mpi_reduce(&arg0, (T0 *)p_a[0]);
  op_mpi_reduce(&arg1, (T1 *)p_a[1]);
  op_mpi_reduce(&arg2, (T2 *)p_a[2]);
  op_mpi_reduce(&arg3, (T3 *)p_a[3]);
  op_mpi_reduce(&arg4, (T4 *)p_a[4]);
  op_mpi_reduce(&arg5, (T5 *)p_a[5]);
  op_mpi_reduce(&arg6, (T6 *)p_a[6]);
  op_mpi_reduce(&arg7, (T7 *)p_a[7]);
  op_mpi_reduce(&arg8, (T8 *)p_a[8]);
  op_mpi_reduce(&arg9, (T9 *)p_a[9]);
  op_mpi_reduce(&arg10, (T10 *)p_a[10]);
  op_mpi_reduce(&arg11, (T11 *)p_a[11]);
  op_mpi_reduce(&arg12, (T12 *)p_a[12]);
  op_mpi_reduce(&arg13, (T13 *)p_a[13]);
  op_mpi_reduce(&arg14, (T14 *)p_a[14]);
  op_mpi_reduce(&arg15, (T15 *)p_a[15]);
  op_mpi_reduce(&arg16, (T16 *)p_a[16]);
  op_mpi_reduce(&arg17, (T17 *)p_a[17]);
  op_mpi_reduce(&arg18, (T18 *)p_a[18]);
  op_mpi_reduce(&arg19, (T19 *)p_a[19]);

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 20, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif

  if (arg0.idx < -1) {
    free(p_a[0]);
  }
  if (arg1.idx < -1) {
    free(p_a[1]);
  }
  if (arg2.idx < -1) {
    free(p_a[2]);
  }
  if (arg3.idx < -1) {
    free(p_a[3]);
  }
  if (arg4.idx < -1) {
    free(p_a[4]);
  }
  if (arg5.idx < -1) {
    free(p_a[5]);
  }
  if (arg6.idx < -1) {
    free(p_a[6]);
  }
  if (arg7.idx < -1) {
    free(p_a[7]);
  }
  if (arg8.idx < -1) {
    free(p_a[8]);
  }
  if (arg9.idx < -1) {
    free(p_a[9]);
  }
  if (arg10.idx < -1) {
    free(p_a[10]);
  }
  if (arg11.idx < -1) {
    free(p_a[11]);
  }
  if (arg12.idx < -1) {
    free(p_a[12]);
  }
  if (arg13.idx < -1) {
    free(p_a[13]);
  }
  if (arg14.idx < -1) {
    free(p_a[14]);
  }
  if (arg15.idx < -1) {
    free(p_a[15]);
  }
  if (arg16.idx < -1) {
    free(p_a[16]);
  }
  if (arg17.idx < -1) {
    free(p_a[17]);
  }
  if (arg18.idx < -1) {
    free(p_a[18]);
  }
  if (arg19.idx < -1) {
    free(p_a[19]);
  }
}
