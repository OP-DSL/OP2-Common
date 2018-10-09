#ifndef __OP_SEQ_H
#define __OP_SEQ_H
//
// header for sequential and MPI+sequentional execution
//

#include "op_lib_cpp.h"
#include <utility>

static int op2_stride = 1;
#define OP2_STRIDE(arr, idx) arr[idx]

#if _cpluspluc >= 201402L
// from c++14
typedef indices std::index_sequence;
typedef build_indices std::make_index_sequence;
#else
template <size_t... Is> struct indices {};

template <size_t N, size_t... Is>
struct build_indices : public build_indices<N - 1, N - 1, Is...> {};

template <size_t... Is> struct build_indices<0, Is...> : indices<Is...> {};
#endif

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
// op_par_loop routine implementation with index sequence
//
template <typename... T, typename... OPARG, size_t... I>
void op_par_loop_impl(indices<I...>, void (*kernel)(T *...), char const *name,
                      op_set set, OPARG... arguments) {
  constexpr int N = sizeof...(OPARG);
  char *p_a[N] = {((arguments.idx < -1)
                       ? (char *)malloc(-1 * arguments.idx * sizeof(T))
                       : nullptr)...};
  op_arg args[N] = {arguments...};
  // allocate scratch mememory to do double counting in indirect reduction
  (void)std::initializer_list<char *>{
      ((arguments.argtype == OP_ARG_GBL && arguments.size > blank_args_size)
           ? (blank_args_size = arguments.size,
              blank_args = (char *)op_malloc(blank_args_size))
           : nullptr)...};
  // consistency checks
  int ninds = 0;
  if (OP_diags > 0)
    op_args_check(set, N, args, &ninds, name);

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
  int n_upper = op_mpi_halo_exchanges(set, N, args);
  // loop over set elements
  int halo = 0;

  for (int n = 0; n < n_upper; n++) {
    if (n == set->core_size)
      op_mpi_wait_all(20, args);
    if (n == set->size)
      halo = 1;
    (void)std::initializer_list<int>{
        (arguments.idx < -1 ? (op_arg_copy_in(n, arguments, (char **)p_a[I]), 0)
                            : (op_arg_set(n, arguments, &p_a[I], halo), 0))...};
    kernel(((T *)p_a[I])...);
  }
  if (n_upper == set->core_size || n_upper == 0)
    op_mpi_wait_all(N, args);

  // set dirty bit on datasets touched
  op_mpi_set_dirtybit(N, args);

  // global reduction for MPI execution, if needed
  // p_a simply used to determine type for MPI reduction
  (void)std::initializer_list<int>{
      (op_mpi_reduce(&arguments, (T *)p_a[I]), 0)...};

  // update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  void *k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  op_mpi_perf_comms(k_i, 20, args);
#else
  op_mpi_perf_time(name, wall_t2 - wall_t1);
#endif
  (void)std::initializer_list<int>{
      (arguments.idx < -1 ? free(p_a[I]), 0 : 0)...};
}

//
// op_par_loop routine wrapper to create index sequence
//
template <typename... T, typename... OPARG>
void op_par_loop(void (*kernel)(T *...), char const *name, op_set set,
                 OPARG... arguments) {
  op_par_loop_impl(build_indices<sizeof...(T)>{}, kernel, name, set,
                   arguments...);
}
#endif /* __OP_SEQ_H */
