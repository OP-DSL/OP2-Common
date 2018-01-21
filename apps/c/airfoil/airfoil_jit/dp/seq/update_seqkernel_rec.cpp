#include "op_lib_cpp.h"

// global constants - values #defined by JIT
#include "jit_const.h"

// user function
#include "../update.h"

// user function
extern "C" {
void op_par_loop_update_rec_execute(op_kernel_descriptor *desc);

// host stub function
void op_par_loop_update_rec_execute(op_kernel_descriptor *desc) {

  op_set set = desc->set;
  char const *name = desc->name;
  int nargs = 5;

  op_arg arg0 = desc->args[0];
  op_arg arg1 = desc->args[1];
  op_arg arg2 = desc->args[2];
  op_arg arg3 = desc->args[3];
  op_arg arg4 = desc->args[4];

  op_arg args[5] = {arg0, arg1, arg2, arg3, arg4};

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(4);
  op_timers_core(&cpu_t1, &wall_t1);

  if (OP_diags > 2) {
    printf(" kernel routine w/o indirection:  update");
  }

  int set_size = op_mpi_halo_exchanges(set, nargs, args);

  if (set->size > 0) {

    for (int n = 0; n < set_size; n++) {
      update(&((double *)arg0.data)[4 * n], &((double *)arg1.data)[4 * n],
             &((double *)arg2.data)[4 * n], &((double *)arg3.data)[1 * n],
             (double *)arg4.data);
    }
  }

  // combine reduction data
  op_mpi_reduce_double(&arg4, (double *)arg4.data);
  op_mpi_set_dirtybit(nargs, args);

  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[4].name = name;
  OP_kernels[4].count += 1;
  OP_kernels[4].time += wall_t2 - wall_t1;
  OP_kernels[4].transfer += (float)set->size * arg0.size;
  OP_kernels[4].transfer += (float)set->size * arg1.size * 2.0f;
  OP_kernels[4].transfer += (float)set->size * arg2.size * 2.0f;
  OP_kernels[4].transfer += (float)set->size * arg3.size;
}
}