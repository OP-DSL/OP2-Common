
//
// header file (includes op_lib_core.h and various system header files)
//

#include "op_lib.h"
#include "op_lib_core.h"
#include "op_rt_support.h"

//
// routines called by user code and kernels
// these wrappers are used by non-CUDA versions
// op_lib.cu provides wrappers for CUDA version
//

void op_init(int argc, char **argv, int diags){
  op_init_core(argc, argv, diags);
}

op_dat op_decl_dat_char(op_set set, int dim, char const *type,
                        int size, char *data, char const *name){
  return op_decl_dat_core(set, dim, type, size, data, name);
}

void op_fetch_data(op_dat dat) {}
void op_decl_const_char(int, char const*, int, char*, char const*){}

op_plan * op_plan_get(char const *name, op_set set, int part_size,
                      int nargs, op_arg *args, int ninds, int *inds){
  return op_plan_core(name, set, part_size, nargs, args, ninds, inds);
}

void op_exit ()
{
  op_rt_exit ();

  op_exit_core ();
}
