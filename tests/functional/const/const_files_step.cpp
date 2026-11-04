#include "op_seq.h"

#include "const_files.h"

void scale(op_set set, op_dat dat) {
  // The kernel, defined in another file, on a continuation line.
  op_par_loop(
      scale_kernel, "scale", set,
      op_arg_dat(dat, -1, OP_ID, 1, "double", OP_RW));
}
