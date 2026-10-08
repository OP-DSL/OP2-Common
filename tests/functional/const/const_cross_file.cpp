#include "op_seq.h"

// my_const1 is declared and registered with op_decl_const in const_tests.cpp.
// Keeping this loop in a separate translation unit verifies that the
// translator resolves constants application-wide, rather than per source file.
extern double my_const1;

void consts_cross_file(double *dat) {
  *dat = my_const1;
}

void run_cross_file_const(op_set set, op_dat dat) {
  op_par_loop(consts_cross_file, "consts_cross_file", set,
              op_arg_dat(dat, -1, OP_ID, 1, "double", OP_WRITE));
}
