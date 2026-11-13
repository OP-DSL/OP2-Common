// An untranslated loop's way into a matrix argument (op_seq.h), which knows
// op_arg but not the matrix storage contract (op2/rt/abi/matrix.h).

#include <op2/rt/abi/matrix.h>
#include <op_lib_core.h>

extern "C" char *op_mat_untranslated(op_arg *arg) { return arg->mat->untranslated(arg->mat, arg->acc); }
