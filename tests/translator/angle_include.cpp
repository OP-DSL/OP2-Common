// A program that includes op_seq.h with angle brackets.

#include <op_seq.h>

void kernel(double *q) { q[0] = 1.0; }

int main(int argc, char **argv) {
    op_init(argc, argv, 2);
    op_set cells = op_decl_set(4, "cells");
    op_dat q = op_decl_dat(cells, 1, "double", (double *)nullptr, "q");
    op_par_loop(kernel, "kernel", cells, op_arg_dat(q, -1, OP_ID, 1, "double", OP_WRITE));
    op_exit();
}
