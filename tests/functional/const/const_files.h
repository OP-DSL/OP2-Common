// Constants declared in const_files.cpp and read by a loop in
// const_files_step.cpp, whose kernel const_files.cpp defines.

#pragma once

#include "op_seq.h"

extern double scale_a;
extern double scale_b;

double shift();
void scale_kernel(double *x);
void scale(op_set set, op_dat dat);
