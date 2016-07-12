#ifndef _CHECK_RESULT_H
#define _CHECK_RESULT_H

#ifndef STRIDE
#define STRIDE 1
#endif

#include "op_lib_c.h"

template <class T> inline int check_value(int i, int j, T val, T ref, T tol) {
  if (fabs(val - ref) > tol) {
    op_printf("Failure: i=%d, j=%d, expected: %f, actual: %f\n", i, j, ref,
              val);
    return 1;
  }
  return 0;
}

template <class T> int check_result(T *u, int nn, T tol) {
  // Check results against reference u solution. This is correct for
  // variations in nn but not NITER

  int failed = 0;
  for (int i = 1; i < nn; i++) {
    for (int j = 1; j < nn; j++) {
      int n = STRIDE * (i - 1 + (j - 1) * (nn - 1));
      if (((i == 1) && (j == 1)) || ((i == 1) && (j == (nn - 1))) ||
          ((i == (nn - 1)) && (j == 1)) ||
          ((i == (nn - 1)) && (j == (nn - 1)))) {
        // Corners of domain
        failed = check_value(i, j, u[n], (T)0.625, tol);
      } else if ((i == 1 && j == 2) || (i == 2 && j == 1) ||
                 (i == 1 && j == (nn - 2)) || (i == 2 && j == (nn - 1)) ||
                 (i == (nn - 2) && j == 1) || (i == (nn - 1) && j == 2) ||
                 (i == (nn - 2) && j == (nn - 1)) ||
                 (i == (nn - 1) && j == (nn - 2))) {
        // Horizontally or vertically-adjacent to a corner
        failed = check_value(i, j, u[n], (T)0.4375, tol);
      } else if ((i == 2 && j == 2) || (i == 2 && j == (nn - 2)) ||
                 (i == (nn - 2) && j == 2) ||
                 (i == (nn - 2) && j == (nn - 2))) {
        // Diagonally adjacent to a corner
        failed = check_value(i, j, u[n], (T)0.125, tol);
      } else if ((i == 1) || (j == 1) || (i == (nn - 1)) || (j == (nn - 1))) {
        // On some other edge node
        failed = check_value(i, j, u[n], (T)0.3750, tol);
      } else if ((i == 2) || (j == 2) || (i == (nn - 2)) || (j == (nn - 2))) {
        // On some other node that is 1 node from the edge
        failed = check_value(i, j, u[n], (T)0.0625, tol);
      } else {
        // 2 or more nodes from the edge
        failed = check_value(i, j, u[n], (T)0.0, tol);
      }
    }
  }

  if (!failed)
    op_printf("\nResults check PASSED!\n");

  return failed;
}

#endif
