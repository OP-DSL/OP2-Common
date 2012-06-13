
#ifndef _CHECK_RESULT_H
#define _CHECK_RESULT_H

#include "op_lib_c.h"

template<class T>
int check_result(T* u, int nn, T tol)
{
  int nnode = (nn-1)*(nn-1);
  T *ur = (T*)malloc(sizeof(double)*2*nnode);

  // create reference u solution. This is correct for variations in nn
  // but not NITER

  for (int i=1; i<nn; i++) {
    for (int j=1; j<nn; j++) {
      int n = 2*((i-1) + (j-1)*(nn-1));

      if ( ((i==1) && (j==1))      || ((i==1) && (j==(nn-1))) ||
           ((i==(nn-1)) && (j==1)) || ((i==(nn-1)) && (j==(nn-1))) ) {
        // Corners of domain
        ur[n] = 0.625;
      }
      else if ( (i==1 && j==2)           || (i==2 && j==1)      ||
                (i==1 && j==(nn-2))      || (i==2 && j==(nn-1)) ||
                (i==(nn-2) && j==1)      || (i==(nn-1) && j==2) ||
                (i==(nn-2) && j==(nn-1)) || (i==(nn-1) && j==(nn-2)) ) {
        // Horizontally or vertically-adjacent to a corner
        ur[n] = 0.4375;
      }
      else if ( (i==2 && j==2)      || (i==2 && j==(nn-2)) ||
                (i==(nn-2) && j==2) || (i==(nn-2) && j==(nn-2)) ) {
        // Diagonally adjacent to a corner
        ur[n] = 0.125;
      }
      else if ( (i==1) || (j==1) || (i==(nn-1)) || (j==(nn-1)) ) {
        // On some other edge node
        ur[n] = 0.3750;
      }
      else if ( (i==2) || (j==2) || (i==(nn-2)) || (j==(nn-2)) ) {
        // On some other node that is 1 node from the edge
        ur[n] = 0.0625;
      }
      else {
        // 2 or more nodes from the edge
        ur[n] = 0.0f;
      }
    }
  }

  // Check results

  int failed = 0;
  for (int j=nn-1; j>0; j--) {
    for (int i=1; i<nn; i++) {
        int n = 2*(i-1 + (j-1)*(nn-1));
        if (fabs(u[n] - ur[n]) > tol) {
          failed = 1;
          op_printf("Failure: i=%d, j=%d, expected: %f, actual: %f\n", i, j, ur[n], u[n]);
        }
    }
  }

  if (!failed)
    op_printf("\nResults check passed!\n");

  return failed;
}

#endif
