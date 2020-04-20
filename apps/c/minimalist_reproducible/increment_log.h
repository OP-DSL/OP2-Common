#include <mpi.h>
inline void increment_log(double* edge, double* n1, double* n2){

  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    *n1=*edge * log(*n1);
    *n2=*edge * log(*n2);
}
