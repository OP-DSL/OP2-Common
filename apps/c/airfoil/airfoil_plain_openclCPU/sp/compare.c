#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>

//#ifndef FPPREC
//# error "Error: define FPPREC!"
//#endif
//
//#if FPPREC == 0
#  define FP float
//#elif FPPREC == 1
//#  define FP double
//#else
//#  error "Macro definition FPPREC unrecognized for CUDA"
//#endif
//
//extern char *optarg;
//extern int  optind, opterr, optopt; 
//static struct option options[] = {
//  {"nx",   required_argument, 0,  0   },
//  {"ny",   required_argument, 0,  0   },
//  {"nz",   required_argument, 0,  0   },
//  {"help", no_argument,       0,  'h' },
//  {0,      0,                 0,  0   }
//};
//
///*
// * Print essential infromation on the use of the program
// */
//void print_help() {
//  printf("\nPlease specify the ADI configuration "
//    "e.g. ./compare file1.dat file2.dat -nx NX -ny NY -nz NZ \n");
//}


int main(int argc, char** argv) {
  // Process arguments
//  int  nx=256;
//  int  ny=256;
//  int  nz=256;
  char filename1[256];
  char filename2[256];


  // Get program arguments
  strcpy(filename1, argv[1]);
  strcpy(filename2, argv[2]);

  // Declare stuff
  FILE *fin1, *fin2;
  FP   *q1, *q2, *diff, *rel_diff;

  int ncell = 720000;
  int dim   = 4;


  int  size = ncell*dim;

  q1     = (FP*) calloc(size, sizeof(FP));
  q2     = (FP*) calloc(size, sizeof(FP));
  diff     = (FP*) calloc(size, sizeof(FP));
  rel_diff = (FP*) calloc(size, sizeof(FP));

  // Open files
  printf("Opening file: %s \n", filename1);
  fin1 = fopen(filename1,"r");
  printf("Opening file: %s \n", filename2);
  fin2 = fopen(filename2,"r");

  // Read files
  if(fread(q1,sizeof(FP),size,fin1) != fread(q2,sizeof(FP),size,fin2)) 
    printf("There was an error while reading the fileis!\n");
//  for (k=0; k<nz; k++) {
//    for (j=0; j<ny; j++) {
//      for (i=0; i<nx; i++) {
//        ind = i + j*nx + k*nx*ny;
//        fprintf(fout, " %5.20e ", h_u[ind]);
//      }
//    }
//  }
  int i,j,k,ind;
  int count = 0;
  FP *sum = (FP*) malloc(dim*sizeof(FP));
  for(i=0; i<dim; i++) sum[i] = 0.0f;
  FP rel_diff_reg;
  for (j=0; j<ncell; j++) {
    for (i=0; i<dim; i++) {
      ind = i + j*dim;
      diff[ind] = q2[ind]-q1[ind];
      //if(diff[ind] != 0.0) {
      //  printf(" %e %e \n", q1[ind], q2[ind]);
      //}
      if(isnan(diff[ind])) {
        printf("%d %d %d\n",i,j,k);
      }
      rel_diff_reg = diff[ind] / q1[ind];
      rel_diff[ind] = isnan(rel_diff_reg) ? 0 : rel_diff_reg;   
      //if(rel_diff[ind] > 1e-5) {
      //  count++;
      //  printf("\nRelative error %g exceeded error tolerance (1e-6) %d times! \n", rel_diff[ind], count);
      //}
      sum[i] += diff[ind];
    }
    //printf("\n");
  }
  //printf("\nSumOfDiff = [%e, %e, %e, %e]; Normalized SumOfDiff = [%e, %e, %e, %e] \n", sum[0], sum[1], sum[2], sum[3], sum[0]/(FP)(ncell*dim));
  printf("\n SumOfDiff =            [%e, %e, %e, %e]; \n \
Normalized SumOfDiff = [%e, %e, %e, %e] \n", sum[0], sum[1], sum[2], sum[3], sum[0]/(FP)ncell, sum[1]/(FP)ncell, sum[2]/(FP)ncell, sum[3]/(FP)ncell);
  fclose(fin1);
  fclose(fin2);
  free(q1);
  free(q2);
  free(diff);
  free(rel_diff);
  free(sum);
}
