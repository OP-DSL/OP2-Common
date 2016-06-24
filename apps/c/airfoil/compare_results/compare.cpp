/*
 * compare.c
 *
 * Simple text file (ASCI) value comparison programe for comparing results
 * printed
 * from the airfoil code
 *
 * written by: Gihan R. Mudalige, (Started 04-04-2011)
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {

  /* read in file1 from disk*/
  FILE *fp1, *fp2;
  int lines1, elem_dim1, lines2, elem_dim2;

  /**indicate your tolerance**/
  double epsi = 0.000001;

  int differ = 0;

  if (argc < 3) {
    printf("Usage: ./compare file1 file2\n");
    exit(-1);
  }

  if ((fp1 = fopen(argv[1], "r")) == NULL) {
    printf("can't open file %s\n", argv[1]);
    exit(-1);
  }

  if (fscanf(fp1, "%d %d \n", &lines1, &elem_dim1) < 0) {
    printf("error reading from %s\n", argv[1]);
    exit(-1);
  }

  if ((fp2 = fopen(argv[2], "r")) == NULL) {
    printf("can't open file %s\n", argv[2]);
    exit(-1);
  }

  if (fscanf(fp2, "%d %d \n", &lines2, &elem_dim2) < 0) {
    printf("error reading from %s\n", argv[2]);
    exit(-1);
  }

  printf("File 1 lines %d, File2 lines %d\n", lines1, lines2);

  if (lines1 != lines2 || elem_dim1 != elem_dim2) {
    printf(
        "File mismatch: number of lines or element dimensions not matching\n");
    exit(-1);
  }

  double values1[elem_dim1];
  double values2[elem_dim2];

  for (int n = 0; n < lines1; n++) {

    for (int d = 0; d < elem_dim1; d++) {
      fscanf(fp1, "%lf ", &values1[d]);
      fscanf(fp2, "%lf ", &values2[d]);
      if (fabs(values1[d] - values2[d]) > epsi) {
        printf("File mismatch: at line %d element %d\n", n + 2, d + 1);
        differ = 1;
      }
    }
    fscanf(fp1, "\n");
    fscanf(fp2, "\n");
  }

  fclose(fp1);
  fclose(fp2);

  if (differ == 0)
    printf("Files Identical\n");
  else
    printf("Files Differ\n");
}
