/* Simple binary file value comparison programe for comparing results 
   from the airfoil code
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


int main(int argc, char *argv[]){
    
  /* read in file1 from disk*/
  FILE *fp1, *fp2;
  int lines1, elem_dim1, lines2, elem_dim2;
  double epsi = 0.00000000000001;//(10^-14)
  int differ = 0;
  
  
  if(argc < 3)
  {
      printf("Usage: ./comparebin file1 file2\n");
      exit(-1);
  }
  
  
  if ( (fp1 = fopen(argv[1],"rb")) == NULL) {
    printf("can't open file %s\n",argv[1]); exit(-1);
  }
  
  fread(&lines1, sizeof(int), 1, fp1);
  fread(&elem_dim1, sizeof(int), 1, fp1);
     
  if ( (fp2 = fopen(argv[2],"rb")) == NULL) {
    printf("can't open file %s\n",argv[2]); exit(-1);
  }
  
  fread(&lines2, sizeof(int), 1, fp2);
  fread(&elem_dim2, sizeof(int), 1, fp2);
      
  printf("File 1 lines %d, File2 lines %d\n",lines1, lines2);
  
  if(lines1 != lines2 || elem_dim1 != elem_dim2)
  {
      printf("File mismatch: number of lines or element dimensions not matching\n"); exit(-1);
  }
  
  double values1[elem_dim1];
  double values2[elem_dim2];
  
  for (int n = 0; n <lines1; n++){
      
      for(int d = 0; d < elem_dim1; d++)
      {
      	  fread(&values1[d],sizeof(double),1,fp1);fread(&values2[d],sizeof(double),1,fp2);
      	  //printf("%f %f\n",values1[d],values2[d]);
      	  if(fabs(values1[d]-values2[d])>epsi)
      	  {
      	      printf("File mismatch: at line %d element %d\n", n+2, d+1); 
      	      differ = 1;
      	  }
      	  
      }//printf("\n");
      //fscanf(fp1,"\n");fscanf(fp2,"\n");
      
  }

  fclose(fp1);  fclose(fp2);
  
  if(differ == 0)
      printf("Files Identical\n");
  else
      printf("Files Differ\n");
	    
  
  
}

