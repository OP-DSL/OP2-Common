//
// standard headers
//

#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<vector>


//
// OP header file
//

#include "op_seq.h"


//
//gmsh_read - reads in data in a gmsh file (i.e. file with extension .msh)
//


//
// main program
//

int main(int argc, char **argv) {
	
  // OP initialisation
	op_init(argc, argv, 2);
	
	int *cell=NULL, *edge=NULL, *ecell=NULL, *bedge=NULL, *becell=NULL, *bound=NULL;
	double *x=NULL;
	int nnode=0, ncell=0, nedge=0, nbedge=0, niter=0;
	
	
}