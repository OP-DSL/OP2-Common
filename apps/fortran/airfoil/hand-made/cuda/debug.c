// for copying results in file
#include <stdlib.h>
#include <stdio.h>


// warning: only one file at time!!!
static FILE * myfile;


int openfile ( const char filename[] )
{
		
	myfile = fopen ( filename, "w" );
	
	printf ( "file opened" );
	
	return 0;
	
}


int closefile ( const char filename[] )
{
	
	fclose ( myfile );
	
	printf ( "file closed" );
	
	return 0;
	
}



int writerealtofile ( double * value )

{
	
	fprintf ( myfile, "%lf\n", *value );
	
	return 0;
	
}


int writeinttofile ( int * data, int dataSize, char filename[20] )

{

	int k;
	
	// copy results to output file
	FILE * myfile;
	
	myfile = fopen ( filename, "w" );
	
	
	for ( k = 0; k < dataSize; k++ )
		fwrite ( &data[k], sizeof(int), 1, myfile );
	
	
	fclose ( myfile );
	
	return 0;
	
}
