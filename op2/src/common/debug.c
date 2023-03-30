/* returns the number of 1 bits in the character */
int bit_count_char(char byte){

	int sum=0;
	for(int i=0;i<8;i++){
		sum+=(byte&1);
		byte=byte>>1;
	}
	return sum;
}

/* returns the number of 1 bits in the */
int bit_count(void *data, int size){
	char *d = (char*) data;
	int sum=0;
	for(int c=0;c<size;c++){
		sum+=bit_count_char(d[c]);
	}

	return sum;
}


/* xors size many successive bytes in data and returns result */
int xor_byte_summation(void *data, int size){
	char *d = (char*) data;
	char res=d[0];
	for(int i=1;i<size;i++){
		res = res ^ d[i];
	}
	return res;
}
