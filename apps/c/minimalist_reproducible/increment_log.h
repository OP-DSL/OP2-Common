inline void increment_log(double* edge, double* n1, double* n2){
    *n1=*edge * log(*n1);    
    *n2=*edge * log(*n2);    
}
