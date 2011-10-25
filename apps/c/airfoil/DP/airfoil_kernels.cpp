// 
// auto-generated by op2.m on 25-Oct-2011 14:51:27 
//

// header                        
                                 
#include "op_lib_cpp.h"          
#include "op_openmp_rt_support.h"
                                 
void __syncthreads(){}           
                                 
// global constants              
                                 
extern double gam;               
extern double gm1;               
extern double cfl;               
extern double eps;               
extern double mach;              
extern double alpha;             
extern double qinf[4];           
                                 
// user kernel files             
                                 
#include "save_soln_kernel.cpp"  
#include "adt_calc_kernel.cpp"   
#include "res_calc_kernel.cpp"   
#include "bres_calc_kernel.cpp"  
#include "update_kernel.cpp"     
