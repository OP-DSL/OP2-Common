#!/usr/bin/python
#######################################################################
#                                                                     #
#       This Python routine generates the header file op_seq.h        #
#                                                                     #
#######################################################################


#
# this sets the max number of arguments in op_par_loop
#
maxargs = 10

#open/create file
f = open('./op_seq.h','w')

#
#first the top bit
#

top =  """
//                                                                 
// header for sequential and MPI+sequentional execution            
//                                                                 
                                                                   
#include "op_lib_cpp.h"                                            
                                                                   
int op2_stride = 1;                                                
#define OP2_STRIDE(arr, idx) arr[idx]                              
                                                                   
// scratch space to use for double counting in indirect reduction  
int blank_args_size = 512;                                         
char* blank_args = (char *)malloc(blank_args_size);                
                                                                   
inline void op_arg_set(int n, op_arg arg, char **p_arg, int halo){ 
  *p_arg = arg.data;                                               
                                                                   
  if (arg.argtype==OP_ARG_GBL) {                                   
    if (halo && (arg.acc != OP_READ)) *p_arg = blank_args;         
  }                                                                
  else {                                                           
    if (arg.map==NULL)         // identity mapping                 
      *p_arg += arg.size*n;                                        
    else                       // standard pointers                
      *p_arg += arg.size*arg.map->map[arg.idx+n*arg.map->dim];     
  }                                                                
}                                                                  
                                                                   
inline void op_arg_copy_in(int n, op_arg arg, char **p_arg) {
  for (int i = 0; i < -1*arg.idx; ++i)
    p_arg[i] = arg.data + arg.map->map[i+n*arg.map->dim]*arg.size;
}                                                                  
                                                                   
inline void op_args_check(op_set set, int nargs, op_arg *args,     
                                      int *ninds, const char *name) {    
  for (int n=0; n<nargs; n++)                                      
    op_arg_check(set,n,args[n],ninds,name);                        
}                                                                  
                                                                   
"""

f.write(top)

#
# now for op_par_loop defns
#

for nargs in range (1,maxargs+1):
    f.write('//\n')
    f.write('//op_par_loop routine for '+str(nargs)+' arguments\n')
    f.write('//\n')
    
    n_per_line = 4
    
    f.write('template <')
    for n in range (0, nargs):
        f.write('class T'+str(n))  
        if nargs <> 1 and n != nargs-1:
          f.write(',')
        else:
          f.write('>\n')
        if n%n_per_line == 3 and n <> nargs-1:
          f.write('\n          ')
    
    f.write('void op_par_loop(void (*kernel)(')
    for n in range (0, nargs):
        f.write('T'+str(n)+'*')  
        if nargs <> 1 and n != nargs-1:
          f.write(', ')
        else:
          f.write('),\n')
        if n%n_per_line == 3 and n <> nargs-1:
         f.write('\n                                ')    
    
    f.write('    char const * name, op_set set,\n    ')
    for n in range (0, nargs):
        f.write('op_arg arg'+str(n))  
        if nargs <> 1 and n != nargs-1:
          f.write(', ')
        else:
          f.write('){\n')
        if n%n_per_line == 3 and n <> nargs-1:
         f.write('\n    ')

    f.write('\n  char *p_a['+str(nargs)+'] = {')
    for n in range (0, nargs):
        f.write('0')  
        if nargs <> 1 and n != nargs-1:
          f.write(',')
        else:
          f.write('};\n')

    f.write('  op_arg args['+str(nargs)+'] = {')
    for n in range (0, nargs):
        f.write('arg'+str(n))  
        if nargs <> 1 and n != nargs-1:
          f.write(', ')
        else:
          f.write('};\n')
        if n%n_per_line == 3 and n <> nargs-1:
          f.write('\n                    ')

    for n in range (0, nargs):
        f.write('  if(arg'+str(n)+'.idx < -1) {\n')
        f.write('    p_a['+str(n)+'] = (char *)malloc(-1*args['+str(n)+'].idx*sizeof(T'+str(n)+'));\n  }\n')
     
    f.write('\n  //allocate scratch mememory to do double counting in indirect reduction\n')
    f.write('  for (int i = 0; i<'+str(nargs)+';i++)\n')
    f.write('    if(args[i].argtype == OP_ARG_GBL && args[i].size > blank_args_size )\n')
    f.write('    {\n')
    f.write('      blank_args_size = args[i].size;\n');
    f.write('      blank_args = (char *)malloc(blank_args_size);\n')
    f.write('    }\n')
    
    f.write('  // consistency checks\n')
    f.write('  int ninds = 0;\n')

    f.write('  if (OP_diags>0) op_args_check(set,'+str(nargs)+',args,&ninds,name);\n\n')

    f.write('  if (OP_diags>2) {\n')
    f.write('  if (ninds==0)\n')
    f.write('    printf(" kernel routine w/o indirection:  %s\\n",name);\n')
    f.write('  else\n')
    f.write('    printf(" kernel routine with indirection: %s\\n",name);\n')
    f.write('  }\n')

    f.write('  // initialise timers\n')
    f.write('  double cpu_t1, cpu_t2, wall_t1, wall_t2;\n')
    f.write('  op_timers_core(&cpu_t1, &wall_t1);\n\n')

    f.write('  // MPI halo exchange and dirty bit setting, if needed\n')
    f.write('  int n_upper = op_mpi_halo_exchanges(set, '+str(nargs)+', args);\n\n')
    f.write('  // loop over set elements\n')
    f.write('  int halo = 0; \n\n')   
     
    f.write('  for (int n=0; n<n_upper; n++) {\n')
    f.write('    if (n==set->core_size) op_mpi_wait_all('+str(nargs)+',args);\n')
    f.write('    if (n==set->size) halo = 1;\n')
    
    for n in range (0, nargs):
        f.write('  if (args['+str(n)+'].idx < -1) op_arg_copy_in(n,args['+str(n)+'], (char **)p_a['+str(n)+']);\n')
        f.write('  else op_arg_set(n,args['+str(n)+'], &p_a['+str(n)+'],halo);\n')
    
    f.write('\n  kernel( ')
    for n in range (0, nargs):
        f.write('(T'+str(n)+' *)p_a['+str(n)+']')  
        if nargs <> 1 and n != nargs-1:
          f.write(', ')
        else:
          f.write(');\n')
        if n%n_per_line == 3 and n <> nargs-1:
          f.write('\n          ')
    
    
    f.write('  }\n')

    f.write('  //set dirty bit on datasets touched\n')
    f.write('  op_mpi_set_dirtybit(10, args);\n\n')

    f.write('  //global reduction for MPI execution, if needed\n')
    for n in range (0, nargs):
        f.write('  op_mpi_reduce(&arg'+str(n)+',(T'+str(n)+' *)p_a['+str(n)+']);\n')
    
    f.write('\n  // update timer record\n')
    f.write('  op_timers_core(&cpu_t2, &wall_t2);\n')
    f.write('#ifdef COMM_PERF\n')
    f.write('  int k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);\n')
    f.write('  op_mpi_perf_comms(k_i, '+str(nargs)+', args);\n')
    f.write('#else\n')
    f.write('  op_mpi_perf_time(name, wall_t2 - wall_t1);\n')
    f.write('#endif\n\n')
    
    for n in range (0, nargs):
        f.write('  if(arg'+str(n)+'.idx < -1) {\n')
        f.write('    free(p_a['+str(n)+']);\n');
        f.write('  }\n');
    
    
    f.write('}\n')

f.close()
