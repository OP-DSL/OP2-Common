##########################################################################
#
# CUDA code generator
#
# This routine is called by op2 which parses the input files
#
# It produces a file xxx_kernel.cu for each kernel,
# plus a master kernel file
#
##########################################################################

import re
import datetime
import os

def comm(line):
  global file_text, FORTRAN, CPP
  global depth
  prefix = ' '*depth
  if len(line) == 0:
    file_text +='\n'
  elif FORTRAN:
    file_text +='!  '+line+'\n'
  elif CPP:
    file_text +=prefix+'//'+line.rstrip()+'\n'

def code(text):
  global file_text, FORTRAN, CPP, g_m
  global depth
  if text == '':
    prefix = ''
  else:
    prefix = ' '*depth
  file_text += prefix+text.rstrip()+'\n'


def FOR(i,start,finish):
  global file_text, FORTRAN, CPP, g_m
  global depth
  if FORTRAN:
    code('do '+i+' = '+start+', '+finish+'-1')
  elif CPP:
    code('for ( int '+i+'='+start+'; '+i+'<'+finish+'; '+i+'++ ){')
  depth += 2

def FOR_INC(i,start,finish,inc):
  global file_text, FORTRAN, CPP, g_m
  global depth
  if FORTRAN:
    code('do '+i+' = '+start+', '+finish+'-1')
  elif CPP:
    code('for ( int '+i+'='+start+'; '+i+'<'+finish+'; '+i+'+='+inc+' ){')
  depth += 2

def ENDFOR():
  global file_text, FORTRAN, CPP, g_m
  global depth
  depth -= 2
  if FORTRAN:
    code('enddo')
  elif CPP:
    code('}')

def IF(line):
  global file_text, FORTRAN, CPP, g_m
  global depth
  if FORTRAN:
    code('if ('+line+') then')
  elif CPP:
    code('if ('+ line + ') {')
  depth += 2

def ENDIF():
  global file_text, FORTRAN, CPP, g_m
  global depth
  depth -= 2
  if FORTRAN:
    code('endif')
  elif CPP:
    code('}')

def op2_gen_cuda_simple_hyb(master, date, consts, kernels,sets):

  global dims, idxs, typs, indtyps, inddims
  global FORTRAN, CPP, g_m, file_text, depth

  OP_ID   = 1;  OP_GBL   = 2;  OP_MAP = 3;

  OP_READ = 1;  OP_WRITE = 2;  OP_RW  = 3;
  OP_INC  = 4;  OP_MAX   = 5;  OP_MIN = 6;

  accsstring = ['OP_READ','OP_WRITE','OP_RW','OP_INC','OP_MAX','OP_MIN' ]

  depth = 0
  FORTRAN = 0
  CPP = 1
  g_m = 0
##########################################################################
#  output one master kernel file
##########################################################################

  file_text = ''
  comm('header')
  code('#ifdef GPUPASS')
  for nk in range (0,len(kernels)):
    name  = kernels[nk]['name']
    code('#define op_par_loop_'+name+' op_par_loop_'+name+'_gpu')
  code('#include "'+master.split('.')[0]+'_kernels.cu"')
  for nk in range (0,len(kernels)):
    name  = kernels[nk]['name']
    code('#undef op_par_loop_'+name)
  code('#else')
  for nk in range (0,len(kernels)):
    name  = kernels[nk]['name']
    code('#define op_par_loop_'+name+' op_par_loop_'+name+'_cpu')
  code('#include "../openmp/'+master.split('.')[0]+'_kernels.cpp"')
  for nk in range (0,len(kernels)):
    name  = kernels[nk]['name']
    code('#undef op_par_loop_'+name)

  code('')
  comm('user kernel files')

  for nk in range(0,len(kernels)):
    name  = kernels[nk]['name']
    unique_args = range(1,kernels[nk]['nargs']+1)
    code('')
    code('void op_par_loop_'+name+'_gpu(char const *name, op_set set,')
    depth += 2
    for m in unique_args:
      g_m = m - 1
      if m == unique_args[len(unique_args)-1]:
        code('op_arg arg'+str(g_m)+');')
      else:
        code('op_arg arg'+str(g_m)+',')
    depth -= 2
    code('')
    comm('GPU host stub function')
    code('#if OP_HYBRID_GPU')
    code('void op_par_loop_'+name+'(char const *name, op_set set,')
    depth += 2

    for m in unique_args:
      g_m = m - 1
      if m == unique_args[len(unique_args)-1]:
        code('op_arg arg'+str(g_m)+'){')
        code('')
      else:
        code('op_arg arg'+str(g_m)+',')

    IF('OP_hybrid_gpu')
    code('op_par_loop_'+name+'_gpu(name, set,')
    depth += 2
    for m in unique_args:
      g_m = m - 1
      if m == unique_args[len(unique_args)-1]:
        code('arg'+str(g_m)+');')
        code('')
      else:
        code('arg'+str(g_m)+',')
    depth -=2
    code('}else{')
    code('op_par_loop_'+name+'_cpu(name, set,')
    depth += 2
    for m in unique_args:
      g_m = m - 1
      if m == unique_args[len(unique_args)-1]:
        code('arg'+str(g_m)+');')
        code('')
      else:
        code('arg'+str(g_m)+',')
    depth -=2
    ENDIF()
    depth-=2
    code('}')
    code('#else')
    code('void op_par_loop_'+name+'(char const *name, op_set set,')
    depth += 2

    for m in unique_args:
      g_m = m - 1
      if m == unique_args[len(unique_args)-1]:
        code('op_arg arg'+str(g_m)+'){')
        code('')
      else:
        code('op_arg arg'+str(g_m)+',')


    code('op_par_loop_'+name+'_gpu(name, set,')
    depth += 2
    for m in unique_args:
      g_m = m - 1
      if m == unique_args[len(unique_args)-1]:
        code('arg'+str(g_m)+');')
        code('')
      else:
        code('arg'+str(g_m)+',')
    depth-=2
    code('}')
    depth-=2
    code('#endif //OP_HYBRID_GPU')
  code("#endif")
  master = master.split('.')[0]
  fid = open('cuda/'+master.split('.')[0]+'_hybkernels.cu','w')
  fid.write('//\n// auto-generated by op2.py\n//\n\n')
  fid.write(file_text)
  fid.close()