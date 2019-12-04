##########################################################################
#
# MPI Sequential code generator
#
# This routine is called by op2 which parses the input files
#
# It produces a file xxx_kernel.cpp for each kernel,
# plus a master kernel file
#
##########################################################################

import re
import datetime
import os
import op2_gen_common

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

def rep(line,m):
  global dims, idxs, typs, indtyps, inddims
  if m < len(inddims):
    line = re.sub('INDDIM',str(inddims[m]),line)
    line = re.sub('INDTYP',str(indtyps[m]),line)

  line = re.sub('INDARG','ind_arg'+str(m),line)
  line = re.sub('DIM',str(dims[m]),line)
  line = re.sub('ARG','arg'+str(m),line)
  line = re.sub('TYP',typs[m],line)
  line = re.sub('IDX',str(int(idxs[m])),line)
  return line

def code(text):
  global file_text, FORTRAN, CPP, g_m
  global depth
  if text == '':
    prefix = ''
  else:
    prefix = ' '*depth
  file_text += prefix+rep(text,g_m).rstrip()+'\n'

def FOR(i,start,finish):
  global file_text, FORTRAN, CPP, g_m
  global depth
  if FORTRAN:
    code('do '+i+' = '+start+', '+finish+'-1')
  elif CPP:
    code('for (int '+i+' = '+start+'; '+i+' < '+finish+'; ++'+i+')')
    code('{') 
  depth += 2

def FOR_INC(i,start,finish,inc):
  global file_text,FORTRAN, CPP, g_m
  global depth
  if FORTRAN:
    code('do '+i+' = '+start+', '+finish+'-1')
  elif CPP:
    code('for (int '+i+' = '+start+'; '+i+' < '+finish+'; '+i+' += '+inc+')') 
    code('{') 
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


def op2_gen_cuda_jit(master, date, consts, kernels):

  global dims, idxs, typs, indtyps, inddims
  global FORTRAN, CPP, g_m, file_text, depth

  OP_ID   = 1;  OP_GBL   = 2;  OP_MAP = 3;

  OP_READ = 1;  OP_WRITE = 2;  OP_RW  = 3;
  OP_INC  = 4;  OP_MAX   = 5;  OP_MIN = 6;

  accsstring = ['OP_READ','OP_WRITE','OP_RW','OP_INC','OP_MAX','OP_MIN' ]

  any_soa = 0
  for nk in range (0,len(kernels)):
    any_soa = any_soa or sum(kernels[nk]['soaflags'])

##########################################################################
#  create new kernel file
##########################################################################

  for nk in range (0,len(kernels)):
    name, nargs, dims, maps, var, typs, accs, idxs, inds, soaflags, optflags, decl_filepath, \
    ninds, inddims, indaccs, indtyps, invinds, mapnames, invmapinds, mapinds, nmaps, nargs_novec, \
    unique_args, vectorised, cumulative_indirect_index = op2_gen_common.create_kernel_info(kernels[nk])
 
    optidxs = [0]*nargs
    indopts = [-1]*nargs
    nopts = 0
    for i in range(0,nargs):
      if optflags[i] == 1 and maps[i] == OP_ID:
        optidxs[i] = nopts
        nopts = nopts+1
      elif optflags[i] == 1 and maps[i] == OP_MAP:
        if i == invinds[inds[i]-1]: #i.e. I am the first occurence of this dat+map combination
          optidxs[i] = nopts
          indopts[inds[i]-1] = i
          nopts = nopts+1
        else:
          optidxs[i] = optidxs[invinds[inds[i]-1]]

#
# set two logicals
#
    j = 0
    for i in range(0,nargs):
      if maps[i] == OP_MAP and accs[i] == OP_INC:
        j = i
    ind_inc = j > 0

    j = 0
    for i in range(0,nargs):
      if maps[i] == OP_GBL and accs[i] <> OP_READ:
        j = i
    reduct = j > 0

    FORTRAN = 0;
    CPP     = 1;
    g_m = 0;
    file_text = ''
    depth = 0

##########################################################################
#  JIT Includes - const file and op_lib
##########################################################################

    code('#include "op_lib_cpp.h"')
    comm('global_constants - values #defined by JIT')
    code('#include "jit_const.h')
    code('')   
 
    jit_include = file_text
    file_text = ''

##########################################################################
#  User Function - Executed on device
##########################################################################

    f = open(decl_filepath, 'r')
    kernel_text = f.read()
    f.close()

    if CPP:
      #Get include statements from kernel file
      includes = op2_gen_common.extract_includes(kernel_text)
      if len(includes) > 0:
        for include in includes:
          code(include)
        code('')

    kernel_text = op2_gen_common.comment_remover(kernel_text)
    kernel_text = op2_gen_common.remove_trailing_w_space(kernel_text)

    #find void, any whitespace and then the name with no extra alphanumeric chars either side
    p = re.compile('void\\s+\\b'+name+'\\b')
    i = p.search(kernel_text).start()

    if (i < 0):
      print "\n********"
      print "Error: cannot locate user kernel function: "+name+" - Aborting code generation"
      exit(2)

    lbra = i + kernel_text[i:].find('{')
    rbra = op2_gen_common.para_parse(kernel_text, lbra, '{', '}')
    signature_text = kernel_text[i:lbra]
    lpar = signature_text.find('(')
    rpar = op2_gen_common.para_parse(signature_text, 0, '(', ')')
  
    signature_text = signature_text[lpar+1:rpar]
    body_text = kernel_text[lbra+1:rbra]

    #Replace "#include <FILE>" with contents of file
    body_text = op2_gen_common.replace_local_includes_with_file_contents(body_text, os.path.dirname(master))
  
    params = signature_text.split(',')
   
    #check number of arguments
    if len(params) != nargs_novec:
      print "Error parsing user kernel ("+name+"): must have "+str(nargs)+" arguments"
      return
  
    for i in range(0,nargs_novec):
      var = params[i].strip()
      #StructOfArrays and ID/GBL or READ, MAX, or MIN
      if kernels[nk]['soaflags'][i] and ( not(kernels[nk]['maps'][i] == OP_MAP and kernels[nk]['accs'][i] == OP_INC)):
        var = var.replace('*','')
        sp = re.compile('\\s+\\b').split(var)
        var2 = sp[len(sp)-1].strip()
        
        stride = op2_gen_common.get_stride_string(unique_args[i]-1,maps,mapnames,name)

        if int(kernels[nk]['idxs'][i]) < 0 and kernels[nk]['maps'][i] == OP_MAP:
        #Multiple second index by stride
          body_text = re.sub(r'\b'+var2+'(\[[^\]]\])\[([\\s+\*A-Za-z0-9]*)\]'+'', var2+r'\1[(\2)*'+stride+']', body_text) 
        else:
          body_text = re.sub('\*\\b'+var2+'\\b\\s*(?!\[)', var2+'[0]', body_text)
          body_text = re.sub(r'\b'+var2+'\[([\\s\+\*A-Za-z0-9]*)\]'+'', var2+r'[(\1)*'+stride+']', body_text)

        
    user_function = "//user function\n" + "__device__ void " + name + "_gpu( " + signature_text + ")\n{" + body_text + "}\n"

##########################################################################
#  Kernel Function - Calls user function
##########################################################################

    code('') 
    comm('C CUDA kernel function')
    
    #Function head 

    if FORTRAN:
      code('subroutine op_cuda_'+name+'(')
    elif CPP:
      code('__global__ void op_cuda_'+name+'(')
  
    depth = 1
  
    if nopts > 0: #Has optional arguments
      code('int optflags,')
 
    for g_m in range(0, ninds):
      is_const = ''
      if indaccs[g_m] == OP_READ:
        is_const = 'const '
      code(is_const + 'INDTYP* __restrict INDARG,')

    if nmaps > 0:
      k = []
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and not (mapnames[g_m] in k):
          k = k + [mapnames[g_m]]
          code('const int* __restrict opDat'+str(invinds[inds[g_m]])+'Map, ')
 
    for g_m in range(0,nargs):
      is_const = ''
      if accs[g_m] == OP_READ:
        is_const = 'const '
      if maps[g_m] == OP_ID:
        code(is_const + 'TYP* __restrict ARG,')
      elif maps[g_m] == OP_GBL:
        code(is_const + 'TYP* ARG,')

    if ninds > 0:
      code('int start')
      code('int end')
    
    code('int set_size)')
    depth = 0
    code('{')

    #Function Body
    depth = 2

    for m in range(1,ninds+1):
      g_m = m-1;
      v = [int(inds[i] == m) for i in range(len(inds))]
      v_i = [vectorised[i] for i in range(len(inds)) if inds[i] == m]
      if sum(v) > 1 and sum(v_i) > 0:
        if indaccs[m-1] == OP_INC:
          ind = int(max([idxs[i] for i in range(len(inds)) if inds[i] == m])) + 1
          code('INDTYP* arg'+str(invinds[m-1])+'_vec['+str(ind)+'] = {')
       
          #Fill list
          depth += 2
          for n in range(0,nargs):
            if ind[n] == m:
              g_m = n
              code('ARG_1,')
          depth -= 2
          code('};')
#
# Has indirection
#
    if ninds > 0:          
      code('int tid = threadIdx.x + blockIdx.x * blockDim.x;')
      IF('tid + start < end')
      code('int n = tid + start;')
 
      comm('Initialise locals')
      for g_m in range(0,nargs):
        if (maps[g_m] == OP_GBL or maps[g_m] == OP_MAP) and accs[g_m] <> OP_READ and accs[g_m] <> OP_WRITE:
          code('TYP ARG_1[DIM]')
          if accs[g_m] == OP_INC:
            FOR('d','0','DIM')
            code('ARG_1[d]=ZERO_TYP;')
            ENDFOR()
          elif maps[g_m] == OP_GBL:
            FOR('d','0','DIM')
            code('ARG_1[d]=ARG[d+blockIdx.x*DIM];')
            ENDFOR()

      #mapIdx decls
      k = []
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and not(mapinds[g_m] in k):
          k = k + [mapinds[g_m]]
          code('int map'+str(mapinds[g_m])+'idx;')

          # Check optional map has been passed
          if optflags[g_m] == 1:
            IF('optflags & 1 << '+str(optidxs[g_m]))

          code('map'+str(mapinds[g_m])+'idx = opDat'+str(invmapinds[inds[g_m]-1])+'Map[n + set_size * '+str(int(idxs[g_m]))+'];')

          # Close IF for optional map
          if optflags[g_m] == 1:
            ENDIF()

  ## SKIPPED op2_gen_cuda_simple.py lines 574 - 594 

#
# No indirection
#
    else:
      comm('Process set elements')
      FOR_INC('n','threadIdx.x+blockIdx.x*blockDim.x','set_size','blockDim.x*gridDim.x')

#
# User Function call
#

    code('')
    comm('user function call')
    func_name = name + '_gpu('
    indent = ' '*(len(func_name))
    for g_m in range(0,nargs):
      start = indent
      end = ','
      if g_m == 0:
        start = func_name
      if g_m == (nargs-1):
        end = ''

      if maps[g_m] == OP_GBL:
        if accs[g_m] == OP_READ or accs[m] == OP_WRITE:
          code(start + 'ARG' + end)
        else:
          code(start + 'ARG_1' + end)

      elif maps[g_m] == OP_MAP:
        if vectorised[g_m]:
          if m+1 in unique_args: 
            code(start + 'ARG_vec' + end)
        elif accs[g_m] == OP_INC:
          code(start + 'ARG_1' + end)
        else:
          is_soa = '*DIM'
          if soaflags[g_m]:
            is_soa = ''
          code(start + 'ind_arg'+str(inds[g_m]-1)+'+map'+str(mapinds[g_m])+'idx'+is_soa + end)
      elif maps[g_m] == OP_ID:
        is_soa = '*DIM'
        if soaflags[g_m]:
          is_soa = ''
        code(start + 'ARG+n' + is_soa + end)
      else:
        print 'internal error 1'
     
    code(');') 
    code('')

#    
# updating for indirect kernels
#
    if ninds > 0:
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and accs[g_m] == OP_INC:
          if optflags[g_m] == 1:
            IF('optflags & 1<<'+str(optidxs[g_m]))
          for d in range(0,int(dims[g_m])):
            stride = ''
            dim = '*DIM'
            if soaflags[g_m]:
              stride = '*' + op2_gen_common.get_stride_string(g_m,maps,mapnames,name)
              dim = ''
            code('atomicAdd(&ind_arg'+str(inds[g_m]-1)+'['+str(d)+stride+'+map'+str(mapinds[g_m])+'idx'+dim+'],ARG_1['+str(d)+']);')
          if optflags[g_m] == 1:
            ENDIF()
    
      ENDIF()
    else:
      ENDFOR()

#
# global reduction
#
    if reduct:
      code('')
      comm('global reductions')
      code('')
    
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL and accs[g_m] <> OP_READ and accs[g_m] <> OP_WRITE:
          op = ''
          if accs[m]   == OP_INC:  
            op = 'OP_INC'
          elif accs[m] == OP_MIN:
            op = 'OP_MIN'
          elif accs[m] == OP_MAX: 
            op = 'OP_MAX'
          else:
            print 'internal error: invalid reduction option'
            sys.exit(2)  
          
          FOR('d','0','DIM')
          code('op_reduction<'+op+'>(&ARG[d+blockIdx.x*DIM],ARG_1[d];')
          ENDFOR()
    
#
# end function
#    
    depth -= 2
    code('}')
    code('')
  
    kernel_function = file_text
    file_text = ''

##########################################################################
#  output individual kernel file
##########################################################################
    if not os.path.exists('cuda'):
        os.makedirs('cuda')
    fid = open('cuda/'+name+'_kernel.cu','w')
    fid.write('//\n// auto-generated by op2.py\n//\n\n')
    fid.write(user_function+kernel_function)
    fid.close()

##########################################################################
#  output individual kernel file - JIT compiled version
##########################################################################
    if not os.path.exists('cuda'):
        os.makedirs('cuda')
    fid = open('cuda/'+name+'_kernel_rec.cu','w')
    fid.write('//\n// auto-generated by op2.py\n//\n\n')
    fid.write(jit_include+user_function+kernel_function)# '\n} //end extern c')
    fid.close()


# end of main kernel call loop


##########################################################################
#  output one master kernel file
##########################################################################

  file_text =''
  if os.path.exists('./user_types.h'):
    code('#include "../user_types.h"')
  comm(' header                 ')
  code('#include "op_lib_cpp.h"       ')
  code('')
  comm(' global constants       ')

  for nc in range (0,len(consts)):
    if consts[nc]['dim']==1:
      code('extern '+consts[nc]['type'][1:-1]+' '+consts[nc]['name']+';')
    else:
      if consts[nc]['dim'] > 0:
        num = str(consts[nc]['dim'])
      else:
        num = 'MAX_CONST_SIZE'

      code('extern '+consts[nc]['type'][1:-1]+' '+consts[nc]['name']+'['+num+'];')

  code('\n#ifdef OP2_JIT')
  code('')
  for nk in range (0,len(kernels)):
    name  = kernels[nk]['name']
    code('void (*'+name+'_function)(struct op_kernel_descriptor *desc) = NULL;')

  code('')
  code('void jit_compile() {')
  depth += 2
  code('op_printf("JIT compiling op_par_loops\\n");')
  code('')

  comm('Write constants to header file')
  IF('op_is_root()')
  code('int ret = system("make -j '+master.split('.')[0]+'_cuda_jit");')
  ENDIF()
  code('op_mpi_barrier();')

  code('void *handle;')
  code('char *error;')
  code('')

  comm('create .so')
  code('handle = dlopen("cuda/'+master.split('.')[0]+'_kernel_rec.so", RTLD_LAZY);')
  IF('!handle')
  code('fputs(dlerror(), stderr);')
  code('exit(1);')
  ENDIF()
  code('')

  comm('dynamically load functions from the  .so')
  for nk in range (0,len(kernels)):
    name  = kernels[nk]['name']

    code(name+'_function = (void (*)(op_kernel_descriptor *))dlsym(')
    code('  handle, "op_par_loop_'+name+'_rec_execute");')
    IF('(error = dlerror()) != NULL')
    code('fputs(error, stderr);')
    code('exit(1);')
    ENDIF()

  code('op_mpi_barrier();')
  code('jit_compiled = 1;')

  depth -= 2
  code('}')
  code('\n#endif')

  comm(' user kernel files')

  for nk in range(0,len(kernels)):
    code('#include "'+kernels[nk]['name']+'_kernel.cu"')
  master = master.split('.')[0]
  fid = open('cuda/'+master.split('.')[0]+'_kernels.cu','w')
  fid.write('//\n// auto-generated by op2.py\n//\n\n')
  fid.write(file_text)
  fid.close()
