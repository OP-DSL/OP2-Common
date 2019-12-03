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
    code('for ( int '+i+'='+start+'; '+i+'<'+finish+'; '+i+'++ ){')
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
#  output individual kernel file
##########################################################################
    if not os.path.exists('cuda'):
        os.makedirs('cuda')
    fid = open('cuda/'+name+'_kernel.cu','w')
    fid.write('//\n// auto-generated by op2.py\n//\n\n')
    fid.write(user_function)
    fid.close()

##########################################################################
#  output individual kernel file - JIT compiled version
##########################################################################
    if not os.path.exists('cuda'):
        os.makedirs('cuda')
    fid = open('cuda/'+name+'_kernel_rec.cu','w')
    fid.write('//\n// auto-generated by op2.py\n//\n\n')
    fid.write(jit_include+user_function)# '\n} //end extern c')
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
