import re
import os
import op2_gen_common
from op2_gen_seq import op2_gen_seq


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
    line = re.sub('<INDDIM>',str(inddims[m]),line)
    line = re.sub('<INDTYP>',str(indtyps[m]),line)

  line = re.sub('<INDARG>','ind_arg'+str(m),line)
  line = re.sub('<DIM>',str(dims[m]),line)
  line = re.sub('<ARG>','arg'+str(m),line)
  line = re.sub('<TYP>',typs[m],line)
  line = re.sub('<IDX>',str(int(idxs[m])),line)
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
  code('if ('+ line + ') {')
  depth += 2

def ENDIF():
  global file_text, FORTRAN, CPP, g_m
  global depth
  depth -= 2
  code('}')


def find_closing_brace(lines, start_index):
    opening_brace_count = 0
    for i in range(start_index, len(lines)):
        line = lines[i]
        opening_brace_count += line.count('{')
        opening_brace_count -= line.count('}')
        if opening_brace_count == 0:
            return i
    return -1

def generate_double_consts(consts,masterFile,suffix):
  
  global file_text
  file_text=''

  for nc in range (0,len(consts)):
    if not consts[nc]['user_declared']:
      if consts[nc]['dim']==1:
        code(''+consts[nc]['type'][1:-1]+' '+consts[nc]['name']+'_d;')
      else:
        if consts[nc]['dim'].isdigit() and int(consts[nc]['dim']) > 0:
          num = str(consts[nc]['dim'])
        else:
          num = 'MAX_CONST_SIZE'
        code(''+consts[nc]['type'][1:-1]+' '+consts[nc]['name']+'_d['+num+'];')
  code('')

  file_text=file_text.replace('float','double')

  master = masterFile.split('.')[0]
  seqkernels_filename='seq/'+master.split('.')[0]+'_seqkernels'+suffix+'.cpp'
  lines=[]
  with open(seqkernels_filename, 'r') as file:
        lines = file.readlines()

  for i, line in enumerate(lines):
        if 'global constants' in line:
            # Insert new text in the line after "global constants"
            lines.insert(i + 1, file_text)
            break
        
  with open(seqkernels_filename, 'w') as file:
        file.writelines(lines)

  content=''
  with open(seqkernels_filename, 'r') as file:
    content = file.read()

 
  for nc in range(0,len(consts)):
    pattern = r'void op_decl_const_{0}\(int dim, char const \*type,\n\s+float \*dat\){{'.format(consts[nc]['name'])
    if consts[nc]['dim']==1:
      replacement_string = "{0}_d = (double){0};".format(consts[nc]['name'])
    else:
      replacement_string = "for (int i=0; i<dim; i++){{{0}_d[i] = (double)({0}[i]);}}".format(consts[nc]['name'])
    content = re.sub(pattern, r'\g<0>\n\t{0}\n'.format(replacement_string), content)
    content = content.replace('op_decl_const_{}('.format(consts[nc]['name']),'op_decl_const_{}_d('.format(consts[nc]['name']))
    


  with open(seqkernels_filename, 'w') as file:
        file.write(content)



def generate_float_consts(consts,masterFile,suffix):
  
  global file_text
  file_text=''

  for nc in range (0,len(consts)):
    if not consts[nc]['user_declared']:
      if consts[nc]['dim']==1:
        code(''+consts[nc]['type'][1:-1]+' '+consts[nc]['name']+'_f;')
      else:
        if consts[nc]['dim'].isdigit() and int(consts[nc]['dim']) > 0:
          num = str(consts[nc]['dim'])
        else:
          num = 'MAX_CONST_SIZE'
        code(''+consts[nc]['type'][1:-1]+' '+consts[nc]['name']+'_f['+num+'];')
  code('')

#  file_text=file_text.replace('float','double')

  master = masterFile.split('.')[0]
  seqkernels_filename='seq/'+master.split('.')[0]+'_seqkernels'+suffix+'.cpp'
  lines=[]
  with open(seqkernels_filename, 'r') as file:
        lines = file.readlines()

  for i, line in enumerate(lines):
        if 'global constants' in line:
            # Insert new text in the line after "global constants"
            lines.insert(i + 1, file_text)
            break
        
  with open(seqkernels_filename, 'w') as file:
        file.writelines(lines)

  content=''
  with open(seqkernels_filename, 'r') as file:
    content = file.read()
    
  for nc in range(0,len(consts)):
    pattern = r'void op_decl_const_{0}\(int dim, char const \*type,\n\s+float \*dat\){{'.format(consts[nc]['name'])
    if consts[nc]['dim']==1:
      replacement_string = "{0}_f = {0};".format(consts[nc]['name'])
    else:
      replacement_string = "for (int i=0; i<dim; i++){{{0}_f[i] = {0}[i];}}".format(consts[nc]['name'])
    content = re.sub(pattern, r'\g<0>\n\t{0}\n'.format(replacement_string), content)
    content = content.replace('op_decl_const_{}('.format(consts[nc]['name']),'op_decl_const_{}_f('.format(consts[nc]['name']))
    
     
  with open(seqkernels_filename, 'w') as file:
        file.write(content)


def generate_master_consts(consts,masterFile,kernels):
  global file_text, depth
  file_text=''
  masterFile = masterFile.split('.')[0]
  app_name=masterFile.split('.')[0]

  
  code('#include "{}_seqkernels_float.cpp"'.format(app_name))
  code('#include "{}_seqkernels_double.cpp"'.format(app_name))


  for nc in range(0,len(consts)):
    code('')
    code('void op_decl_const_'+consts[nc]['name']+'(int dim, char const *type,')
    code('                       '+consts[nc]['type'][1:-1]+' *dat){')
    depth+=2
    code('op_decl_const_'+consts[nc]['name']+'_f(dim, type, dat);')
    code('op_decl_const_'+consts[nc]['name']+'_d(dim, type, dat);')
    depth-=2
    code('}')
  
  
  code('// user kernel files')

  for nk in range(0,len(kernels)):
    code('#include "'+kernels[nk]['name']+'_seqkernel.cpp"')
  


  fid = open('seq/'+app_name+'_seqkernels.cpp','w')
  fid.write('//\n// auto-generated by op2.py\n//\n\n')
  fid.write(file_text)
  fid.close()



def op2_gen_mixedprec(masterFile, date, consts, kernels):

  OP_ID   = 1;  OP_GBL   = 2;  OP_MAP = 3;

  OP_READ = 1;  OP_WRITE = 2;  OP_RW  = 3;
  OP_INC  = 4;  OP_MAX   = 5;  OP_MIN = 6;

  global FORTRAN, CPP, g_m, file_text, depth  
  global dims, idxs, typs, indtyps, inddims
  depth = 0
  g_m=0
  file_text=''

  # List of global constant names to modify
  #global_constants = ['gm1', 'eps']  #airfoil
  global_constants = [const['name'] for const in consts]
  
  # create output directory if it doesn't exist
  output_dir = 'mixed_kernels'
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
  
  suffix='_float'
  user_kernels=[]

  for nk in range (0,len(kernels)):
    name, nargs, dims, maps, var, typs, accs, idxs, inds, soaflags, optflags, decl_filepath, \
            ninds, inddims, indaccs, indtyps, invinds, mapnames, invmapinds, mapinds, nmaps, nargs_novec, \
            unique_args, vectorised, cumulative_indirect_index = op2_gen_common.create_kernel_info(kernels[nk])
    
    user_kernels.append('{}/{}_float.h'.format(output_dir,name))
    red_vars=[]
    for i in range(0,nargs):
      if (maps[i]==OP_GBL and accs[i]==OP_INC):
        red_vars.append(i)
        
    if len(red_vars)>0:
      generate_mixed_prec_red_kernel(name,global_constants,output_dir,red_vars)
    else:
      generate_mixed_prec_kernel(name,global_constants,output_dir)

  

  op2_gen_seq(masterFile, date, consts, kernels,suffix,user_kernels) # MPI+GENSEQ version - initial version, no vectorisation
  
  generate_float_consts(consts,masterFile,suffix)

  for nk in range (0,len(kernels)):
    name, nargs, dims, maps, var, typs, accs, idxs, inds, soaflags, optflags, decl_filepath, \
            ninds, inddims, indaccs, indtyps, invinds, mapnames, invmapinds, mapinds, nmaps, nargs_novec, \
            unique_args, vectorised, cumulative_indirect_index = op2_gen_common.create_kernel_info(kernels[nk])
 
    for i in range(0,nargs_novec):
      if (not (maps[i]==OP_GBL and accs[i]==OP_INC)):
        kernels[nk]['typs'][i] = kernels[nk]['typs'][i].replace('float','double')

    for i in range(0,ninds):
      if (not (maps[i]==OP_GBL and accs[i]==OP_INC)):
        kernels[nk]['indtyps'][i] = kernels[nk]['indtyps'][i].replace('float','double')

    #print(kernels[nk])
    
  suffix='_double'
  for i in range(len(user_kernels)):
    user_kernels[i] = user_kernels[i].replace('float', 'double')

  op2_gen_seq(masterFile, date, consts, kernels,suffix,user_kernels) # MPI+GENSEQ version - initial version, no vectorisation


  for nk in range (0,len(kernels)):
    name, nargs, dims, maps, var, typs, accs, idxs, inds, soaflags, optflags, decl_filepath, \
            ninds, inddims, indaccs, indtyps, invinds, mapnames, invmapinds, mapinds, nmaps, nargs_novec, \
            unique_args, vectorised, cumulative_indirect_index = op2_gen_common.create_kernel_info(kernels[nk])
 
    file_text = ''
    depth = 0
    g_m = 0

    fid = open('seq/'+name+'_seqkernel.cpp','w')
    fid.write('//\n// auto-generated by op2.py\n//\n\n')

    #code('#include "{}_seqkernel_float.cpp"'.format(name))
    #code('#include "{}_seqkernel_double.cpp"'.format(name))
    code('')
    code('void op_par_loop_'+name+'(char const *name, op_set set,')
    depth += 2

    for m in unique_args:
      g_m = m - 1
      if m == unique_args[len(unique_args)-1]:
        code('op_arg <ARG>){');
        code('')
      else:
        code('op_arg <ARG>,')

    IF('OP_precision==0')
    code('op_par_loop_'+name+'_float(name, set,')
    
   # args_text = 'arg0'
   # for i in range(1,nargs_novec):
   #   args_text+=',arg{}'.format(i)
    args_text = ''
    for m in unique_args:
      g_m = m - 1
      if m == unique_args[len(unique_args)-1]:
         args_text+='arg{}'.format(g_m)
      else:
        args_text+='arg{},'.format(g_m)

      
    depth += 2
    code(args_text+');')
    depth -= 2
    ENDIF()    
    IF('OP_precision==1')
    code('op_par_loop_'+name+'_double(name, set,')
    depth += 2
    code(args_text+');')
    depth -= 2
    ENDIF()
    depth -= 2

    code('}')
    fid.write(file_text)
    fid.close()
  
  # Create double consts
  generate_double_consts(consts,masterFile,suffix)
  generate_master_consts(consts,masterFile,kernels)
  




  
def generate_float(name, global_constants):
    # Read the contents of the input file into a variable
    with open('{}.h'.format(name), 'r') as input_file:
        input_contents = input_file.read()

    # Replace occurrences of the function name with the new name and type (float)    
    output_contents = input_contents.replace('#ifndef op2_mf_{}'.format(name), '#ifndef op2_mf_{}_f'.format(name))
    output_contents = output_contents.replace('#define op2_mf_{}'.format(name), '#define op2_mf_{}_f'.format(name))
    output_contents = output_contents.replace('inline void {}'.format(name), 'inline void {}_float'.format(name))

    # Rename global constants with '_f' suffix in the first copy
    for var_name in global_constants:
        output_contents = re.sub(r'\b{}\b'.format(var_name), '{}_f'.format(var_name), output_contents)

    # Return the modified contents as float version
    return output_contents

def generate_double(name, global_constants):
    # Read the contents of the input file into a variable
    with open('{}.h'.format(name), 'r') as input_file:
        input_contents = input_file.read()

    # Replace occurrences of the function name with the new name and type (double)
    output_contents = input_contents.replace('#ifndef op2_mf_{}'.format(name), '#ifndef op2_mf_{}_d'.format(name))
    output_contents = output_contents.replace('#define op2_mf_{}'.format(name), '#define op2_mf_{}_d'.format(name))
    output_contents = output_contents.replace('inline void {}'.format(name), 'inline void {}_double'.format(name))
    output_contents = output_contents.replace('float', 'double')

    # Rename global constants with '_d' suffix in the double copy
    for var_name in global_constants:
        output_contents = re.sub(r'\b{}\b'.format(var_name), '{}_d'.format(var_name), output_contents)

    # Replace all literal constants with their double counterparts
    output_contents = re.sub(r'(?<!\w)(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?[fF]?', r'\g<1>', output_contents)

    # Return the modified contents as double version
    return output_contents



def generate_mixed_prec_kernel(name,global_constants,output_dir):

  print('generating kernel: ',name)
  
  with open('{}/{}_float.h'.format(output_dir,name), 'w') as f:
    f.write(generate_float(name, global_constants))
  
  with open('{}/{}_double.h'.format(output_dir,name), 'w') as f:
    f.write(generate_double(name, global_constants))


def generate_double_red(name, global_constants, red_vars):
    output_contents = generate_double(name, global_constants)

    # Pattern to match the function declaration
    pattern = r'\w+\s+{}\s*\(([\s\S]*?)\)'.format('{}_double'.format(name))
    match = re.search(pattern, output_contents)

    parameters = match.group(1).split(',')
    parameters = [param.strip() for param in parameters]
   
    # Redo all reduction variables
    for i, param in enumerate(parameters):
      if i in red_vars:
        output_contents = output_contents.replace(param, param.replace('double','float'))

    # Return the modified contents as double version
    return output_contents


def generate_mixed_prec_red_kernel(name,global_constants,output_dir,red_vars):
  print('generating red kernel: ',name)
  
  with open('{}/{}_float.h'.format(output_dir,name), 'w') as f:
    f.write(generate_float(name, global_constants))
  
  with open('{}/{}_double.h'.format(output_dir,name), 'w') as f:
    f.write(generate_double_red(name, global_constants,red_vars))

