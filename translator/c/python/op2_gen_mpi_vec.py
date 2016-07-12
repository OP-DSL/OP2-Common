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
import glob

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

def FOR2(i,start,finish,inc):
  global file_text, FORTRAN, CPP, g_m
  global depth
  if FORTRAN:
    code('do '+i+' = '+start+', '+finish+'-1, '+inc)
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

def comment_remover(text):
    """Remove comments from text"""

    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return ''
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)

def remove_trailing_w_space(text):
  text = text+' '
  line_start = 0
  line = ""
  line_end = 0
  striped_test = ''
  count = 0
  while 1:
    line_end =  text.find("\n",line_start+1)
    line = text[line_start:line_end]
    line = line.rstrip()
    striped_test = striped_test + line +'\n'
    line_start = line_end + 1
    line = ""
    if line_end < 0:
      return striped_test[:-1]

def para_parse(text, j, op_b, cl_b):
    """Parsing code block, i.e. text to find the correct closing brace"""

    depth = 0
    loc2 = j

    while 1:
      if text[loc2] == op_b:
            depth = depth + 1

      elif text[loc2] == cl_b:
            depth = depth - 1
            if depth == 0:
                return loc2
      loc2 = loc2 + 1

def op2_gen_mpi_vec(master, date, consts, kernels):

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

    name  = kernels[nk]['name']
    nargs = kernels[nk]['nargs']
    dims  = kernels[nk]['dims']
    maps  = kernels[nk]['maps']
    var   = kernels[nk]['var']
    typs  = kernels[nk]['typs']
    accs  = kernels[nk]['accs']
    idxs  = kernels[nk]['idxs']
    inds  = kernels[nk]['inds']
    soaflags = kernels[nk]['soaflags']

    ninds   = kernels[nk]['ninds']
    inddims = kernels[nk]['inddims']
    indaccs = kernels[nk]['indaccs']
    indtyps = kernels[nk]['indtyps']
    invinds = kernels[nk]['invinds']
    mapnames = kernels[nk]['mapnames']
    invmapinds = kernels[nk]['invmapinds']
    mapinds = kernels[nk]['mapinds']
    nmaps = 0
    if ninds > 0:
      nmaps = max(mapinds)+1

    vec =  [m for m in range(0,nargs) if int(idxs[m])<0 and maps[m] == OP_MAP]

    if len(vec) > 0:
      unique_args = [1];
      vec_counter = 1;
      vectorised = []
      new_dims = []
      new_maps = []
      new_vars = []
      new_typs = []
      new_accs = []
      new_idxs = []
      new_inds = []
      new_soaflags = []
      for m in range(0,nargs):
          if int(idxs[m])<0 and maps[m] == OP_MAP:
            if m > 0:
              unique_args = unique_args + [len(new_dims)+1]
            temp = [0]*(-1*int(idxs[m]))
            for i in range(0,-1*int(idxs[m])):
              temp[i] = var[m]
            new_vars = new_vars+temp
            for i in range(0,-1*int(idxs[m])):
              temp[i] = typs[m]
            new_typs = new_typs+temp
            for i in range(0,-1*int(idxs[m])):
              temp[i] = dims[m]
            new_dims = new_dims+temp
            new_maps = new_maps+[maps[m]]*int(-1*int(idxs[m]))
            new_soaflags = new_soaflags+[0]*int(-1*int(idxs[m]))
            new_accs = new_accs+[accs[m]]*int(-1*int(idxs[m]))
            for i in range(0,-1*int(idxs[m])):
              new_idxs = new_idxs+[i]
            new_inds = new_inds+[inds[m]]*int(-1*int(idxs[m]))
            vectorised = vectorised + [vec_counter]*int(-1*int(idxs[m]))
            vec_counter = vec_counter + 1;
          else:
            if m > 0:
              unique_args = unique_args + [len(new_dims)+1]
            new_dims = new_dims+[dims[m]]
            new_maps = new_maps+[maps[m]]
            new_accs = new_accs+[int(accs[m])]
            new_soaflags = new_soaflags+[soaflags[m]]
            new_idxs = new_idxs+[int(idxs[m])]
            new_inds = new_inds+[inds[m]]
            new_vars = new_vars+[var[m]]
            new_typs = new_typs+[typs[m]]
            vectorised = vectorised+[0]
      dims = new_dims
      maps = new_maps
      accs = new_accs
      idxs = new_idxs
      inds = new_inds
      var = new_vars
      typs = new_typs
      soaflags = new_soaflags;
      nargs = len(vectorised);

      for i in range(1,ninds+1):
        for index in range(0,len(inds)+1):
          if inds[index] == i:
            invinds[i-1] = index
            break
    else:
      vectorised = [0]*nargs
      unique_args = range(1,nargs+1)

    cumulative_indirect_index = [-1]*nargs;
    j = 0;
    for i in range (0,nargs):
      if maps[i] == OP_MAP:
        cumulative_indirect_index[i] = j
        j = j + 1
#
# set three logicals
#
    j = -1
    for i in range(0,nargs):
      if maps[i] == OP_MAP and accs[i] == OP_INC:
        j = i
    ind_inc = j >= 0

    j = -1
    for i in range(0,nargs):
      if maps[i] == OP_GBL and accs[i] <> OP_READ:
        j = i
    reduct = j >= 0

    j = -1
    for i in range(0,nargs):
      if maps[i] == OP_MAP :
        j = i
    indirect_kernel = j >= 0

####################################################################################
#  generate the user kernel function - creating versions for vectorisation as needed
####################################################################################

    FORTRAN = 0;
    CPP     = 1;
    g_m = 0;
    file_text = ''
    depth = 0

#
# First original version
#
    comm('user function')
    found = 0
    for files in glob.glob( "*.h" ):
      f = open( files, 'r' )
      for line in f:
        match = re.search(r''+'\\b'+name+'\\b', line)
        if match :
          file_name = f.name
          found = 1;
          break
      if found == 1:
        break;

    if found == 0:
      print "COUND NOT FIND KERNEL", name

    f = open(file_name, 'r')
    kernel_text = f.read()
    file_text += kernel_text
    f.close()

#
# Modified vectorisable version if its an indirect kernel
# - direct kernels can be vectorised without modification
#
    if indirect_kernel:
      code('#ifdef VECTORIZE')
      comm('user function -- modified for vectorisation')
      f = open(file_name, 'r')
      kernel_text = f.read()
      f.close()

      kernel_text = comment_remover(kernel_text)
      kernel_text = remove_trailing_w_space(kernel_text)

      p = re.compile('void\\s+\\b'+name+'\\b')
      i = p.search(kernel_text).start()

      if(i < 0):
        print "\n********"
        print "Error: cannot locate user kernel function name: "+name+" - Aborting code generation"
        exit(2)
      i2 = i

      #i = kernel_text[0:i].rfind('\n') #reverse find
      j = kernel_text[i:].find('{')
      k = para_parse(kernel_text, i+j, '{', '}')
      signature_text = kernel_text[i:i+j]
      l = signature_text[0:].find('(')
      head_text = signature_text[0:l] #save function name
      m = para_parse(signature_text, 0, '(', ')')
      signature_text = signature_text[l+1:m]
      body_text = kernel_text[i+j+1:k]


      # check for number of arguments
      if len(signature_text.split(',')) != nargs:
          print 'Error parsing user kernel(%s): must have %d arguments' \
                % name, nargs
          return

      new_signature_text = ''
      for i in range(0,nargs):
        var = signature_text.split(',')[i].strip()

        if maps[i] <> OP_GBL and maps[i] <> OP_ID:
          #remove * and add [*][SIMD_VEC]
          var = var.replace('*','')
          #locate var in body and replace by adding [idx]
          length = len(re.compile('\\s+\\b').split(var))
          var2 = re.compile('\\s+\\b').split(var)[length-1].strip()

          #print var2

          body_text = re.sub('\*\\b'+var2+'\\b\\s*(?!\[)', var2+'[0]', body_text)
          body_text = re.sub(r'('+var2+'\[[A-Za-z0-9]*\]'+')', r'\1'+'[idx]', body_text)


          var = var + '[*][SIMD_VEC]'
          #var = var + '[restrict][SIMD_VEC]'
        new_signature_text +=  var+', '


      #add ( , idx and )
      signature_text = head_text + '( '+new_signature_text + 'int idx ) {'
      #finally update name
      signature_text = signature_text.replace(name,name+'_vec')

      #print head_text
      #print signature_text
      #print  body_text

      file_text += signature_text + body_text + '}\n'
      code('#endif');



##########################################################################
# then C++ stub function
##########################################################################

    code('')
    comm(' host stub function')
    code('void op_par_loop_'+name+'(char const *name, op_set set,')
    depth += 2

    for m in unique_args:
      g_m = m - 1
      if m == unique_args[len(unique_args)-1]:
        code('op_arg ARG){');
        code('')
      else:
        code('op_arg ARG,')

    code('int nargs = '+str(nargs)+';')
    code('op_arg args['+str(nargs)+'];')
    code('')

    for g_m in range (0,nargs):
      u = [i for i in range(0,len(unique_args)) if unique_args[i]-1 == g_m]
      if len(u) > 0 and vectorised[g_m] > 0:
        code('ARG.idx = 0;')
        code('args['+str(g_m)+'] = ARG;')

        v = [int(vectorised[i] == vectorised[g_m]) for i in range(0,len(vectorised))]
        first = [i for i in range(0,len(v)) if v[i] == 1]
        first = first[0]

        FOR('v','1',str(sum(v)))
        code('args['+str(g_m)+' + v] = op_arg_dat(arg'+str(first)+'.dat, v, arg'+\
        str(first)+'.map, DIM, "TYP", '+accsstring[accs[g_m]-1]+');')
        ENDFOR()
        code('')
      elif vectorised[g_m]>0:
        pass
      else:
        code('args['+str(g_m)+'] = ARG;')

#
# create aligned pointers
#
    comm('create aligned pointers for dats')
    for g_m in range (0,nargs):
        if maps[g_m] <> OP_GBL:
          if (accs[g_m] == OP_INC or accs[g_m] == OP_RW or accs[g_m] == OP_WRITE):
            code('ALIGNED_TYP       TYP * __restrict__ ptr'+\
            str(g_m)+' = (TYP *) arg'+str(g_m)+'.data;')
            #code('TYP* __restrict__ __attribute__((align_value (TYP_ALIGN)))  ptr'+\
            #str(g_m)+' = (TYP *) arg'+str(g_m)+'.data;')
            code('__assume_aligned(ptr'+str(g_m)+',TYP_ALIGN);')

          else:
            code('ALIGNED_TYP const TYP * __restrict__ ptr'+\
            str(g_m)+' = (TYP *) arg'+str(g_m)+'.data;')
            code('__assume_aligned(ptr'+str(g_m)+',TYP_ALIGN);')
            #code('const TYP* __restrict__ __attribute__((align_value (TYP_ALIGN)))  ptr'+\
            #str(g_m)+' = (TYP *) arg'+str(g_m)+'.data;')



#
# start timing
#
    code('')
    comm(' initialise timers')
    code('double cpu_t1, cpu_t2, wall_t1, wall_t2;')
    code('op_timing_realloc('+str(nk)+');')
    code('op_timers_core(&cpu_t1, &wall_t1);')
    code('')

#
#   indirect bits
#
    if ninds>0:
      IF('OP_diags>2')
      code('printf(" kernel routine with indirection: '+name+'\\n");')
      ENDIF()

#
# direct bit
#
    else:
      code('')
      IF('OP_diags>2')
      code('printf(" kernel routine w/o indirection:  '+ name + '");')
      ENDIF()

    code('')
    code('int exec_size = op_mpi_halo_exchanges(set, nargs, args);')

    code('')
    IF('exec_size >0')
    code('')

#
# kernel call for indirect version
#
    if ninds>0:
      code('#ifdef VECTORIZE')

      #initialze globals
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL:
          if accs[g_m] == OP_INC:
            code('TYP dat'+str(g_m)+'[SIMD_VEC] = {0.0};')
          elif accs[g_m] == OP_MAX:
            code('TYP dat'+str(g_m)+'[SIMD_VEC] = {INFINITY};')
          elif accs[g_m] == OP_MIN:
            code('TYP dat'+str(g_m)+'[SIMD_VEC] = {-INFINITY};')

      code('#pragma novector')
      FOR2('n','0','(exec_size/SIMD_VEC)*SIMD_VEC','SIMD_VEC')
      IF('n+SIMD_VEC >= set->core_size')
      code('op_mpi_wait_all(nargs, args);')
      ENDIF()
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (accs[g_m] == OP_READ \
          or accs[g_m] == OP_RW or accs[g_m] == OP_WRITE \
          or accs[g_m] == OP_INC):
          code('ALIGNED_TYP TYP dat'+str(g_m)+'[DIM][SIMD_VEC];')

      #setup gathers
      code('#pragma simd')
      FOR('i','0','SIMD_VEC')
      if nmaps > 0:
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP :
            if (accs[g_m] == OP_READ or accs[g_m] == OP_RW or accs[g_m] == OP_WRITE):#and (not mapinds[g_m] in k):
              code('int idx'+str(g_m)+'_DIM = DIM * arg'+str(invmapinds[inds[g_m]-1])+'.map_data[(n+i) * arg'+str(invmapinds[inds[g_m]-1])+'.map->dim + '+str(idxs[g_m])+'];')
      code('')
      for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP :
            if (accs[g_m] == OP_READ or accs[g_m] == OP_RW):#and (not mapinds[g_m] in k):
              for d in range(0,int(dims[g_m])):
                code('dat'+str(g_m)+'['+str(d)+'][i] = (ptr'+str(g_m)+')[idx'+str(g_m)+'_DIM + '+str(d)+'];')
              code('')
            elif (accs[g_m] == OP_INC):
              for d in range(0,int(dims[g_m])):
                code('dat'+str(g_m)+'['+str(d)+'][i] = 0.0;')
              code('')
          else: #globals
            if (accs[g_m] == OP_INC):
              for d in range(0,int(dims[g_m])):
                code('dat'+str(g_m)+'[i] = 0.0;')
              code('')

      ENDFOR()
      #kernel call
      code('#pragma simd')
      FOR('i','0','SIMD_VEC')
      line = name+'_vec('
      indent = '\n'+' '*(depth+2)
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          line = line + indent + '&(ptr'+str(g_m)+')['+str(dims[g_m])+' * (n+i)],'
        elif maps[g_m] == OP_GBL and accs[g_m] == OP_READ:
          line = line + indent +'('+typs[g_m]+'*)arg'+str(g_m)+'.data,'
        elif maps[g_m] == OP_GBL and accs[g_m] == OP_INC:
          line = line + indent +'&dat'+str(g_m)+'[i],'
        else:
          line = line + indent + 'dat'+str(g_m)+','
      line = line +indent +'i);'
      code(line)
      ENDFOR()
      #do the scatters
      FOR('i','0','SIMD_VEC')
      if nmaps > 0:
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP :
            if (accs[g_m] == OP_INC or accs[g_m] == OP_RW or accs[g_m] == OP_WRITE):#and (not mapinds[g_m] in k):
              code('int idx'+str(g_m)+'_DIM = DIM * arg'+str(invmapinds[inds[g_m]-1])+'.map_data[(n+i) * arg'+str(invmapinds[inds[g_m]-1])+'.map->dim + '+str(idxs[g_m])+'];')
      code('')
      for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP :
            if (accs[g_m] == OP_INC ):
              for d in range(0,int(dims[g_m])):
                code('(ptr'+str(g_m)+')[idx'+str(g_m)+'_DIM + '+str(d)+'] += dat'+str(g_m)+'['+str(d)+'][i];')
              code('')
            if (accs[g_m] == OP_WRITE or accs[g_m] == OP_RW):
              for d in range(0,int(dims[g_m])):
                code('(ptr'+str(g_m)+')[idx'+str(g_m)+'_DIM + '+str(d)+'] = dat'+str(g_m)+'['+str(d)+'][i];')
              code('')
      ENDFOR()

      #do reductions
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL:
          FOR('i','0','SIMD_VEC')
          if accs[g_m] == OP_INC:
            code('*(TYP*)arg'+str(g_m)+'.data += dat'+str(g_m)+'[i];')
          elif accs[g_m] == OP_MAX:
            code('*(TYP*)arg'+str(g_m)+'.data = MAX(*(TYP*)arg'+str(g_m)+'.data,dat'+str(g_m)+'[i]);')
          elif accs[g_m] == OP_MIN:
            code('*(TYP*)arg'+str(g_m)+'.data = MIN(*(TYP*)arg'+str(g_m)+'.data,dat'+str(g_m)+'[i]);')
          ENDFOR()


      ENDFOR()
      code('')
      comm('remainder')
      FOR('n','(exec_size/SIMD_VEC)*SIMD_VEC','exec_size')
      depth = depth -2
      code('#else')
      FOR('n','0','exec_size')
      depth = depth -2
      code('#endif')
      depth = depth +2
      IF('n==set->core_size')
      code('op_mpi_wait_all(nargs, args);')
      ENDIF()
      if nmaps > 0:
        k = []
        #print name
        #print maps
        #print mapinds
        for g_m in range(0,nargs):
          #print g_m
          if maps[g_m] == OP_MAP and (not mapinds[g_m] in k):
            k = k + [mapinds[g_m]]
            code('int map'+str(mapinds[g_m])+'idx = arg'+str(invmapinds[inds[g_m]-1])+'.map_data[n * arg'+str(invmapinds[inds[g_m]-1])+'.map->dim + '+str(idxs[g_m])+'];')
      code('')
      line = name+'('
      indent = '\n'+' '*(depth+2)
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          line = line + indent + '&(ptr'+str(g_m)+')['+str(dims[g_m])+' * n]'
        if maps[g_m] == OP_MAP:
          line = line + indent + '&(ptr'+str(g_m)+')['+str(dims[g_m])+' * map'+str(mapinds[g_m])+'idx]'
        if maps[g_m] == OP_GBL:
          line = line + indent +'('+typs[g_m]+'*)arg'+str(g_m)+'.data'
        if g_m < nargs-1:
          line = line +','
        else:
           line = line +');'
      code(line)
      ENDFOR()

#
# kernel call for direct version
#
    else:
      code('#ifdef VECTORIZE')
      code('#pragma novector')
      FOR2('n','0','(exec_size/SIMD_VEC)*SIMD_VEC','SIMD_VEC')

      #initialize globals
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL:
          if accs[g_m] == OP_INC:
            code('TYP dat'+str(g_m)+'[SIMD_VEC] = {0.0};')
          elif accs[g_m] == OP_MAX:
            code('TYP dat'+str(g_m)+'[SIMD_VEC] = {INFINITY};')
          elif accs[g_m] == OP_MIN:
            code('TYP dat'+str(g_m)+'[SIMD_VEC] = {-INFINITY};')

      code('#pragma simd')
      FOR('i','0','SIMD_VEC')
      line = name+'('
      indent = '\n'+' '*(depth+2)
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          line = line + indent + '&(ptr'+str(g_m)+')['+str(dims[g_m])+' * (n+i)]'
        if maps[g_m] == OP_MAP:
          line = line + indent + '&(ptr'+str(g_m)+')['+str(dims[g_m])+' * map'+str(mapinds[g_m])+'idx]'
        if maps[g_m] == OP_GBL:
          line = line + indent +'&dat'+str(g_m)+'[i]'
        if g_m < nargs-1:
          line = line +','
        else:
           line = line +');'
      code(line)
      ENDFOR()
      #do reductions
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL:
          FOR('i','0','SIMD_VEC')
          if accs[g_m] == OP_INC:
            code('*(TYP*)arg'+str(g_m)+'.data += dat'+str(g_m)+'[i];')
          elif accs[g_m] == OP_MAX:
            code('*(TYP*)arg'+str(g_m)+'.data = MAX(*(TYP*)arg'+str(g_m)+'.data,dat'+str(g_m)+'[i]);')
          elif accs[g_m] == OP_MIN:
            code('*(TYP*)arg'+str(g_m)+'.data = MIN(*(TYP*)arg'+str(g_m)+'.data,dat'+str(g_m)+'[i]);')
          ENDFOR()
      ENDFOR()

      comm('remainder')
      FOR ('n','(exec_size/SIMD_VEC)*SIMD_VEC','exec_size')
      depth = depth -2
      code('#else')
      FOR('n','0','exec_size')
      depth = depth -2
      code('#endif')
      depth = depth +2
      line = name+'('
      indent = '\n'+' '*(depth+2)
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          line = line + indent + '&(ptr'+str(g_m)+')['+str(dims[g_m])+'*n]'
        if maps[g_m] == OP_GBL:
          line = line + indent +'('+typs[g_m]+'*)arg'+str(g_m)+'.data'
        if g_m < nargs-1:
          line = line +','
        else:
           line = line +');'
      code(line)
      ENDFOR()
    ENDIF()
    code('')

    #zero set size issues
    if ninds>0:
      IF('exec_size == 0 || exec_size == set->core_size')
      code('op_mpi_wait_all(nargs, args);')
      ENDIF()

#
# combine reduction data from multiple OpenMP threads
#
    comm(' combine reduction data')
    for g_m in range(0,nargs):
      if maps[g_m]==OP_GBL and accs[g_m]<>OP_READ:
        code('op_mpi_reduce(&ARG,('+typs[g_m]+'*)ARG.data);')

    code('op_mpi_set_dirtybit(nargs, args);')
    code('')

#
# update kernel record
#

    comm(' update kernel record')
    code('op_timers_core(&cpu_t2, &wall_t2);')
    code('OP_kernels[' +str(nk)+ '].name      = name;')
    code('OP_kernels[' +str(nk)+ '].count    += 1;')
    code('OP_kernels[' +str(nk)+ '].time     += wall_t2 - wall_t1;')

    if ninds == 0:
      line = 'OP_kernels['+str(nk)+'].transfer += (float)set->size *'

      for g_m in range (0,nargs):
        if maps[g_m]<>OP_GBL:
          if accs[g_m]==OP_READ:
            code(line+' ARG.size;')
          else:
            code(line+' ARG.size * 2.0f;')
    else:
      names = []
      for g_m in range(0,ninds):
        mult=''
        if indaccs[g_m] <> OP_WRITE and indaccs[g_m] <> OP_READ:
          mult = ' * 2.0f'
        if not var[invinds[g_m]] in names:
          code('OP_kernels['+str(nk)+'].transfer += (float)set->size * arg'+str(invinds[g_m])+'.size'+mult+';')
          names = names + [var[invinds[g_m]]]
      for g_m in range(0,nargs):
        mult=''
        if accs[g_m] <> OP_WRITE and accs[g_m] <> OP_READ:
          mult = ' * 2.0f'
        if not var[g_m] in names:
          names = names + [var[invinds[g_m]]]
          if maps[g_m] == OP_ID:
            code('OP_kernels['+str(nk)+'].transfer += (float)set->size * arg'+str(g_m)+'.size'+mult+';')
          elif maps[g_m] == OP_GBL:
            code('OP_kernels['+str(nk)+'].transfer += (float)set->size * arg'+str(g_m)+'.size'+mult+';')
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
            k = k + [mapnames[g_m]]
            code('OP_kernels['+str(nk)+'].transfer += (float)set->size * arg'+str(invinds[inds[g_m]-1])+'.map->dim * 4.0f;')

    depth -= 2
    code('}')


##########################################################################
#  output individual kernel file
##########################################################################
    fid = open(name+'_veckernel.cpp','w')
    date = datetime.datetime.now()
    #fid.write('//\n// auto-generated by op2.py on '+date.strftime("%Y-%m-%d %H:%M")+'\n//\n\n')
    fid.write('//\n// auto-generated by op2.py\n//\n\n')
    fid.write(file_text)
    fid.close()

# end of main kernel call loop


##########################################################################
#  output one master kernel file
##########################################################################

  file_text =''
  comm(' header                 ')
  code('#include "op_lib_cpp.h"       ')
  code('#define double_ALIGN 128')
  code('#define float_ALIGN 64')
  code('#define int_ALIGN 64')
  code('#ifdef VECTORIZE')
  code('#define SIMD_VEC 4')
  code('#define ALIGNED_double __attribute__((aligned(double_ALIGN)))')
  code('#define ALIGNED_float __attribute__((aligned(float_ALIGN)))')
  code('#define ALIGNED_int __attribute__((aligned(int_ALIGN)))')
  code('#else')
  code('#define ALIGNED_double')
  code('#define ALIGNED_float')
  code('#define ALIGNED_int')
  code('#endif')
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

  comm(' user kernel files')

  for nk in range(0,len(kernels)):
    code('#include "'+kernels[nk]['name']+'_veckernel.cpp"')
  master = master.split('.')[0]
  fid = open(master.split('.')[0]+'_veckernels.cpp','w')
  fid.write('//\n// auto-generated by op2.py\n//\n\n')
  fid.write(file_text)
  fid.close()
