##########################################################################
#
# OpenMP code generator
#
# This routine is called by op2 which parses the input files
#
# It produces a file xxx_kernel.cpp for each kernel,
# plus a master kernel file
#
##########################################################################

import re
import glob
import datetime
import op2_gen_common
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

def op2_gen_openmp4(master, date, consts, kernels):

  global dims, idxs, typs, indtyps, inddims
  global FORTRAN, CPP, g_m, file_text, depth

  OP_ID   = 1;  OP_GBL   = 2;  OP_MAP = 3;

  OP_READ = 1;  OP_WRITE = 2;  OP_RW  = 3;
  OP_INC  = 4;  OP_MAX   = 5;  OP_MIN = 6;

  accsstring = ['OP_READ','OP_WRITE','OP_RW','OP_INC','OP_MAX','OP_MIN' ]
  op2_compiler = os.getenv('OP2_COMPILER','0');
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
# set two logicals
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

##########################################################################
#  start with the user kernel function
##########################################################################

    FORTRAN = 0;
    CPP     = 1;
    g_m = 0;
    file_text = ''
    depth = 0

    comm('user function')

    #strides for SoA
    if any_soa:
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
            k = k + [mapnames[g_m]]
            code('int opDat'+str(invinds[inds[g_m]-1])+'_'+name+'_stride_OP2CONSTANT;')
            code('int opDat'+str(invinds[inds[g_m]-1])+'_'+name+'_stride_OP2HOST=-1;')
      dir_soa = -1
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID and ((not dims[g_m].isdigit()) or int(dims[g_m]) > 1):
          code('int direct_'+name+'_stride_OP2CONSTANT;')
          code('int direct_'+name+'_stride_OP2HOST=-1;')
          dir_soa = g_m
          break

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
    f.close()

    kernel_text = op2_gen_common.comment_remover(kernel_text)
    kernel_text = op2_gen_common.remove_trailing_w_space(kernel_text)

    p = re.compile('void\\s+\\b'+name+'\\b')
    i = p.search(kernel_text).start()

    if(i < 0):
      print "\n********"
      print "Error: cannot locate user kernel function name: "+name+" - Aborting code generation"
      exit(2)
    i2 = i

    #i = kernel_text[0:i].rfind('\n') #reverse find
    j = kernel_text[i:].find('{')
    k = op2_gen_common.para_parse(kernel_text, i+j, '{', '}')
    signature_text = kernel_text[i:i+j]
    l = signature_text[0:].find('(')
    head_text = signature_text[0:l] #save function name
    m = op2_gen_common.para_parse(signature_text, 0, '(', ')')
    signature_text = signature_text[l+1:m]
    body_text = kernel_text[i+j+1:k]

    # check for number of arguments
    if len(signature_text.split(',')) != nargs:
        print 'Error parsing user kernel(%s): must have %d arguments' \
              % name, nargs
        return

    for i in range(0,nargs):
        var = signature_text.split(',')[i].strip()
        if soaflags[i]:
          var = var.replace('*','')
          #locate var in body and replace by adding [idx]
          length = len(re.compile('\\s+\\b').split(var))
          var2 = re.compile('\\s+\\b').split(var)[length-1].strip()

          body_text = re.sub('\*'+var2+'(?!\[)', var2+'[0]', body_text)
          body_text = re.sub(r''+var2+'\[([A-Za-z0-9]*)\]'+'', var2+r'[\1*'+op2_gen_common.get_stride_string(i,maps,mapnames,name)+']', body_text)
    for nc in range(0,len(consts)): 
      varname = consts[nc]['name']
      if consts[nc]['dim'] == 1:
        body_text = re.sub(varname+'(?!\w)', varname+'_ompkernel', body_text)
      else:
        body_text = re.sub('\*'+varname+'(?!\[)', varname+'[0]', body_text)
        body_text = re.sub(r''+varname+'\[([A-Za-z0-9]*)\]'+'', varname+r'_ompkernel[\1]', body_text)

    kernel_params = [ var.strip() for var in signature_text.split(',')]
    # collect constants used by kernel
    kernel_consts = []
    for nc in range(0,len(consts)):
      if body_text.find(consts[nc]['name']+'_ompkernel') != -1:
        kernel_consts.append(nc)

############################################################
#  omp4 function call definition
############################################################
    code('')
    func_call_signaure_text = 'void ' + name + '_omp4_kernel('
    params = ''
    indent = '\n' + '  '
    k = []
    for g_m in range(0, nargs):
      if maps[g_m] == OP_GBL:
        params += indent + rep('TYP *ARG,',g_m)
      if maps[g_m] == OP_MAP and (not invmapinds[inds[g_m]-1] in k):
        k = k + [invmapinds[inds[g_m]-1]]
        params += indent +  'int *map'+str(mapinds[g_m])+','
      if maps[g_m] == OP_ID:
        params += indent + rep('TYP *data'+str(g_m)+',', g_m)
    for m in range(1,ninds+1):
      g_m = invinds[m-1]
      params += indent + rep('TYP *data'+str(g_m)+',', g_m)
    if ninds>0:
      # add indirect kernel specific params to kernel func call 
      params += indent + 'int *col_reord,' + indent + 'int set_size1,' + indent + 'int start,' + indent + 'int end,'
    else:
      # add direct kernel specific params to kernel func call 
      params += indent + 'int count,'
    params += indent + 'int num_teams,' + indent + 'int nthread'
    #add strides for SoA to params
    if any_soa:
      indent = ','+indent
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
            k = k + [mapnames[g_m]]
            params += indent + 'int opDat'+str(invinds[inds[g_m]-1])+'_'+name+'_stride_OP2CONSTANT'
      
      if dir_soa<>-1:
        params += indent + 'int direct_'+name+'_stride_OP2CONSTANT'
    code(func_call_signaure_text+params+');')

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

    for g_m in range (0,nargs):
      if maps[g_m]==OP_GBL: #and accs[g_m] <> OP_READ:
        code('TYP*ARGh = (TYP *)ARG.data;')

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
# start timing
#
    code('')
    comm(' initialise timers')
    code('double cpu_t1, cpu_t2, wall_t1, wall_t2;')
    code('op_timing_realloc('+str(nk)+');')
    code('op_timers_core(&cpu_t1, &wall_t1);')
    code('OP_kernels[' +str(nk)+ '].name      = name;')
    code('OP_kernels[' +str(nk)+ '].count    += 1;')
    code('')

#
#   indirect bits
#
    if ninds>0:
      code('int  ninds   = '+str(ninds)+';')
      line = 'int  inds['+str(nargs)+'] = {'
      for m in range(0,nargs):
        line += str(inds[m]-1)+','
      code(line[:-1]+'};')
      code('')

      IF('OP_diags>2')
      code('printf(" kernel routine with indirection: '+name+'\\n");')
      ENDIF()
      code('')
      comm(' get plan')
      code('int set_size = op_mpi_halo_exchanges_cuda(set, nargs, args);')

#
# direct bit
#
    else:
      code('')
      IF('OP_diags>2')
      code('printf(" kernel routine w/o indirection:  '+ name + '");')
      ENDIF()
      code('')
      code('op_mpi_halo_exchanges_cuda(set, nargs, args);')

#
# get part and block size
#
    code('')
    code('#ifdef OP_PART_SIZE_'+ str(nk))
    code('  int part_size = OP_PART_SIZE_'+str(nk)+';')
    code('#else')
    code('  int part_size = OP_part_size;')
    code('#endif')
    code('#ifdef OP_BLOCK_SIZE_'+ str(nk))
    code('  int nthread = OP_BLOCK_SIZE_'+str(nk)+';')
    code('#else')
    code('  int nthread = OP_block_size;')
    code('#endif')

    code('')
    for g_m in range(0,nargs):
      if maps[g_m]==OP_GBL: #and accs[g_m]<>OP_READ:
        if not dims[g_m].isdigit() or int(dims[g_m]) > 1:
          print 'ERROR: OpenACC does not support multi-dimensional variables'
          exit(-1)
        code('TYP ARG_l = ARGh[0];')

    if ninds > 0:
      code('')
      code('int ncolors = 0;')
    code('')
    IF('set->size >0')
    #managing constants
    if any_soa:
      code('')
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
            k = k + [mapnames[g_m]]
            IF('(OP_kernels[' +str(nk)+ '].count==1) || (opDat'+str(invinds[inds[g_m]-1])+'_'+name+'_stride_OP2HOST != getSetSizeFromOpArg(&arg'+str(g_m)+'))')
            code('opDat'+str(invinds[inds[g_m]-1])+'_'+name+'_stride_OP2HOST = getSetSizeFromOpArg(&arg'+str(g_m)+');')
            code('opDat'+str(invinds[inds[g_m]-1])+'_'+name+'_stride_OP2CONSTANT = opDat'+str(invinds[inds[g_m]-1])+'_'+name+'_stride_OP2HOST;')
            ENDIF()
      if dir_soa<>-1:
          IF('(OP_kernels[' +str(nk)+ '].count==1) || (direct_'+name+'_stride_OP2HOST != getSetSizeFromOpArg(&arg'+str(dir_soa)+'))')
          code('direct_'+name+'_stride_OP2HOST = getSetSizeFromOpArg(&arg'+str(dir_soa)+');')
          code('direct_'+name+'_stride_OP2CONSTANT = direct_'+name+'_stride_OP2HOST;')
          ENDIF()
    code('')
    comm('Set up typed device pointers for OpenMP')
    if nmaps > 0:
      k = []
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (not invmapinds[inds[g_m]-1] in k):
          k = k + [invmapinds[inds[g_m]-1]]
          code('int *map'+str(mapinds[g_m])+' = arg'+str(invmapinds[inds[g_m]-1])+'.map_data_d;')

    code('')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code(typs[g_m]+'* data'+str(g_m)+' = ('+typs[g_m]+'*)arg'+str(g_m)+'.data_d;')

    for m in range(1,ninds+1):
      g_m = invinds[m-1]
      code('TYP *data'+str(g_m)+' = (TYP *)ARG.data_d;')
    
#
# prepare kernel params for indirect version
#
    if ninds>0:
      code('')
      code('op_plan *Plan = op_plan_get_stage(name,set,part_size,nargs,args,ninds,inds,OP_COLOR2);')
      code('ncolors = Plan->ncolors;')
      code('int *col_reord = Plan->col_reord;')
      code('int set_size1 = set->size + set->exec_size;')
      code('')
      comm(' execute plan')
      FOR('col','0','Plan->ncolors')
      IF('col==1')
      code('op_mpi_wait_all_cuda(nargs, args);')
      ENDIF()
      code('int start = Plan->col_offsets[0][col];')
      code('int end = Plan->col_offsets[0][col+1];')
      code('')
#
# kernel function call
#
    indent = '\n' + ' ' * (depth+2)
    call_params = ','.join([ indent + re.sub(r'\*arg(\d+)',r'&arg\1_l',param.strip().split(' ')[-1]) for param in params.split(',')])
    call_params = call_params.replace('*','')
    # set params for indirect version
    if ninds>0:
      call_params = call_params.replace('num_teams','part_size!=0?(end-start-1)/part_size+1:(end-start-1)/nthread')
    # set params for direct version
    else:
      call_params = re.sub('count','set->size',call_params);
      call_params = call_params.replace('num_teams','part_size!=0?(set->size-1)/part_size+1:(set->size-1)/nthread') 
    code(func_call_signaure_text.split(' ')[-1]+call_params+');')
    code('')

    if ninds>0:
      if reduct:
        comm(' combine reduction data')
        IF('col == Plan->ncolors_owned-1')
        for g_m in range(0,nargs):
          if maps[g_m] == OP_GBL and accs[g_m] <> OP_READ:
            if accs[g_m]==OP_INC:
              code('ARGh[0] = ARG_l;')
            elif accs[g_m]==OP_MIN:
              code('ARGh[0]  = MIN(ARGh[0],ARG_l);')
            elif  accs[g_m]==OP_MAX:
              code('ARGh[0]  = MAX(ARGh[0],ARG_l);')
            else:
              error('internal error: invalid reduction option')
        ENDIF()
      ENDFOR()
      code('OP_kernels['+str(nk)+'].transfer  += Plan->transfer;')
      code('OP_kernels['+str(nk)+'].transfer2 += Plan->transfer2;')

    ENDIF()
    code('')

    #zero set size issues
    if ninds>0:
      IF('set_size == 0 || set_size == set->core_size || ncolors == 1')
      code('op_mpi_wait_all_cuda(nargs, args);')
      ENDIF()

#
# combine reduction data from multiple OpenMP threads
#
    comm(' combine reduction data')
    for g_m in range(0,nargs):
      if maps[g_m]==OP_GBL and accs[g_m]<>OP_READ:
        if ninds==0: #direct version only
          if accs[g_m]==OP_INC:
            code('ARGh[0] = ARG_l;')
          elif accs[g_m]==OP_MIN:
            code('ARGh[0]  = MIN(ARGh[0],ARG_l);')
          elif accs[g_m]==OP_MAX:
            code('ARGh[0]  = MAX(ARGh[0],ARG_l);')
          else:
            print 'internal error: invalid reduction option'
        if typs[g_m] == 'double': #need for both direct and indirect
          code('op_mpi_reduce_double(&ARG,ARGh);')
        elif typs[g_m] == 'float':
          code('op_mpi_reduce_float(&ARG,ARGh);')
        elif typs[g_m] == 'int':
          code('op_mpi_reduce_int(&ARG,ARGh);')
        else:
          print 'Type '+typs[g_m]+' not supported in OpenMP4 code generator, please add it'
          exit(-1)


    code('op_mpi_set_dirtybit_cuda(nargs, args);')
    code('')

#
# update kernel record
#

    code('if (OP_diags>1) deviceSync();')
    comm(' update kernel record')
    code('op_timers_core(&cpu_t2, &wall_t2);')
    code('OP_kernels[' +str(nk)+ '].time     += wall_t2 - wall_t1;')

    if ninds == 0:
      line = 'OP_kernels['+str(nk)+'].transfer += (float)set->size *'

      for g_m in range (0,nargs):
        if maps[g_m]<>OP_GBL:
          if accs[g_m]==OP_READ or accs[g_m]==OP_WRITE:
            code(line+' ARG.size;')
          else:
            code(line+' ARG.size * 2.0f;')

    depth -= 2
    code('}')

##########################################################################
#  output individual kernel file
##########################################################################
    fid = open(name+'_omp4kernel.cpp','w')
    date = datetime.datetime.now()
    fid.write('//\n// auto-generated by op2.py\n//\n\n')
    fid.write(file_text)
    fid.close()

##############################################################
# generate ****_omp4kernel_func.cpp
##############################################################
    file_text = ''

    code(func_call_signaure_text+params+'){')
    code('')
    depth += 2
    for g_m in range(0, nargs):
      if maps[g_m] == OP_GBL:
        code('TYP ARG_l = *ARG;')
    line = '#pragma omp target teams'
    if op2_compiler == 'clang':
      line +=' distribute parallel for schedule(static,1)\\\n' + (depth+2)*' '
    line +=' num_teams(num_teams) thread_limit(nthread) '
    map_clause = 'is_device_ptr('
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        map_clause += 'data'+str(g_m)+','
    if map_clause != 'is_device_ptr(' and map_clause != 'map(to:':
      map_clause = map_clause[:-1]+')'
      line += map_clause
    # mapping global consts
    if len(kernel_consts) != 0:
      line += ' \\\n' + (depth+2)*' ' + 'map(to:'
      for nc in kernel_consts:
        line += ' ' + consts[nc]['name']+'_ompkernel,'
        if consts[nc]['dim'] != 1:
          if consts[nc]['dim'] > 0:
            num = str(consts[nc]['dim'])
          else:
            num = 'MAX_CONST_SIZE'
          line = line[:-1] + '[:'+ num +'],'
      line = line[:-1]+')'
    # prepare reduction
    reduction_string = ''
    reduction_mapping = ''
    if reduct:
      reduction_mapping ='\\\n'+(depth+2)*' '+ 'map(tofrom:'
      for g_m in range(0,nargs):
        if maps[g_m]==OP_GBL and accs[g_m]<>OP_READ:
          if accs[g_m] == OP_INC:
            reduction_string += ' reduction(+:arg%d_l)' % g_m
            reduction_mapping += ' arg%d_l,' % g_m
          if accs[g_m] == OP_MIN:
            reduction_string += ' reduction(min:arg%d_l)' % g_m
            reduction_mapping += ' arg%d_l,' % g_m
          if accs[g_m] == OP_MAX:
            reduction_string += ' reduction(max:arg%d_l)' % g_m
            reduction_mapping += ' arg%d_l,' % g_m
      reduction_mapping = reduction_mapping[0:-1]+')' 
#
# map extra pointers for indirect version
#
    if ninds>0:
      line += '\\\n'+(depth+2)*' '+'is_device_ptr(col_reord,'
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not invmapinds[inds[g_m]-1] in k):
            k = k + [invmapinds[inds[g_m]-1]]
            line = line + 'map'+str(mapinds[g_m])+','
      for m in range(1,ninds+1):
        g_m = invinds[m-1]
        line = line + 'data'+str(g_m)+','
      line = line[:-1]+')'
#
# write omp pragma
#
    code(line + reduction_mapping + reduction_string)
    if op2_compiler != 'clang':
      line = '#pragma omp distribute parallel for schedule(static,1)'
      code(line + reduction_string)
#
# start for loop indirect version
#
    if ninds>0:
      FOR('e','start','end')
      code('int n_op = col_reord[e];')
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapinds[g_m] in k):
            k = k + [mapinds[g_m]]
            code('int map'+str(mapinds[g_m])+'idx = map'+str(invmapinds[inds[g_m]-1])+\
              '[n_op + set_size1 * '+str(idxs[g_m])+'];')
#
# direct version
#
    else:
      FOR('n_op','0','count')
#
# write inlined kernel function
#
    comm('variable mapping')
    for g_m in range(0,nargs):
      line = kernel_params[g_m] + ' = '
      if maps[g_m] == OP_ID:
        if soaflags[g_m]:
          line += '&data%d[n_op]' % g_m
        else:
          line += '&data'+str(g_m)+'['+str(dims[g_m])+'*n_op]'
      if maps[g_m] == OP_MAP:
        if soaflags[g_m]:
          line += '&data'+str(invinds[inds[g_m]-1])+'[map'+str(mapinds[g_m])+'idx]'
        else:
          line += '&data'+str(invinds[inds[g_m]-1])+'['+str(dims[g_m])+' * map'+str(mapinds[g_m])+'idx]'
      if maps[g_m] == OP_GBL:
        line += '&arg%d_l' % g_m
      line += ';'
      code(line)
    
    code('')
    comm('inline function')
    indent = ' ' * (depth-2)
    inline_body_text = ''
    for line in body_text.split('\n'):
      inline_body_text += indent+line+'\n'
    code(inline_body_text)
    comm('end inline func')

    ENDFOR()
    code('')
    # end kernel function
    for g_m in range(0, nargs):
      if maps[g_m] == OP_GBL:
        code('*ARG = ARG_l;')
    depth -= 2;
    code('}')


##########################################################################
#  output individual omp4kernel file
##########################################################################
    fid = open(name+'_omp4kernel_func.cpp','w')
    date = datetime.datetime.now()
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
  code('')
  comm(' user kernel files')

  for nk in range(0,len(kernels)):
    code('#include "'+kernels[nk]['name']+'_omp4kernel.cpp"')
  master = master.split('.')[0]
  fid = open(master.split('.')[0]+'_omp4kernels.cpp','w')
  fid.write('//\n// auto-generated by op2.py\n//\n\n')
  fid.write(file_text)
  fid.close()


##########################################################################
#  output omp4 master kernel file
##########################################################################

  file_text =''
  comm(' header                 ')
  code('#include "op_lib_cpp.h"       ')
  code('')
  comm(' global constants       ')

  for nc in range (0,len(consts)):
    if consts[nc]['dim']==1:
      code(consts[nc]['type'][1:-1]+' '+consts[nc]['name']+'_ompkernel;')
    else:
      if consts[nc]['dim'] > 0:
        num = str(consts[nc]['dim'])
      else:
        num = 'MAX_CONST_SIZE'
      code(consts[nc]['type'][1:-1]+' '+consts[nc]['name']+'_ompkernel['+num+'];')
  code('')
  code('void op_decl_const_char(int dim, char const *type,')
  code('  int size, char *dat, char const *name){')
  indent = ' ' * ( 2+ depth)
  line = '  ' 
  for nc in range (0,len(consts)):
    varname = consts[nc]['name']
    if nc > 0:
        line += ' else '
    line += 'if(!strcmp(name, "%s")) {\n' %varname + indent + 2*' ' + 'memcpy('
    if consts[nc]['dim']==1:
      line += '&'
    line += varname+ '_ompkernel, dat, dim*size);\n' + indent + '#pragma omp target enter data map(to:'+varname+'_ompkernel'
    if consts[nc]['dim'] !=1:
      line += '[:%s]' % str(consts[nc]['dim']) if consts[nc]['dim'] > 0 else 'MAX_CONST_SIZE'
    line += ')\n'+indent + '}'
  code(line)
  code('}')

  comm(' user kernel files')

  for nk in range(0,len(kernels)):
    code('#include "'+kernels[nk]['name']+'_omp4kernel_func.cpp"')
  master = master.split('.')[0]
  fid = open(master.split('.')[0]+'_omp4kernel_funcs.cpp','w')
  fid.write('//\n// auto-generated by op2.py\n//\n\n')
  fid.write(file_text)
  fid.close()


