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

def op2_gen_openacc(master, date, consts, kernels):

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
    j = -1
    for i in range(0,nargs):
      if maps[i] == OP_MAP and accs[i] == OP_INC:
        j = i
    ind_inc = j >= 0

    j = -1
    for i in range(0,nargs):
      if maps[i] == OP_GBL and accs[i] != OP_READ and accs[i] != OP_WRITE:
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

    file_name = decl_filepath
    f = open(file_name, 'r')
    kernel_text = f.read()
    f.close()

    comm('user function')

    if CPP:
      includes = op2_gen_common.extract_includes(kernel_text)
      if len(includes) > 0:
        for include in includes:
          code(include)
        code("")

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

    kernel_text = op2_gen_common.comment_remover(kernel_text)
    kernel_text = op2_gen_common.remove_trailing_w_space(kernel_text)

    p = re.compile('void\\s+\\b'+name+'\\b')
    i = p.search(kernel_text).start()

    if(i < 0):
      print("\n********")
      print("Error: cannot locate user kernel function name: "+name+" - Aborting code generation")
      exit(2)
    i2 = i

    #i = kernel_text[0:i].rfind('\n') #reverse find
    j = kernel_text[i:].find('{')
    k = op2_gen_common.para_parse(kernel_text, i+j, '{', '}')
    signature_text = kernel_text[i:i+j]
    l = signature_text[0:].find('(')
    head_text = signature_text[0:l].strip() #save function name
    m = op2_gen_common.para_parse(signature_text, 0, '(', ')')
    signature_text = signature_text[l+1:m]
    body_text = kernel_text[i+j+1:k]

    # check for number of arguments
    if len(signature_text.split(',')) != nargs_novec:
        print('Error parsing user kernel('+name+'): must have '+str(nargs_novec)+' arguments')
        return

    for i in range(0,nargs_novec):
        var = signature_text.split(',')[i].strip()
        if kernels[nk]['soaflags'][i]:
          var = var.replace('*','')
          #locate var in body and replace by adding [idx]
          length = len(re.compile('\\s+\\b').split(var))
          var2 = re.compile('\\s+\\b').split(var)[length-1].strip()

          if int(kernels[nk]['idxs'][i]) < 0 and kernels[nk]['maps'][i] == OP_MAP:
            body_text = re.sub(r'\b'+var2+'(\[[^\]]\])\[([\\s\+\*A-Za-z0-9]*)\]'+'', var2+r'\1[(\2)*'+ \
                    op2_gen_common.get_stride_string(unique_args[i]-1,maps,mapnames,name)+']', body_text)
          else:
            body_text = re.sub('\*\\b'+var2+'\\b\\s*(?!\[)', var2+'[0]', body_text)
            body_text = re.sub(r'\b'+var2+'\[([\\s\+\*A-Za-z0-9]*)\]'+'', var2+r'[(\1)*'+ \
                    op2_gen_common.get_stride_string(unique_args[i]-1,maps,mapnames,name)+']', body_text)

    head_text += "_openacc"
    signature_text = '//#pragma acc routine\ninline ' + head_text + '( '+signature_text + ') {'
    file_text += signature_text + body_text + '}\n'

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
        code('op_arg <ARG>){');
        code('')
      else:
        code('op_arg <ARG>,')

    for g_m in range (0,nargs):
      if maps[g_m]==OP_GBL: #and accs[g_m] <> OP_READ:
        code('<TYP>*<ARG>h = (<TYP> *)<ARG>.data;')

    code('int nargs = '+str(nargs)+';')
    code('op_arg args['+str(nargs)+'];')
    code('')

    for g_m in range (0,nargs):
      u = [i for i in range(0,len(unique_args)) if unique_args[i]-1 == g_m]
      if len(u) > 0 and vectorised[g_m] > 0:
        code('<ARG>.idx = 0;')
        code('args['+str(g_m)+'] = <ARG>;')

        v = [int(vectorised[i] == vectorised[g_m]) for i in range(0,len(vectorised))]
        first = [i for i in range(0,len(v)) if v[i] == 1]
        first = first[0]
        if (optflags[g_m] == 1):
          argtyp = 'op_opt_arg_dat(arg'+str(first)+'.opt, '
        else:
          argtyp = 'op_arg_dat('

        FOR('v','1',str(sum(v)))
        code('args['+str(g_m)+' + v] = '+argtyp+'arg'+str(first)+'.dat, v, arg'+\
        str(first)+'.map, <DIM>, "<TYP>", '+accsstring[accs[g_m]-1]+');')
        ENDFOR()
        code('')
      elif vectorised[g_m]>0:
        pass
      else:
        code('args['+str(g_m)+'] = <ARG>;')

    if nopts>0:
      code('int optflags = 0;')
      for i in range(0,nargs):
        if optflags[i] == 1:
          IF('args['+str(i)+'].opt')
          code('optflags |= 1<<'+str(optidxs[i])+';')
          ENDIF()
    if nopts > 30:
      print('ERROR: too many optional arguments to store flags in an integer')

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
      code('#ifdef OP_PART_SIZE_'+ str(nk))
      code('  int part_size = OP_PART_SIZE_'+str(nk)+';')
      code('#else')
      code('  int part_size = OP_part_size;')
      code('#endif')
      code('')
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
      code('int set_size = op_mpi_halo_exchanges_cuda(set, nargs, args);')

    code('')
    for g_m in range(0,nargs):
      if maps[g_m]==OP_GBL: #and accs[g_m]<>OP_READ:
        if not dims[g_m].isdigit() or int(dims[g_m]) > 1:
          print('ERROR: OpenACC does not support multi-dimensional op_arg_gbl variables')
          exit(-1)
        code('<TYP> <ARG>_l = <ARG>h[0];')

    if ninds > 0:
      code('')
      code('int ncolors = 0;')
    code('')
    IF('set_size >0')
    code('')
    #managing constants
    if any_soa:
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
            k = k + [mapnames[g_m]]
            IF('(OP_kernels[' +str(nk)+ '].count==1) || (opDat'+str(invinds[inds[g_m]-1])+'_'+name+'_stride_OP2HOST != getSetSizeFromOpArg(&arg'+str(g_m)+'))')
            code('opDat'+str(invinds[inds[g_m]-1])+'_'+name+'_stride_OP2HOST = getSetSizeFromOpArg(&arg'+str(g_m)+');')
            code('opDat'+str(invinds[inds[g_m]-1])+'_'+name+'_stride_OP2CONSTANT = opDat'+str(invinds[inds[g_m]-1])+'_'+name+'_stride_OP2HOST;')
            ENDIF()
      if dir_soa!=-1:
          IF('(OP_kernels[' +str(nk)+ '].count==1) || (direct_'+name+'_stride_OP2HOST != getSetSizeFromOpArg(&arg'+str(dir_soa)+'))')
          code('direct_'+name+'_stride_OP2HOST = getSetSizeFromOpArg(&arg'+str(dir_soa)+');')
          code('direct_'+name+'_stride_OP2CONSTANT = direct_'+name+'_stride_OP2HOST;')
          ENDIF()
    code('')
    comm('Set up typed device pointers for OpenACC')
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
      code('<TYP> *data'+str(g_m)+' = (<TYP> *)<ARG>.data_d;')

#
# kernel call for indirect version
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
#      code('#pragma omp parallel for')
      line = '#pragma acc parallel loop independent deviceptr(col_reord,'
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not invmapinds[inds[g_m]-1] in k):
            k = k + [invmapinds[inds[g_m]-1]]
            line = line + 'map'+str(mapinds[g_m])+','

      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          line = line+'data'+str(g_m)+','
      for m in range(1,ninds+1):
        g_m = invinds[m-1]
        line = line + 'data'+str(g_m)+','
      line = line[:-1]+')'

      if reduct:
        for g_m in range(0,nargs):
          if maps[g_m]==OP_GBL and accs[g_m]!=OP_READ and accs[g_m] != OP_WRITE:
            if accs[g_m] == OP_INC:
              line = line + ' reduction(+:arg'+str(g_m)+'_l)'
            if accs[g_m] == OP_MIN:
              line = line + ' reduction(min:arg'+str(g_m)+'_l)'
            if accs[g_m] == OP_MAX:
              line = line + ' reduction(max:arg'+str(g_m)+'_l)'
      code(line)
      FOR('e','start','end')
      code('int n = col_reord[e];')
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapinds[g_m] in k):
            k = k + [mapinds[g_m]]
            code('int map'+str(mapinds[g_m])+'idx;')
      #do non-optional ones
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapinds[g_m] in k) and (not optflags[g_m]):
            k = k + [mapinds[g_m]]
            code('map'+str(mapinds[g_m])+'idx = map'+str(invmapinds[inds[g_m]-1])+\
              '[n + set_size1 * '+str(idxs[g_m])+'];')
      #do optional ones
      if nmaps > 0:
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapinds[g_m] in k):
            if optflags[g_m]:
              IF('optflags & 1<<'+str(optidxs[g_m]))
            else:
              k = k + [mapinds[g_m]]
            code('map'+str(mapinds[g_m])+'idx = map'+str(invmapinds[inds[g_m]-1])+\
              '[n + set_size1 * '+str(idxs[g_m])+'];')
            if optflags[g_m]:
              ENDIF()

      code('')
      for g_m in range (0,nargs):
        u = [i for i in range(0,len(unique_args)) if unique_args[i]-1 == g_m]
        if len(u) > 0 and vectorised[g_m] > 0:
          if accs[g_m] == OP_READ:
            line = 'const <TYP>* <ARG>_vec[] = {\n'
          else:
            line = '<TYP>* <ARG>_vec[] = {\n'

          v = [int(vectorised[i] == vectorised[g_m]) for i in range(0,len(vectorised))]
          first = [i for i in range(0,len(v)) if v[i] == 1]
          first = first[0]

          indent = ' '*(depth+2)
          for k in range(0,sum(v)):
            if soaflags[g_m]:
              line = line + indent + ' &data'+str(first)+'[map'+str(mapinds[g_m+k])+'idx],\n'
            else:
              line = line + indent + ' &data'+str(first)+'[<DIM> * map'+str(mapinds[g_m+k])+'idx],\n'
          line = line[:-2]+'};'
          code(line)
      code('')
      line = name+'_openacc('
      indent = '\n'+' '*(depth+2)
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          if soaflags[g_m]:
            line = line + indent + '&data'+str(g_m)+'[n]'
          else:
            line = line + indent + '&data'+str(g_m)+'['+str(dims[g_m])+' * n]'
        if maps[g_m] == OP_MAP:
          if vectorised[g_m]:
            if g_m+1 in unique_args:
              line = line + indent + 'arg'+str(g_m)+'_vec'
          else:
            if soaflags[g_m]:
              line = line + indent + '&data'+str(invinds[inds[g_m]-1])+'[map'+str(mapinds[g_m])+'idx]'
            else:
              line = line + indent + '&data'+str(invinds[inds[g_m]-1])+'['+str(dims[g_m])+' * map'+str(mapinds[g_m])+'idx]'
        if maps[g_m] == OP_GBL:
          line = line + indent +'&arg'+str(g_m)+'_l'
        if g_m < nargs-1:
          if g_m+1 in unique_args and not g_m+1 == unique_args[-1]:
            line = line +','
        else:
           line = line +');'
      code(line)
      ENDFOR()
      code('')

      if reduct:
        comm(' combine reduction data')
        IF('col == Plan->ncolors_owned-1')
        for g_m in range(0,nargs):
          if maps[g_m] == OP_GBL and accs[g_m] != OP_READ:
            if accs[g_m]==OP_INC or accs[g_m]==OP_WRITE:
              code('<ARG>h[0] = <ARG>_l;')
            elif accs[g_m]==OP_MIN:
              code('<ARG>h[0]  = MIN(<ARG>h[0],<ARG>_l);')
              ENDFOR()
            elif  accs[g_m]==OP_MAX:
              code('<ARG>h[0]  = MAX(<ARG>h[0],<ARG>_l);')
            else:
              error('internal error: invalid reduction option')
            ENDFOR()
        ENDIF()
      ENDFOR()

#
# kernel call for direct version
#
    else:
      line = '#pragma acc parallel loop independent deviceptr('
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          line = line+'data'+str(g_m)+','
      line = line[:-1]+')'

      if reduct:
        for g_m in range(0,nargs):
          if maps[g_m]==OP_GBL and accs[g_m]!=OP_READ and accs[g_m]!=OP_WRITE:
            if accs[g_m] == OP_INC:
              line = line + ' reduction(+:arg'+str(g_m)+'_l)'
            if accs[g_m] == OP_MIN:
              line = line + ' reduction(min:arg'+str(g_m)+'_l)'
            if accs[g_m] == OP_MAX:
              line = line + ' reduction(max:arg'+str(g_m)+'_l)'
      code(line)
      FOR('n','0','set->size')
      line = name+'_openacc('
      indent = '\n'+' '*(depth+2)
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          if soaflags[g_m]:
            line = line + indent + '&data'+str(g_m)+'[n]'
          else:
            line = line + indent + '&data'+str(g_m)+'['+str(dims[g_m])+'*n]'
        if maps[g_m] == OP_GBL:
          line = line + indent +'&arg'+str(g_m)+'_l'
        if g_m < nargs-1:
          line = line +','
        else:
           line = line +');'
      code(line)
      ENDFOR()

    if ninds>0:
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
      if maps[g_m]==OP_GBL and accs[g_m]!=OP_READ:
        if ninds==0: #direct version only
          if accs[g_m]==OP_INC or accs[g_m]==OP_WRITE:
            code('<ARG>h[0] = <ARG>_l;')
          elif accs[g_m]==OP_MIN:
            code('<ARG>h[0]  = MIN(<ARG>h[0],<ARG>_l);')
          elif accs[g_m]==OP_MAX:
            code('<ARG>h[0]  = MAX(<ARG>h[0],<ARG>_l);')
          else:
            print('internal error: invalid reduction option')
        if typs[g_m] == 'double': #need for both direct and indirect
          code('op_mpi_reduce_double(&<ARG>,<ARG>h);')
        elif typs[g_m] == 'float':
          code('op_mpi_reduce_float(&<ARG>,<ARG>h);')
        elif typs[g_m] == 'int':
          code('op_mpi_reduce_int(&<ARG>,<ARG>h);')
        else:
          print('Type '+typs[g_m]+' not supported in OpenACC code generator, please add it')
          exit(-1)


    code('op_mpi_set_dirtybit_cuda(nargs, args);')
    code('')

#
# update kernel record
#

    comm(' update kernel record')
    code('op_timers_core(&cpu_t2, &wall_t2);')
    code('OP_kernels[' +str(nk)+ '].time     += wall_t2 - wall_t1;')

    if ninds == 0:
      line = 'OP_kernels['+str(nk)+'].transfer += (float)set->size *'

      for g_m in range (0,nargs):
        if optflags[g_m]==1:
          IF('<ARG>.opt')
        if maps[g_m]!=OP_GBL:
          if accs[g_m]==OP_READ:
            code(line+' <ARG>.size;')
          else:
            code(line+' <ARG>.size * 2.0f;')
        if optflags[g_m]==1:
          ENDIF()
    depth -= 2
    code('}')


##########################################################################
#  output individual kernel file
##########################################################################
    if not os.path.exists('openacc'):
        os.makedirs('openacc')
    fid = open('openacc/'+name+'_acckernel.c','w')
    date = datetime.datetime.now()
    fid.write('//\n// auto-generated by op2.py\n//\n\n')
    fid.write(file_text)
    fid.close()

# end of main kernel call loop


##########################################################################
#  output one master kernel file
##########################################################################

  file_text =''

  comm(' global constants       ')

  for nc in range (0,len(consts)):
    if not consts[nc]['user_declared']:
      if consts[nc]['dim']==1:
        code('extern '+consts[nc]['type'][1:-1]+' '+consts[nc]['name']+';')
      else:
        if consts[nc]['dim'].isdigit() and int(consts[nc]['dim']) > 0:
          num = str(consts[nc]['dim'])
        else:
          num = 'MAX_CONST_SIZE'

        code('extern '+consts[nc]['type'][1:-1]+' '+consts[nc]['name']+'['+num+'];')
  code('')

  comm(' header                 ')

  if os.path.exists('./user_types.h'):
    code('#include "../user_types.h"')
  code('#include "op_lib_c.h"       ')
  code('')

  code('void op_decl_const_char(int dim, char const *type,')
  code('int size, char *dat, char const *name){}')

  comm(' user kernel files')

  for nk in range(0,len(kernels)):
    code('#include "'+kernels[nk]['name']+'_acckernel.c"')
  master = master.split('.')[0]
  fid = open('openacc/'+master.split('.')[0]+'_acckernels.c','w')
  fid.write('//\n// auto-generated by op2.py\n//\n\n')
  fid.write(file_text)
  fid.close()




