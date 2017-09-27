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
import datetime

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


def op2_gen_openmp_simple(master, date, consts, kernels):

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
    decl_filename = kernels[nk]['decl_filename']

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
      new_mapnames = []
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
            new_mapnames = new_mapnames+[mapnames[m]]*int(-1*int(idxs[m]))
            new_soaflags = new_soaflags+[soaflags[m]]*int(-1*int(idxs[m]))
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
            new_mapnames = new_mapnames+[mapnames[m]]
            new_accs = new_accs+[int(accs[m])]
            new_soaflags = new_soaflags+[soaflags[m]]
            new_idxs = new_idxs+[int(idxs[m])]
            new_inds = new_inds+[inds[m]]
            new_vars = new_vars+[var[m]]
            new_typs = new_typs+[typs[m]]
            vectorised = vectorised+[0]
      dims = new_dims
      maps = new_maps
      mapnames = new_mapnames
      accs = new_accs
      idxs = new_idxs
      inds = new_inds
      var = new_vars
      typs = new_typs
      soaflags = new_soaflags;
      nargs = len(vectorised);
      mapinds = [0]*nargs
      for i in range(0,nargs):
        mapinds[i] = i
        for j in range(0,i):
          if (maps[i] == OP_MAP) and (mapnames[i] == mapnames[j]) and (idxs[i] == idxs[j]):
            mapinds[i] = mapinds[j]

      for i in range(1,ninds+1):
        for index in range(0,len(inds)+1):
          if inds[index] == i:
            invinds[i-1] = index
            break
      invmapinds = invinds[:]
      for i in range(0,ninds):
        for j in range(0,i):
          if (mapnames[invinds[i]] == mapnames[invinds[j]]):
            invmapinds[i] = invmapinds[j]
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
      if maps[i] == OP_GBL and accs[i] <> OP_READ and accs[i] <> OP_WRITE:
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

    if FORTRAN:
      code('include '+name+'.inc')
    elif CPP:
      code('#include "'+decl_filename+'"')

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
      if maps[g_m]==OP_GBL and accs[g_m] <> OP_READ:
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
      code('int set_size = op_mpi_halo_exchanges(set, nargs, args);')

#
# direct bit
#
    else:
      code('')
      IF('OP_diags>2')
      code('printf(" kernel routine w/o indirection:  '+ name + '");')
      ENDIF()
      code('')
      code('op_mpi_halo_exchanges(set, nargs, args);')

#
# set number of threads in x86 execution and create arrays for reduction
#

    if reduct or ninds==0:
      comm(' set number of threads')
      code('#ifdef _OPENMP')
      code('  int nthreads = omp_get_max_threads();')
      code('#else')
      code('  int nthreads = 1;')
      code('#endif')

    if reduct:
      code('')
      comm(' allocate and initialise arrays for global reduction')
      for g_m in range(0,nargs):
        if maps[g_m]==OP_GBL and accs[g_m]<>OP_READ and accs[g_m] <> OP_WRITE:
          code('TYP ARG_l[nthreads*64];')
          FOR('thr','0','nthreads')
          if accs[g_m]==OP_INC:
            FOR('d','0','DIM')
            code('ARG_l[d+thr*64]=ZERO_TYP;')
            ENDFOR()
          else:
            FOR('d','0','DIM')
            code('ARG_l[d+thr*64]=ARGh[d];')
            ENDFOR()
          ENDFOR()

    code('')
    IF('set->size >0')
    code('')

#
# kernel call for indirect version
#
    if ninds>0:
      code('op_plan *Plan = op_plan_get_stage_upload(name,set,part_size,nargs,args,ninds,inds,OP_STAGE_ALL,0);')
      code('')
      comm(' execute plan')
      code('int block_offset = 0;')
      FOR('col','0','Plan->ncolors')
      IF('col==Plan->ncolors_core')
      code('op_mpi_wait_all(nargs, args);')
      ENDIF()
      code('int nblocks = Plan->ncolblk[col];')
      code('')
      code('#pragma omp parallel for')
      FOR('blockIdx','0','nblocks')
      code('int blockId  = Plan->blkmap[blockIdx + block_offset];')
      code('int nelem    = Plan->nelems[blockId];')
      code('int offset_b = Plan->offset[blockId];')
      FOR('n','offset_b','offset_b+nelem')
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapinds[g_m] in k):
            k = k + [mapinds[g_m]]
            code('int map'+str(mapinds[g_m])+'idx = arg'+str(invmapinds[inds[g_m]-1])+\
              '.map_data[n * arg'+str(invmapinds[inds[g_m]-1])+'.map->dim + '+str(idxs[g_m])+'];')
      code('')
      for g_m in range (0,nargs):
        u = [i for i in range(0,len(unique_args)) if unique_args[i]-1 == g_m]
        if len(u) > 0 and vectorised[g_m] > 0:
          if accs[g_m] == OP_READ:
            line = 'const TYP* ARG_vec[] = {\n'
          else:
            line = 'TYP* ARG_vec[] = {\n'

          v = [int(vectorised[i] == vectorised[g_m]) for i in range(0,len(vectorised))]
          first = [i for i in range(0,len(v)) if v[i] == 1]
          first = first[0]

          indent = ' '*(depth+2)
          for k in range(0,sum(v)):
            line = line + indent + ' &((TYP*)arg'+str(first)+'.data)[DIM * map'+str(mapinds[g_m+k])+'idx],\n'
          line = line[:-2]+'};'
          code(line)
      code('')
      line = name+'('
      indent = '\n'+' '*(depth+2)
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          line = line + indent + '&(('+typs[g_m]+'*)arg'+str(g_m)+'.data)['+str(dims[g_m])+' * n]'
        if maps[g_m] == OP_MAP:
          if vectorised[g_m]:
            if g_m+1 in unique_args:
                line = line + indent + 'arg'+str(g_m)+'_vec'
          else:
            line = line + indent + '&(('+typs[g_m]+'*)arg'+str(invinds[inds[g_m]-1])+'.data)['+str(dims[g_m])+' * map'+str(mapinds[g_m])+'idx]'
        if maps[g_m] == OP_GBL:
          if accs[g_m] <> OP_READ and accs[g_m] <> OP_WRITE:
            line = line + indent +'&arg'+str(g_m)+'_l[64*omp_get_thread_num()]'
          else:
            line = line + indent +'('+typs[g_m]+'*)arg'+str(g_m)+'.data'
        if g_m < nargs-1:
          if g_m+1 in unique_args and not g_m+1 == unique_args[-1]:
            line = line +','
        else:
           line = line +');'
      code(line)
      ENDFOR()
      ENDFOR()
      code('')

      if reduct:
        comm(' combine reduction data')
        IF('col == Plan->ncolors_owned-1')
        for m in range(0,nargs):
          if maps[m] == OP_GBL and accs[m] <> OP_READ:
            FOR('thr','0','nthreads')
            if accs[m]==OP_INC:
              FOR('d','0','DIM')
              code('ARGh[d] += ARG_l[d+thr*64];')
              ENDFOR()
            elif accs[m]==OP_MIN:
              FOR('d','0','DIM')
              code('ARGh[d]  = MIN(ARGh[d],ARG_l[d+thr*64]);')
              ENDFOR()
            elif  accs(m)==OP_MAX:
              FOR('d','0','DIM')
              code('ARGh[d]  = MAX(ARGh[d],ARG_l[d+thr*64]);')
              ENDFOR()
            else:
              error('internal error: invalid reduction option')
            ENDFOR()
        ENDIF()
      code('block_offset += nblocks;');
      ENDIF()

#
# kernel call for direct version
#
    else:
      comm(' execute plan')
      code('#pragma omp parallel for')
      FOR('thr','0','nthreads')
      code('int start  = (set->size* thr)/nthreads;')
      code('int finish = (set->size*(thr+1))/nthreads;')
      FOR('n','start','finish')
      line = name+'('
      indent = '\n'+' '*(depth+2)
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          line = line + indent + '&(('+typs[g_m]+'*)arg'+str(g_m)+'.data)['+str(dims[g_m])+'*n]'
        if maps[g_m] == OP_GBL:
          if accs[g_m] <> OP_READ and accs[g_m] <> OP_WRITE:
            line = line + indent +'&arg'+str(g_m)+'_l[64*omp_get_thread_num()]'
          else:
            line = line + indent +'('+typs[g_m]+'*)arg'+str(g_m)+'.data'
        if g_m < nargs-1:
          line = line +','
        else:
           line = line +');'
      code(line)
      ENDFOR()
      ENDFOR()

    if ninds>0:
      code('OP_kernels['+str(nk)+'].transfer  += Plan->transfer;')
      code('OP_kernels['+str(nk)+'].transfer2 += Plan->transfer2;')

    ENDIF()
    code('')

    #zero set size issues
    if ninds>0:
      IF('set_size == 0 || set_size == set->core_size')
      code('op_mpi_wait_all(nargs, args);')
      ENDIF()

#
# combine reduction data from multiple OpenMP threads, direct version
#
    comm(' combine reduction data')
    for g_m in range(0,nargs):
      if maps[g_m]==OP_GBL and accs[g_m]<>OP_READ and accs[g_m] <> OP_WRITE and ninds==0:
        FOR('thr','0','nthreads')
        if accs[g_m]==OP_INC:
          FOR('d','0','DIM')
          code('ARGh[d] += ARG_l[d+thr*64];')
          ENDFOR()
        elif accs[g_m]==OP_MIN:
          FOR('d','0','DIM')
          code('ARGh[d]  = MIN(ARGh[d],ARG_l[d+thr*64]);')
          ENDFOR()
        elif accs[g_m]==OP_MAX:
          FOR('d','0','DIM')
          code('ARGh[d]  = MAX(ARGh[d],ARG_l[d+thr*64]);')
          ENDFOR()
        else:
          print 'internal error: invalid reduction option'
        ENDFOR()
      if maps[g_m]==OP_GBL and accs[g_m]<>OP_READ:
        code('op_mpi_reduce(&ARG,ARGh);')

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
          if accs[g_m]==OP_READ or accs[g_m]==OP_WRITE:
            code(line+' ARG.size;')
          else:
            code(line+' ARG.size * 2.0f;')

    depth -= 2
    code('}')


##########################################################################
#  output individual kernel file
##########################################################################
    fid = open(name+'_kernel.cpp','w')
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
    code('#include "'+kernels[nk]['name']+'_kernel.cpp"')
  master = master.split('.')[0]
  fid = open(master.split('.')[0]+'_kernels.cpp','w')
  fid.write('//\n// auto-generated by op2.py\n//\n\n')
  fid.write(file_text)
  fid.close()




