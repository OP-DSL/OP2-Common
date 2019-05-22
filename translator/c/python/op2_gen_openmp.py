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
    file_text +=prefix+'//'+line+'\n'

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
  prefix = ' '*depth
  file_text += prefix+rep(text,g_m)+'\n'

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


def op2_gen_openmp(master, date, consts, kernels):

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
      if maps[i] == OP_GBL and accs[i] <> OP_READ and accs[i] <> OP_WRITE:
        j = i
    reduct = j >= 0
    print name, reduct
##########################################################################
#  start with OpenMP kernel function
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
      code('#include "../'+decl_filepath+'"')

    comm('')
    comm(' x86 kernel function')

    if FORTRAN:
      code('subroutine op_x86_'+name+'(')
    elif CPP:
      code('void op_x86_'+name+'(')

    depth = 2

    if ninds>0:
      if FORTRAN:
        code('integer(4)  blockIdx,')
      elif CPP:
        code('int    blockIdx,')

    for g_m in range(0,ninds):
      if FORTRAN:
        code('INDTYP *ind_ARG,')
      elif CPP:
        code('INDTYP *ind_ARG,')

    if ninds>0:
      if FORTRAN:
        code('int   *ind_map,')
        code('short *arg_map,')
      elif CPP:
        code('int   *ind_map,')
        code('short *arg_map,')

    for g_m in range (0,nargs):
      if maps[g_m]==OP_GBL and accs[g_m] == OP_READ:
        # declared const for performance
        if FORTRAN:
          code('const TYP *ARG,')
        elif CPP:
          code('const TYP *ARG,')
      elif maps[g_m]==OP_ID and ninds>0:
        if FORTRAN:
          code('ARG,')
        elif CPP:
          code('TYP *ARG,')
      elif maps[g_m]==OP_GBL or maps[g_m]==OP_ID:
        if FORTRAN:
          code('ARG,')
        elif CPP:
          code('TYP *ARG,')

    if ninds>0:
      if FORTRAN:
        code('ind_arg_sizes,')
        code('ind_arg_offs, ')
        code('block_offset, ')
        code('blkmap,       ')
        code('offset,       ')
        code('nelems,       ')
        code('ncolors,      ')
        code('colors,       ')
        code('set_size) {    ')
        code('')
      elif CPP:
        code('int   *ind_arg_sizes,')
        code('int   *ind_arg_offs, ')
        code('int    block_offset, ')
        code('int   *blkmap,       ')
        code('int   *offset,       ')
        code('int   *nelems,       ')
        code('int   *ncolors,      ')
        code('int   *colors,       ')
        code('int   set_size) {    ')
        code('')
    else:
      if FORTRAN:
        code('start, finish )')
      elif CPP:
        code('int  start, int  finish ) {')
      code('')

    for g_m in range (0,nargs):
      if maps[g_m]==OP_MAP and accs[g_m]==OP_INC:
        code('TYP  ARG_l[DIM];')

    for m in range (1,ninds+1):
      g_m = m-1
      v = [int(inds[i]==m) for i in range(len(inds))]
      v_i = [vectorised[i] for i in range(len(inds)) if inds[i] == m]
      if sum(v)>1 and sum(v_i)>0:
        if indaccs[m-1] == OP_INC:
          ind = int(max([idxs[i] for i in range(len(inds)) if inds[i]==m])) + 1
          code('INDTYP *ARG_vec['+str(ind)+'] = {'); depth += 2;
          for n in range(0,nargs):
            if inds[n] == m:
              g_m = n
              code('ARG_l,')
          depth -= 2
          code('};')
        else:
          ind = int(max([idxs[i] for i in range(len(inds)) if inds[i]==m])) + 1
          if indaccs[m-1] == OP_READ:
            code('const INDTYP *ARG_vec['+str(ind)+'];')
          else:
            code('INDTYP *ARG_vec['+str(ind)+'];')
#
# lengthy code for general case with indirection
#
    if ninds>0:
      code('')
      for g_m in range (0,ninds):
        code('int  *ind_ARG_map, ind_ARG_size;')
      for g_m in range (0,ninds):
        code('INDTYP *ind_ARG_s;')

      if FORTRAN:
        code('integer(4) :: nelem, offset_b, blockId')
        code('character :: shared[64000]')
      elif CPP:
        code('int    nelem, offset_b;')
        code('')
        code('char shared[128000];')

      code('')
      IF('0==0')
      code('')
      comm(' get sizes and shift pointers and direct-mapped data')
      code('')
      code('int blockId = blkmap[blockIdx + block_offset];')
      code('nelem    = nelems[blockId];')
      code('offset_b = offset[blockId];')
      code('')

      for g_m in range (0,ninds):
        code('ind_ARG_size = ind_arg_sizes['+str(g_m)+'+blockId*'+ str(ninds)+'];')
      code('')
      for m in range (1,ninds+1):
        g_m = m-1
        c = [i for i in range(len(inds)) if inds[i]==m]
        code('ind_ARG_map = &ind_map['+str(cumulative_indirect_index[c[0]])+\
        '*set_size] + ind_arg_offs['+str(m-1)+'+blockId*'+str(ninds)+'];')

      code('')
      comm(' set shared memory pointers')
      code('int nbytes = 0;')

      for g_m in range(0,ninds):
        code('ind_ARG_s = (INDTYP *) &shared[nbytes];')
        if g_m < ninds-1:
          code('nbytes += ROUND_UP(ind_ARG_size*sizeof(INDTYP)*INDDIM);')
      ENDIF()
      code('')
      comm(' copy indirect datasets into shared memory or zero increment')
      code('')

      for g_m in range(0,ninds):
        if indaccs[g_m]==OP_READ or indaccs[g_m]==OP_RW or indaccs[g_m]==OP_INC:
          FOR('n','0','INDARG_size')
          FOR('d','0','INDDIM')
          if indaccs[g_m]==OP_READ or indaccs[g_m]==OP_RW:
            code('INDARG_s[d+n*INDDIM] = INDARG[d+INDARG_map[n]*INDDIM];')
            code('')
          elif indaccs[g_m]==OP_INC:
            code('INDARG_s[d+n*INDDIM] = ZERO_INDTYP;')
          ENDFOR()
          ENDFOR()

      code('')
      comm(' process set elements')
      code('')

      if ind_inc:
        FOR('n','0','nelem')
        comm(' initialise local variables            ')
        for g_m in range(0,nargs):
          if maps[g_m]==OP_MAP and accs[g_m]==OP_INC:
            FOR('d','0','DIM')
            code('ARG_l[d] = ZERO_TYP;')
            ENDFOR()
      else:
        FOR('n','0','nelem')

#
# simple alternative when no indirection
#
    else:
      comm(' process set elements')
      FOR('n','start','finish')

#
# kernel call
#
    # xxx: array of pointers for non-locals
    for m in range(1,ninds+1):
      s = [i for i in range(len(inds)) if inds[i]==m]
      if sum(s)>1:
        if indaccs[m-1] <> OP_INC:
          code('')
          ctr = 0
          for n in range(0,nargs):
            if inds[n] == m and vectorised[n]:
              code('arg'+str(m-1)+'_vec['+str(ctr)+'] = ind_arg'+\
              str(inds[n]-1)+'_s+arg_map['+str(cumulative_indirect_index[n])+\
              '*set_size+n+offset_b]*'+str(dims[n])+';')
              ctr = ctr+1

    code('')
    comm(' user-supplied kernel call')

    line = name+'('
    prefix = ' '*len(name)
    a = 0 #only apply indentation if its not the 0th argument
    indent =''
    for m in range (0, nargs):
      if a > 0:
        indent = '     '+' '*len(name)

      if maps[m] == OP_GBL:
        line += rep(indent+'ARG,\n',m)
        a = a+1
      elif maps[m]==OP_MAP and  accs[m]==OP_INC and vectorised[m]==0:
        line += rep(indent+'ARG_l,\n',m);
        a = a+1
      elif maps[m]==OP_MAP and vectorised[m]==0:
        line += rep(indent+'ind_arg'+str(inds[m]-1)+'_s+arg_map['+\
        str(cumulative_indirect_index[m])+'*set_size+n+offset_b]*DIM,\n',m)
        a = a+1
      elif maps[m]==OP_MAP and m == 0:
        line += rep(indent+'ARG_vec,'+'\n',inds[m]-1)
        a = a+1
      elif maps[m]==OP_MAP and m>0 and vectorised[m] <> vectorised[m-1]: #xxx:vector
        line += rep(indent+'ARG_vec,'+'\n',inds[m]-1)
        a = a+1
      elif maps[m]==OP_MAP and m>0 and vectorised[m] == vectorised[m-1]:
        line = line
        a = a+1
      elif maps[m]==OP_ID:
        if ninds>0:
          line += rep(indent+'ARG+(n+offset_b)*DIM,'+'\n',m)
          a = a+1
        else:
          line += rep(indent+'ARG+n*DIM,'+'\n',m)
          a = a+1
      else:
        print 'internal error 1 '

    code(line[0:-2]+');') #remove final ',' and \n

#
# updating for indirect kernels ...
#
    if ninds>0:
      if ind_inc:
        code('')
        comm(' store local variables            ')

        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and accs[g_m] == OP_INC:
            code('int ARG_map = arg_map['+ str(cumulative_indirect_index[g_m])+\
                '*set_size+n+offset_b];')
        code('')

        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and accs[g_m] == OP_INC:
            FOR('d','0','DIM')
            code('ind_arg'+str(inds[g_m]-1)+'_s[d+ARG_map*DIM] += ARG_l[d];')
            ENDFOR()

      ENDFOR()

      s = [i for i in range(1,ninds+1) if indaccs[i-1]<> OP_READ]

      if len(s)>0 and max(s)>0:
        code('')
        comm(' apply pointered write/increment')

      for g_m in range(0,ninds):
        if indaccs[g_m]==OP_WRITE or indaccs[g_m]==OP_RW or indaccs[g_m]==OP_INC:
          FOR('n','0','INDARG_size')
          FOR('d','0','INDDIM')
          if indaccs[g_m]==OP_WRITE or indaccs[g_m]==OP_RW:
            code('INDARG[d+INDARG_map[n]*INDDIM] = INDARG_s[d+n*INDDIM];')
          elif indaccs[g_m]==OP_INC:
            code('INDARG[d+INDARG_map[n]*INDDIM] += INDARG_s[d+n*INDDIM];')
          ENDFOR()
          ENDFOR()
#
# ... and direct kernels
#
    else:
      depth -= 2
      code('}')

#
# global reduction
#
    depth -= 2
    code('}')
    code('')

##########################################################################
# then C++ stub function
##########################################################################

    code('')
    comm(' host stub function          ')
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
        if (optflags[g_m] == 1):
          argtyp = 'op_opt_arg_dat(arg'+str(first)+'.opt, '
        else:
          argtyp = 'op_arg_dat('

        FOR('v','1',str(sum(v)))
        code('args['+str(g_m)+' + v] = '+argtyp+'arg'+str(first)+'.dat, v, arg'+\
        str(first)+'.map, DIM, "TYP", '+accsstring[accs[g_m]-1]+');')
        ENDFOR()
        code('')
      elif vectorised[g_m]>0:
        pass
      else:
        code('args['+str(g_m)+'] = ARG;')

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
# start timing
#
    code('')
    comm(' initialise timers')
    code('double cpu_t1, cpu_t2, wall_t1, wall_t2;')
    code('op_timers_core(&cpu_t1, &wall_t1);')
    code('')

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
        if maps[g_m]==OP_GBL and accs[g_m]<>OP_READ and accs[g_m]<>OP_WRITE:
          code('TYP ARG_l[DIM+64*64];')
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
      code('op_x86_'+name+'( blockIdx,')

      for m in range(1,ninds+1):
        g_m = invinds[m-1]
        code('(TYP *)ARG.data,')

      code('Plan->ind_map,')
      code('Plan->loc_map,')

      for m in range(0,nargs):
        g_m = m
        if inds[m]==0 and maps[m] == OP_GBL and accs[m] <> OP_READ and accs[m] <> OP_WRITE:
          code('&ARG_l[64*omp_get_thread_num()],')
        elif inds[m]==0:
          code('(TYP *)ARG.data,')

      code('Plan->ind_sizes,')
      code('Plan->ind_offs,')
      code('block_offset,')
      code('Plan->blkmap,')
      code('Plan->offset,')
      code('Plan->nelems,')
      code('Plan->nthrcol,')
      code('Plan->thrcol,')
      code('set_size);')
      ENDFOR()
      code('')

      if reduct:
        comm(' combine reduction data')
        IF('col == Plan->ncolors_owned-1')
        for m in range(0,nargs):
          if maps[m] == OP_GBL and accs[m] <> OP_READ and accs[m] <> OP_WRITE:
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
      code('op_x86_'+name+'(')

      for g_m in range(0,nargs):
        indent = ''
        if maps[g_m]==OP_GBL and accs[g_m] <> OP_READ and accs[g_m] <> OP_WRITE:
          code(indent+'ARG_l + thr*64,')
        else:
          code(indent+'(TYP *) ARG.data,')

      code('start, finish );')
      ENDFOR()

    if ninds>0:
      code('op_timing_realloc('+str(nk)+');')
      code('OP_kernels['+str(nk)+'].transfer  += Plan->transfer; ')
      code('OP_kernels['+str(nk)+'].transfer2 += Plan->transfer2;')

    ENDIF()
    code('')

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
    code('op_timing_realloc('+str(nk)+');')
    code('OP_kernels[' +str(nk)+ '].name      = name;')
    code('OP_kernels[' +str(nk)+ '].count    += 1;')
    code('OP_kernels[' +str(nk)+ '].time     += wall_t2 - wall_t1;')

    if ninds == 0:
      line = 'OP_kernels['+str(nk)+'].transfer += (float)set->size *'

      for g_m in range (0,nargs):
        if optflags[g_m]==1:
          IF('ARG.opt')
        if maps[g_m]<>OP_GBL:
          if accs[g_m]==OP_READ:
            code(line+' ARG.size;')
          else:
            code(line+' ARG.size * 2.0f;')
        if optflags[g_m]==1:
          ENDIF()
    depth -= 2
    code('}')


##########################################################################
#  output individual kernel file
##########################################################################
    if not os.path.exists('openmp'):
        os.makedirs('openmp')
    fid = open('openmp/'+name+'_kernel.cpp','w')
    date = datetime.datetime.now()
    fid.write('//\n// auto-generated by op2.py on '+date.strftime("%Y-%m-%d %H:%M")+'\n//\n\n')
    fid.write(file_text)
    fid.close()

# end of main kernel call loop


##########################################################################
#  output one master kernel file
##########################################################################

  file_text =''

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
  code('')

  comm(' header                 ')

  if os.path.exists('./user_types.h'):
    code('#include "../user_types.h"')
  code('#include "op_lib_cpp.h"       ')
  code('')

  comm(' user kernel files')

  for nk in range(0,len(kernels)):
    code('#include "'+kernels[nk]['name']+'_kernel.cpp"')
  master = master.split('.')[0]
  fid = open('openmp/'+master.split('.')[0]+'_kernels.cpp','w')
  fid.write('//\n// auto-generated by op2.py on '+date.strftime("%Y-%m-%d %H:%M")+'\n//\n\n')
  fid.write(file_text)
  fid.close()



