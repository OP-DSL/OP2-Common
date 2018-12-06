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

def op2_gen_cuda(master, date, consts, kernels, sets):

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
      if maps[i] == OP_MAP and accs[i] == OP_RW:
        j = i
    ind_rw= j >= 0
    if (ind_rw):
      print 'Error: indirect OP_RW not supported by op2_gen_cuda code generator'
      sys.exit(2);


    j = -1
    for i in range(0,nargs):
      if maps[i] == OP_GBL and accs[i] <> OP_READ and accs[i] <> OP_WRITE:
        j = i
    reduct = j >= 0

##########################################################################
#  start with CUDA kernel function
##########################################################################

    FORTRAN = 0;
    CPP     = 1;
    g_m = 0;
    file_text = ''
    depth = 0

    comm('user function')

    code('__device__')
    if FORTRAN:
      code('include '+name+'.inc')
    elif CPP:
      code('#include "../'+name+'.h"')

    comm('')
    comm(' CUDA kernel function')

    if FORTRAN:
      code('subroutine op_cuda_'+name+'(')
    elif CPP:
      code('__global__ void op_cuda_'+name+'(')

    depth = 2

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
          code('TYP  *ARG,')
      elif maps[g_m]==OP_GBL or maps[g_m]==OP_ID:
        if FORTRAN:
          code('ARG,')
        elif CPP:
          code('TYP *ARG,')

    if ninds>0:
      if FORTRAN:
        code('int   *ind_arg_sizes,')
        code('int   *ind_arg_offs, ')
        code('int    block_offset, ')
        code('int   *blkmap,       ')
        code('int   *offset,       ')
        code('int   *nelems,       ')
        code('int   *ncolors,      ')
        code('int   *colors,       ')
        code('int   nblocks,       ')
        code('int   set_size) {    ')
      if CPP:
        code('int   *ind_arg_sizes,')
        code('int   *ind_arg_offs, ')
        code('int    block_offset, ')
        code('int   *blkmap,       ')
        code('int   *offset,       ')
        code('int   *nelems,       ')
        code('int   *ncolors,      ')
        code('int   *colors,       ')
        code('int   nblocks,       ')
        code('int   set_size) {    ')
    else:
      code('int   offset_s,    ')
      code('int   set_size ) {')
      code('')

    for g_m in range(0,nargs):
      if maps[g_m]==OP_GBL and accs[g_m]<>OP_READ and accs[g_m]<>OP_WRITE:
        code('TYP ARG_l[DIM];')
        if accs[g_m] == OP_INC:
          FOR('d','0','DIM')
          code('ARG_l[d]=ZERO_TYP;')
          ENDFOR()
        else:
          FOR('d','0','DIM')
          code('ARG_l[d]=ARG[d+blockIdx.x*DIM];')
          ENDFOR()
      elif maps[g_m]==OP_MAP and accs[g_m]==OP_INC:
        code('TYP ARG_l[DIM];')
      elif (ninds==0 and maps[g_m]==OP_ID and dims[g_m]<>'1') and not(soaflags[g_m]):
        code('TYP ARG_l[DIM];')

    for m in range (1,ninds+1):
      g_m = m -1
      v = [int(inds[i]==m) for i in range(len(inds))]
      v_i = [vectorised[i] for i in range(len(inds)) if inds[i] == m]
      if sum(v)>1 and sum(v_i)>0: #check this sum(v_i)
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
        code('__shared__  int  *ind_ARG_map, ind_ARG_size;')
      for g_m in range (0,ninds):
        code('__shared__  INDTYP *ind_ARG_s;')

      if ind_inc:
        code('__shared__ int    nelems2, ncolor;')

      code('__shared__ int    nelem, offset_b;')
      code('')
      code('extern __shared__ char shared[];')
      code('')
      IF('blockIdx.x+blockIdx.y*gridDim.x >= nblocks')
      code('return;')
      ENDIF()
      IF('threadIdx.x==0')
      code('')
      comm('get sizes and shift pointers and direct-mapped data')
      code('')
      code('int blockId = blkmap[blockIdx.x + blockIdx.y*gridDim.x  + block_offset];')
      code('')
      code('nelem    = nelems[blockId];')
      code('offset_b = offset[blockId];')
      code('')

      if ind_inc:
        code('nelems2  = blockDim.x*(1+(nelem-1)/blockDim.x);')
        code('ncolor   = ncolors[blockId];')
        code('')

      for g_m in range (0,ninds):
        code('ind_ARG_size = ind_arg_sizes['+str(g_m)+'+blockId*'+ str(ninds)+'];')

      code('')
      
      for m in range (1,ninds+1):
        g_m = m - 1
        c = [i for i in range(len(inds)) if inds[i]==m]
        code('ind_ARG_map = &ind_map['+str(cumulative_indirect_index[c[0]])+\
        '*set_size] + ind_arg_offs['+str(m-1)+'+blockId*'+str(ninds)+'];')

      code('')
      comm('set shared memory pointers')
      code('int nbytes = 0;')

      for g_m in range(0,ninds):
        code('ind_ARG_s = (INDTYP *) &shared[nbytes];')
        if g_m < ninds-1:
          code('nbytes    += ROUND_UP(ind_ARG_size*sizeof(INDTYP)*INDDIM);')

      ENDIF()
      code('__syncthreads(); // make sure all of above completed')
      code('')
      comm('copy indirect datasets into shared memory or zero increment')
      code('')

      for m in range(0,ninds):
        g_m = m
        if indaccs[m]==OP_READ or indaccs[m]==OP_RW or indaccs[m]==OP_INC:
          FOR_INC('n','threadIdx.x','ind_ARG_size*INDDIM','blockDim.x')
          if indaccs[m]==OP_READ or indaccs[m]==OP_RW:
            code('ind_arg'+str(m)+'_s[n] = ind_arg'+str(m)+'[n%'+inddims[m]+
            '+ind_arg'+str(m)+'_map[n/'+inddims[m]+']*'+inddims[m]+'];')
            code('')
          elif indaccs[m]==OP_INC:
            code('ind_ARG_s[n] = ZERO_INDTYP;')
          ENDFOR()

      code('')
      code('__syncthreads();')
      comm('process set elements')
      code('')

      if ind_inc:
        FOR_INC('n','threadIdx.x','nelems2','blockDim.x')
        code('int col2 = -1;')
        IF('n<nelem')
        comm('initialise local variables')

        for g_m in range(0,nargs):
          if maps[g_m]==OP_MAP and accs[g_m]==OP_INC:
            FOR('d','0','DIM')
            code('ARG_l[d] = ZERO_TYP;')
            ENDFOR()
      else:
        FOR_INC('n','threadIdx.x','nelem','blockDim.x')

#
# simple alternative when no indirection
#
    else:
      use_shared = 0;
      for m in range(0,nargs):
        if maps[m]<>OP_GBL and dims[m]<>'1':
          use_shared = 1

      if use_shared:
        code('int   tid = threadIdx.x%OP_WARPSIZE;')
        code('')
        code('extern __shared__ char shared[];')
        code('char *arg_s = shared + offset_s*(threadIdx.x/OP_WARPSIZE);')

      code('')
      comm('process set elements')
      FOR_INC('n','threadIdx.x+blockIdx.x*blockDim.x','set_size','blockDim.x*gridDim.x')

      if use_shared:
        code('int offset = n - tid;')
        code('int nelems = MIN(OP_WARPSIZE,set_size-offset);')
        comm('copy data into shared memory, then into local')

      for m in range(0,nargs):
        g_m = m
        if (maps[m]<>OP_GBL and accs[m]<>OP_WRITE and dims[m]<>'1') and not(soaflags[m]):
          FOR('m','0','DIM')
          code('((TYP *)arg_s)[tid+m*nelems] = ARG[tid+m*nelems+offset*DIM];')
          ENDFOR()
          code('')
          FOR('m','0','DIM')
          code('ARG_l[m] = ((TYP *)arg_s)[m+tid*DIM];')
          ENDFOR()
          code('')



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
    comm('user-supplied kernel call')

    line = name+'('
    prefix = ' '*len(name)
    a = 0 #only apply indentation if its not the 0th argument
    indent =''
    for m in range (0, nargs):
      if a > 0:
        indent = '     '+' '*len(name)

      if maps[m] == OP_GBL:
        if accs[m] == OP_READ or accs[m] == OP_WRITE:
          line += rep(indent+'ARG,\n',m)
        else:
          line += rep(indent+'ARG_l,\n',m);
        a =a+1
      elif maps[m]==OP_MAP and  accs[m]==OP_INC and vectorised[m]==0:
        line += rep(indent+'ARG_l,\n',m)
        a =a+1
      elif maps[m]==OP_MAP and vectorised[m]==0:
        line += rep(indent+'ind_arg'+str(inds[m]-1)+'_s+arg_map['+\
        str(cumulative_indirect_index[m])+'*set_size+n+offset_b]*DIM,'+'\n',m)
        a =a+1
      elif maps[m]==OP_MAP and m == 0:
        line += rep(indent+'ARG_vec,'+'\n',inds[m]-1)
        a =a+1
      elif maps[m]==OP_MAP and m>0 and vectorised[m] <> vectorised[m-1]: #xxx:vector
        line += rep(indent+'ARG_vec,'+'\n',inds[m]-1)
        a =a+1
      elif maps[m]==OP_MAP and m>0 and vectorised[m] == vectorised[m-1]:
        line = line
        a =a+1
      elif maps[m]==OP_ID:
        if ninds>0:
          if soaflags[m]:
            line += rep(indent+'ARG+(n+offset_b),\n',m)
          else:
            line += rep(indent+'ARG+(n+offset_b)*DIM,\n',m)
          a =a+1
        else:
          if dims[m] == '1' or soaflags[m]:
            line += rep(indent+'ARG+n,\n',m)
          else:
            line += rep(indent+'ARG_l,\n',m)
          a =a+1
      else:
        print 'internal error 1 '

    code(line[0:-2]+');') #remove final ',' and \n

#
# updating for indirect kernels ...
#
    if ninds>0:
      if ind_inc:
        code('col2 = colors[n+offset_b];')
        ENDIF()
        code('')
        comm('store local variables')
        code('')

        for g_m in range(0,nargs):
          if maps[g_m]==OP_MAP and accs[g_m]==OP_INC:
            code('int ARG_map;')

        IF('col2>=0')

        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and accs[g_m] == OP_INC:
            code('ARG_map = arg_map['+str(cumulative_indirect_index[g_m])+'*set_size+n+offset_b];')

        ENDIF()
        code('')
        FOR('col','0','ncolor')
        IF('col2==col')

        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and accs[g_m] == OP_INC:
            FOR('d','0','DIM')
            code('ind_arg'+str(inds[g_m]-1)+'_s[d+ARG_map*DIM] += ARG_l[d];')
            ENDFOR()

        ENDFOR()
        code('__syncthreads();')
        ENDFOR()
      ENDFOR()

      s = [i for i in range(1,ninds+1) if indaccs[i-1]<> OP_READ]

      if len(s)>0 and max(s)>0:
        code('')
        comm('apply pointered write/increment')

      for g_m in range(0,ninds):
        if indaccs[g_m]==OP_WRITE or indaccs[g_m]==OP_RW or indaccs[g_m]==OP_INC:
          FOR_INC('n','threadIdx.x','INDARG_size*INDDIM','blockDim.x')
          if indaccs[g_m]==OP_WRITE or indaccs[g_m]==OP_RW:
            code('INDARG[n%INDDIM+INDARG_map[n/INDDIM]*INDDIM] = INDARG_s[n];')
          elif indaccs[g_m]==OP_INC:
            code('INDARG[n%INDDIM+INDARG_map[n/INDDIM]*INDDIM] += INDARG_s[n];')
          ENDFOR()
#
# ... and direct kernels
#
    else:
      if use_shared:
        comm('copy back into shared memory, then to device')
      for m in range(0,nargs):
        g_m = m
        if (maps[m]<>OP_GBL and accs[m]<>OP_READ and dims[m]<>'1') and not(soaflags[m]):
          code('')
          FOR('m','0','DIM')
          code('((TYP *)arg_s)[m+tid*DIM] = ARG_l[m];')
          ENDFOR()
          FOR('m','0','DIM')
          code('ARG[tid+m*nelems+offset*DIM] = ((TYP *)arg_s)[tid+m*nelems];')
          ENDFOR()

      depth -= 2
      code('}')

#
# global reduction
#
    if reduct:
       code('')
       comm('global reductions')
       code('')
       for m in range (0,nargs):
         g_m = m
         if maps[m]==OP_GBL and accs[m]<>OP_READ and accs[m] <> OP_WRITE:
           FOR('d','0','DIM')
           if accs[m]==OP_INC:
             code('op_reduction<OP_INC>(&ARG[d+blockIdx.x*DIM],ARG_l[d]);')
           elif accs[m]==OP_MIN:
             code('op_reduction<OP_MIN>(&ARG[d+blockIdx.x*DIM],ARG_l[d]);')
           elif accs[m]==OP_MAX:
             code('op_reduction<OP_MAX>(&ARG[d+blockIdx.x*DIM],ARG_l[d]);')
           else:
             print 'internal error: invalid reduction option'
             sys.exit(2);
           ENDFOR()
    depth -= 2
    code('}')
    code('')

##########################################################################
# then C++ stub function
##########################################################################

    code('')
    comm('host stub function')
    code('void op_par_loop_'+name+'(char const *name, op_set set,')
    depth += 2

    for m in unique_args:
      g_m = m - 1
      if m == unique_args[len(unique_args)-1]:
        code('op_arg ARG){')
        code('')
      else:
        code('op_arg ARG,')

    for g_m in range (0,nargs):
      if maps[g_m]==OP_GBL:
        code('TYP*ARGh = (TYP *)ARG.data;')

    code('int nargs = '+str(nargs)+';')
    code('op_arg args['+str(nargs)+'];')
    code('')

    #print vectorised

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
      code('')
      code('int    ninds   = '+str(ninds)+';')
      line = '  int    inds['+str(nargs)+'] = {'
      for m in range(0,nargs):
        line += str(inds[m]-1)+','
      code(line[:-1]+'};')
      code('')

      IF('OP_diags>2')
      code('printf(" kernel routine with indirection: '+name+'\\n");')
      ENDIF()

      code('')
      comm('get plan')
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
      code('op_mpi_halo_exchanges_cuda(set, nargs, args);')
#
# start timing
#
    code('')
    comm(' initialise timers')
    code('double cpu_t1, cpu_t2, wall_t1, wall_t2;')
    code('op_timers_core(&cpu_t1, &wall_t1);')
    code('')

    IF('set->size > 0')
    code('')
    code('op_timing_realloc('+str(nk)+');')
    code('OP_kernels[' +str(nk)+ '].name      = name;')
    code('OP_kernels[' +str(nk)+ '].count    += 1;')
    code('')
    if any_soa:
      code('int op2_stride_internal = set->size + set->exec_size + set->nonexec_size;')
      #code('op_decl_const_char(1, "int", sizeof(int), (char *)&op2_stride, "op2_stride");')
      code('cutilSafeCall(cudaMemcpyToSymbol(op2_stride , &op2_stride_internal, sizeof(int)));');
      code('')

#
# kernel call for indirect version
#
    if ninds>0:
      code('op_plan *Plan = op_plan_get(name,set,part_size,nargs,args,ninds,inds);')
      code('')


#
# transfer constants
#
    g = [i for i in range(0,nargs) if maps[i] == OP_GBL and (accs[i] == OP_READ or accs[i] == OP_WRITE)]
    if len(g)>0:
      comm('transfer constants to GPU')
      code('int consts_bytes = 0;')
      for m in range(0,nargs):
        g_m = m
        if maps[m]==OP_GBL and (accs[m]==OP_READ or accs[m]==OP_WRITE):
          code('consts_bytes += ROUND_UP(DIM*sizeof(TYP));')

      code('reallocConstArrays(consts_bytes);')
      code('consts_bytes = 0;')

      for m in range(0,nargs):
        if maps[m]==OP_GBL and (accs[m]==OP_READ  or accs[m]==OP_WRITE):
          g_m = m
          code('ARG.data   = OP_consts_h + consts_bytes;')
          code('ARG.data_d = OP_consts_d + consts_bytes;')
          FOR('d','0','DIM')
          code('((TYP *)ARG.data)[d] = ARGh[d];')
          ENDFOR()
          code('consts_bytes += ROUND_UP(DIM*sizeof(TYP));')
      code('mvConstArraysToDevice(consts_bytes);')
      code('')


#
# transfer global reduction initial data
#

    if ninds == 0:
      comm('set CUDA execution parameters')
      code('#ifdef OP_BLOCK_SIZE_'+str(nk))
      code('  int nthread = OP_BLOCK_SIZE_'+str(nk)+';')
      code('#else')
      comm('  int nthread = OP_block_size;')
      code('  int nthread = 128;')
      code('#endif')
      code('')
      code('int nblocks = 200;')
      code('')

    if reduct:
      comm('transfer global reduction data to GPU')
      if ninds>0:
        code('int maxblocks = 0;')
        FOR('col','0','Plan->ncolors')
        code('maxblocks = MAX(maxblocks,Plan->ncolblk[col]);')
        ENDFOR()
      else:
        code('int maxblocks = nblocks;')

      code('int reduct_bytes = 0;')
      code('int reduct_size  = 0;')

      for g_m in range(0,nargs):
        if maps[g_m]==OP_GBL and accs[g_m]<>OP_READ and accs[g_m]<>OP_WRITE:
          code('reduct_bytes += ROUND_UP(maxblocks*DIM*sizeof(TYP));')
          code('reduct_size   = MAX(reduct_size,sizeof(TYP));')

      code('reallocReductArrays(reduct_bytes);')
      code('reduct_bytes = 0;')

      for g_m in range(0,nargs):
        if maps[g_m]==OP_GBL and accs[g_m]<>OP_READ and accs[g_m]<>OP_WRITE:
          code('ARG.data   = OP_reduct_h + reduct_bytes;')
          code('ARG.data_d = OP_reduct_d + reduct_bytes;')
          FOR('b','0','maxblocks')
          FOR('d','0','DIM')
          if accs[g_m]==OP_INC:
            code('((TYP *)ARG.data)[d+b*DIM] = ZERO_TYP;')
          else:
            code('((TYP *)ARG.data)[d+b*DIM] = ARGh[d];')
          ENDFOR()
          ENDFOR()
          code('reduct_bytes += ROUND_UP(maxblocks*DIM*sizeof(TYP));')
      code('mvReductArraysToDevice(reduct_bytes);')
      code('')

#
# kernel call for indirect version
#
    if ninds>0:
      comm('execute plan')
      code('')
      code('int block_offset = 0;')
      FOR('col','0','Plan->ncolors')
      IF('col==Plan->ncolors_core')
      code('op_mpi_wait_all_cuda(nargs, args);')
      ENDIF()
      code('#ifdef OP_BLOCK_SIZE_'+str(nk))
      code('int nthread = OP_BLOCK_SIZE_'+str(nk)+';')
      code('#else')
      code('int nthread = OP_block_size;')
      code('#endif')
      code('')
      code('dim3 nblocks = dim3(Plan->ncolblk[col] >= (1<<16) ? 65535 : Plan->ncolblk[col],')
      code('Plan->ncolblk[col] >= (1<<16) ? (Plan->ncolblk[col]-1)/65535+1: 1, 1);')
      IF('Plan->ncolblk[col] > 0')

      if reduct:
        code('int nshared = MAX(Plan->nshared,reduct_size*nthread);')
      else:
        code('int nshared = Plan->nsharedCol[col];')

      code('op_cuda_'+name+'<<<nblocks,nthread,nshared>>>(')

      for m in range(1,ninds+1):
        g_m = invinds[m-1]
        code('(TYP *)ARG.data_d,')

      code('Plan->ind_map,')
      code('Plan->loc_map,')

      for g_m in range(0,nargs):
        if inds[g_m]==0:
          code('(TYP*)ARG.data_d,')


      code('Plan->ind_sizes,')
      code('Plan->ind_offs,')
      code('block_offset,')
      code('Plan->blkmap,')
      code('Plan->offset,')
      code('Plan->nelems,')
      code('Plan->nthrcol,')
      code('Plan->thrcol,')
      code('Plan->ncolblk[col],')
      code('set_size);')
      code('')
      if reduct:
        comm('transfer global reduction data back to CPU')
        IF('col == Plan->ncolors_owned-1')
        code('mvReductArraysToHost(reduct_bytes);')
        ENDIF()

      ENDFOR()
      code('block_offset += Plan->ncolblk[col];')
      ENDIF()
#
# kernel call for direct version
#
    else:
      comm('work out shared memory requirements per element')
      code('')
      code('int nshared = 0;')

      for g_m in range(0,nargs):
         if maps[g_m]<>OP_GBL and dims[g_m]<>'1':
           code('nshared = MAX(nshared,sizeof(TYP)*DIM);')

      code('')
      comm('execute plan')
      code('int offset_s = nshared*OP_WARPSIZE;')
      code('')

      if reduct:
        code('nshared = MAX(nshared*nthread,reduct_size*nthread);')
      else:
        code('nshared = nshared*nthread;')

      code('op_cuda_'+name+'<<<nblocks,nthread,nshared>>>(')

      indent = '  '#*(len(name)+42)
      for g_m in range(0,nargs):
        if g_m > 0:
          code(indent+'(TYP *) ARG.data_d,')
        else:
          code(indent+'(TYP *) ARG.data_d,')

      code(indent+'offset_s,')
      code(indent+'set->size );')

    if ninds>0:
      code('OP_kernels['+str(nk)+'].transfer  += Plan->transfer;')
      code('OP_kernels['+str(nk)+'].transfer2 += Plan->transfer2;')


#
# transfer global reduction initial data
#
    if reduct:
      if ninds == 0:
        comm('transfer global reduction data back to CPU')
        code('mvReductArraysToHost(reduct_bytes);')

      for m in range(0,nargs):
        g_m = m
        if maps[m]==OP_GBL and accs[m]<>OP_READ and accs[m]<>OP_WRITE:
          FOR('b','0','maxblocks')
          FOR('d','0','DIM')
          if accs[m]==OP_INC:
            code('ARGh[d] = ARGh[d] + ((TYP *)ARG.data)[d+b*DIM];')
          elif accs[m]==OP_MIN:
            code('ARGh[d] = MIN(ARGh[d],((TYP *)ARG.data)[d+b*DIM]);')
          elif accs[m]==OP_MAX:
            code('ARGh[d] = MAX(ARGh[d],((TYP *)ARG.data)[d+b*DIM]);')
          ENDFOR()
          ENDFOR()

          code('ARG.data = (char *)ARGh;')
          code('op_mpi_reduce(&ARG,ARGh);')

    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and accs[g_m] == OP_WRITE:
        code('mvConstArraysToHost(consts_bytes);')
        break

    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and accs[g_m] == OP_WRITE:
        FOR('d','0','DIM')
        code('ARGh[d] = ((TYP *)ARG.data)[d];') 
        ENDFOR()
        code('ARG.data = (char *)ARGh;')
        code('op_mpi_reduce(&ARG,ARGh);')

    ENDIF()
    code('op_mpi_set_dirtybit_cuda(nargs, args);')

#
# update kernel record
#

    comm('update kernel record')
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

    depth = depth - 2
    code('}')


##########################################################################
#  output individual kernel file
##########################################################################
    if not os.path.exists('cuda'):
        os.makedirs('cuda')
    fid = open('cuda/'+name+'_kernel.cu','w')
    date = datetime.datetime.now()
    fid.write('//\n// auto-generated by op2.py on '+date.strftime("%Y-%m-%d %H:%M")+'\n//\n\n')
    fid.write(file_text)
    fid.close()

# end of main kernel call loop


##########################################################################
#  output one master kernel file
##########################################################################

  file_text = ''

  comm('global constants')

  code('#ifndef MAX_CONST_SIZE')
  code('#define MAX_CONST_SIZE 128')
  code('#endif')
  code('')

  for nc in range (0,len(consts)):
    if consts[nc]['dim']==1:
      code('__constant__ '+consts[nc]['type'][1:-1]+' '+consts[nc]['name']+';')
    else:
      if consts[nc]['dim'] > 0:
        num = str(consts[nc]['dim'])
      else:
        num = 'MAX_CONST_SIZE'

      code('__constant__ '+consts[nc]['type'][1:-1]+' '+consts[nc]['name']+'['+num+'];')
  code('')

  comm('header')
  if os.path.exists('./user_types.h'):
    code('#ifndef OP_FUN_PREFIX\n#define OP_FUN_PREFIX __host__ __device__\n#endif')
    code('#include "../user_types.h"')
  code('#include "op_lib_cpp.h"')
  code('#include "op_cuda_rt_support.h"')
  code('#include "op_cuda_reduction.h"')
  code('')

  # if any_soa:
  #   code('__constant__ int op2_stride;')
  #   code('')
  #   code('#define OP2_STRIDE(arr, idx) arr[op2_stride*(idx)]')

  code('')
  code('void op_decl_const_char(int dim, char const *type,')
  code('int size, char *dat, char const *name){')
  depth = depth + 2

  for nc in range(0,len(consts)):
    IF('!strcmp(name,"'+consts[nc]['name']+'")')
    if consts[nc]['dim'] < 0:
      IF('!strcmp(name,"'+consts[nc]['name']+'") && size>MAX_CONST_SIZE) {')
      code('printf("error: MAX_CONST_SIZE not big enough\n"); exit(1);')
      ENDIF()
    code('cutilSafeCall(cudaMemcpyToSymbol('+consts[nc]['name']+', dat, dim*size));')
    ENDIF()
    code('else ')

  code('{')
  depth = depth + 2
  code('printf("error: unknown const name\\n"); exit(1);')
  ENDIF()


  depth = depth - 2
  code('}')
  code('')
  comm('user kernel files')

  for nk in range(0,len(kernels)):
    file_text = file_text +\
    '#include "'+kernels[nk]['name']+'_kernel.cu"\n'

  master = master.split('.')[0]
  fid = open('cuda/'+master.split('.')[0]+'_kernels.cu','w')
  fid.write('//\n// auto-generated by op2.py on '+date.strftime("%Y-%m-%d %H:%M")+'\n//\n\n')
  fid.write(file_text)
  fid.close()



