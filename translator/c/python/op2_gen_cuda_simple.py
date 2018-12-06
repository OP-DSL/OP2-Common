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
import glob
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

def op2_gen_cuda_simple(master, date, consts, kernels,sets, macro_defs):

  global dims, idxs, typs, indtyps, inddims
  global FORTRAN, CPP, g_m, file_text, depth

  OP_ID   = 1;  OP_GBL   = 2;  OP_MAP = 3;

  OP_READ = 1;  OP_WRITE = 2;  OP_RW  = 3;
  OP_INC  = 4;  OP_MAX   = 5;  OP_MIN = 6;

  accsstring = ['OP_READ','OP_WRITE','OP_RW','OP_INC','OP_MAX','OP_MIN' ]

  inc_stage=0
  op_color2=0
  op_color2_force=0
##########################################################################
#  create new kernel file
##########################################################################

  for nk in range (0,len(kernels)):

    name, nargs, dims, maps, var, typs, accs, idxs, inds, soaflags, optflags, decl_filepath, \
           ninds, inddims, indaccs, indtyps, invinds, mapnames, invmapinds, mapinds, nmaps, nargs_novec, \
           unique_args, vectorised, cumulative_indirect_index = op2_gen_common.create_kernel_info(kernels[nk], inc_stage)

    any_soa = 0
    any_soa = any_soa or sum(soaflags)
#
# set logicals
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
    ind_rw = j >= 0

    if ind_rw or op_color2_force:
        op_color2 = 1
    else:
        op_color2 = 0

    #no staging with 2 level colouring
    if op_color2:
      inc_stage=0

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

    j = -1
    for i in range(0,nargs):
      if maps[i] == OP_GBL and accs[i] <> OP_READ and accs[i] <> OP_WRITE:
        j = i
    reduct = j >= 0

    if inc_stage:
      ninds_staged = 0
      inds_staged = [-1]*nargs
      for i in range(0,nargs):
        if maps[i]==OP_MAP and accs[i]==OP_INC:
          if inds_staged[invinds[inds[i]-1]] == -1:
            inds_staged[i] = ninds_staged
            ninds_staged = ninds_staged + 1
          else:
            inds_staged[i] = inds_staged[invinds[inds[i]-1]]
      invinds_staged = [-1]*ninds_staged
      inddims_staged = [-1]*ninds_staged
      indopts_staged = [-1]*ninds_staged
      for i in range(0,nargs):
        if inds_staged[i] >= 0 and invinds_staged[inds_staged[i]] == -1:
          invinds_staged[inds_staged[i]] = i
          inddims_staged[inds_staged[i]] = dims[i]
          if optflags[i] == 1:
            indopts_staged[inds_staged[i]] = i
      for i in range(0,nargs):
        inds_staged[i] = inds_staged[i] + 1

##########################################################################
#  start with CUDA kernel function
##########################################################################

    FORTRAN = 0;
    CPP     = 1;
    g_m = 0;
    file_text = ''
    depth = 0


    #strides for SoA
    if any_soa:
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
            k = k + [mapnames[g_m]]
            code('__constant__ int opDat'+str(invinds[inds[g_m]-1])+'_'+name+'_stride_OP2CONSTANT;')
            code('int opDat'+str(invinds[inds[g_m]-1])+'_'+name+'_stride_OP2HOST=-1;')
      dir_soa = -1
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID and ((not dims[g_m].isdigit()) or int(dims[g_m]) > 1):
          code('__constant__ int direct_'+name+'_stride_OP2CONSTANT;')
          code('int direct_'+name+'_stride_OP2HOST=-1;')
          dir_soa = g_m
          break

    file_name = decl_filepath

    f = open(file_name, 'r')
    kernel_text = f.read()
    f.close()

    if CPP:
      includes = op2_gen_common.extract_includes(kernel_text)
      if len(includes) > 0:
        for include in includes:
          code(include)
        code("")

    comm('user function')
    
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
    head_text = signature_text[0:l].strip() #save function name
    m = op2_gen_common.para_parse(signature_text, 0, '(', ')')
    signature_text = signature_text[l+1:m]
    body_text = kernel_text[i+j+1:k]

    # check for number of arguments
    if len(signature_text.split(',')) != nargs_novec:
        print 'Error parsing user kernel('+name+'): must have '+str(nargs)+' arguments'
        return

    for i in range(0,nargs_novec):
        var = signature_text.split(',')[i].strip()
        if kernels[nk]['soaflags'][i] and (op_color2 or  not (kernels[nk]['maps'][i] == OP_MAP and kernels[nk]['accs'][i] == OP_INC)):
          var = var.replace('*','')
          #locate var in body and replace by adding [idx]
          length = len(re.compile('\\s+\\b').split(var))
          var2 = re.compile('\\s+\\b').split(var)[length-1].strip()

          if int(kernels[nk]['idxs'][i]) < 0 and kernels[nk]['maps'][i] == OP_MAP:
            body_text = re.sub(r'\b'+var2+'(\[[^\]]\])\[([\\s\+\*A-Za-z0-9]*)\]'+'', var2+r'\1[(\2)*'+op2_gen_common.get_stride_string(unique_args[i]-1,maps,mapnames,name)+']', body_text)
          else:
            body_text = re.sub('\*\\b'+var2+'\\b\\s*(?!\[)', var2+'[0]', body_text)
            body_text = re.sub(r'\b'+var2+'\[([\\s\+\*A-Za-z0-9]*)\]'+'', var2+r'[(\1)*'+ \
                               op2_gen_common.get_stride_string(unique_args[i]-1,maps,mapnames,name)+']', body_text)

    signature_text = '__device__ '+head_text + '_gpu( '+signature_text + ') {'
    file_text += signature_text + body_text + '}\n'

    comm('')
    comm(' CUDA kernel function')

    if FORTRAN:
      code('subroutine op_cuda_'+name+'(')
    elif CPP:
      code('__global__ void op_cuda_'+name+'(')

    depth = 2

    if nopts > 0:
      code('int optflags,')
      
    for g_m in range(0,ninds):
      if (indaccs[g_m]==OP_READ):
        code('const INDTYP *__restrict INDARG,')
      else:
        code('INDTYP *__restrict INDARG,')

    if nmaps > 0:
      k = []
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
          k = k + [mapnames[g_m]]
          code('const int *__restrict opDat'+str(invinds[inds[g_m]-1])+'Map, ')



    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        if accs[g_m] == OP_READ:
          code('const TYP *__restrict ARG,')
        else:
          code('TYP *ARG,')
      elif maps[g_m] == OP_GBL:
        if accs[g_m] == OP_INC or accs[g_m] == OP_MIN or accs[g_m] == OP_MAX or accs[g_m] == OP_WRITE:
          code('TYP *ARG,')
        elif accs[g_m] == OP_READ:
          code('const TYP *ARG,')

    if ind_inc and inc_stage==1:
      code('int   *ind_map,')
      code('short *arg_map,')
      code('int   *ind_arg_sizes,')
      code('int   *ind_arg_offs, ')

    if ninds>0:
      if not op_color2:
        code('int    block_offset, ')
        code('int   *blkmap,       ')
        code('int   *offset,       ')
        code('int   *nelems,       ')
        code('int   *ncolors,      ')
        code('int   *colors,       ')
        code('int   nblocks,       ')
      else:
        code('int start,           ')
        code('int end,             ')
        code('int *col_reord,      ')
      code('int   set_size) {    ')
    else:
      code('int   set_size ) {')
      code('')


    for g_m in range(0,nargs):
      if maps[g_m]==OP_GBL and accs[g_m]<>OP_READ and accs[g_m] <> OP_WRITE:
        code('TYP ARG_l[DIM];')
        if accs[g_m] == OP_INC:
          FOR('d','0','DIM')
          code('ARG_l[d]=ZERO_TYP;')
          ENDFOR()
        else:
          FOR('d','0','DIM')
          code('ARG_l[d]=ARG[d+blockIdx.x*DIM];')
          ENDFOR()
      elif maps[g_m]==OP_MAP and accs[g_m]==OP_INC and not op_color2:
        code('TYP ARG_l[DIM];')

    if not op_color2:
      for m in range (1,ninds+1):
        g_m = m -1
        v = [int(inds[i]==m) for i in range(len(inds))]
        v_i = [vectorised[i] for i in range(len(inds)) if inds[i] == m]
        if sum(v)>1 and sum(v_i)>0: #check this sum(v_i)
          if indaccs[m-1] == OP_INC:
            ind = int(max([idxs[i] for i in range(len(inds)) if inds[i]==m])) + 1
            code('INDTYP *arg'+str(invinds[m-1])+'_vec['+str(ind)+'] = {'); depth += 2;
            for n in range(0,nargs):
              if inds[n] == m:
                g_m = n
                code('ARG_l,')
            depth -= 2
            code('};')
#
# lengthy code for general case with indirection
#
    if ninds>0 and not op_color2:
      code('')
      if inc_stage==1:
        for g_m in range (0,ninds):
          if indaccs[g_m] == OP_INC:
            code('__shared__  int  *INDARG_map, INDARG_size;')
            code('__shared__  INDTYP *INDARG_s;')
        code('')
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

      if inc_stage==1 and ind_inc:
        for g_m in range (0,ninds_staged):
          if indopts_staged[g_m-1] > 0:
            IF('optflags & 1<<'+str(optidxs[indopts_staged[g_m-1]]))
          code('ind_arg'+str(inds[invinds_staged[g_m]]-1)+'_size = ind_arg_sizes['+str(g_m)+'+blockId*'+ str(ninds_staged)+'];')
          if indopts_staged[g_m-1] > 0:
            ENDIF()

        code('')
        for m in range (1,ninds_staged+1):
          g_m = m - 1
          c = [i for i in range(nargs) if inds_staged[i]==m]
          code('ind_arg'+str(inds[invinds_staged[g_m]]-1)+'_map = &ind_map['+str(cumulative_indirect_index[c[0]])+\
          '*set_size] + ind_arg_offs['+str(m-1)+'+blockId*'+str(ninds_staged)+'];')

        code('')
        comm('set shared memory pointers')
        code('int nbytes = 0;')

        for g_m in range(0,ninds_staged):
          code('ind_arg'+str(inds[invinds_staged[g_m]]-1)+'_s = ('+typs[invinds_staged[g_m]]+' *) &shared[nbytes];')
          if g_m < ninds_staged-1:
            if indopts_staged[g_m-1] > 0:
              IF('optflags & 1<<'+str(optidxs[indopts_staged[g_m-1]]))
            code('nbytes    += ROUND_UP(ind_arg'+str(inds[invinds_staged[g_m]]-1)+'_size*sizeof('+typs[invinds_staged[g_m]]+')*'+dims[invinds_staged[g_m]]+');')
            if indopts_staged[g_m-1] > 0:
              ENDIF()


      ENDIF()
      code('__syncthreads(); // make sure all of above completed')
      code('')

      if inc_stage==1:
        for g_m in range(0,ninds):
          if indaccs[g_m] == OP_INC:
            FOR_INC('n','threadIdx.x','ind_ARG_size*INDDIM','blockDim.x')
            code('ind_ARG_s[n] = ZERO_INDTYP;')
            ENDFOR()
        if ind_inc:
          code('')
          code('__syncthreads();')
          code('')

      if ind_inc:
        FOR_INC('n','threadIdx.x','nelems2','blockDim.x')
        code('int col2 = -1;')
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapinds[g_m] in k):
            k = k + [mapinds[g_m]]
            code('int map'+str(mapinds[g_m])+'idx;')
        IF('n<nelem')
        comm('initialise local variables')

        for g_m in range(0,nargs):
          if maps[g_m]==OP_MAP and accs[g_m]==OP_INC:
            FOR('d','0','DIM')
            code('ARG_l[d] = ZERO_TYP;')
            ENDFOR()
      else:
        FOR_INC('n','threadIdx.x','nelem','blockDim.x')
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapinds[g_m] in k):
            k = k + [mapinds[g_m]]
            code('int map'+str(mapinds[g_m])+'idx;')

      #non-optional maps
      k = []
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (not optflags[g_m]) and (not mapinds[g_m] in k):
          k = k + [(0*nargs+mapinds[g_m])] #non-opt
          k = k + [(1*nargs+mapinds[g_m])] #opt
          code('map'+str(mapinds[g_m])+'idx = opDat'+str(invmapinds[inds[g_m]-1])+'Map[n + offset_b + set_size * '+str(int(idxs[g_m]))+'];')

      #whatever didn't come up and is opt
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and ((not (optflags[g_m]*nargs+mapinds[g_m]) in k) and (not mapinds[g_m] in      k)):
          k = k + [(optflags[g_m]*nargs+mapinds[g_m])]
          if optflags[g_m]==1:
            IF('optflags & 1<<'+str(optidxs[g_m]))

          code('map'+str(mapinds[g_m])+'idx = opDat'+str(invmapinds[inds[g_m]-1])+'Map[n + offset_b + set_size * '+str(int(idxs[g_m]))+'];')
          if optflags[g_m]==1:
            ENDIF()
 
      code('')
      for g_m in range (0,nargs):
        if accs[g_m] <> OP_INC: #TODO: add opt handling here
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
              if soaflags[g_m]:
                line = line + indent + ' &ind_arg'+str(inds[first]-1)+'[map'+str(mapinds[g_m+k])+'idx],\n'
              else:
                line = line + indent + ' &ind_arg'+str(inds[first]-1)+'[DIM * map'+str(mapinds[g_m+k])+'idx],\n'
            line = line[:-2]+'};'
            code(line)
#
# simple version for global coloring
#
    elif ninds>0:
      code('int tid = threadIdx.x + blockIdx.x * blockDim.x;')
      code('if(tid + start >= end) return;')
      code('int n = col_reord[tid + start];')
      comm('initialise local variables')
      #mapidx declarations
      k = []
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (not mapinds[g_m] in k):
          k = k + [mapinds[g_m]]
          code('int map'+str(mapinds[g_m])+'idx;')

      #non-optional maps
      k = []
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (not optflags[g_m]) and (not mapinds[g_m] in k):
          k = k + [(0*nargs+mapinds[g_m])] #non-opt
          k = k + [(1*nargs+mapinds[g_m])] #opt
          code('map'+str(mapinds[g_m])+'idx = opDat'+str(invmapinds[inds[g_m]-1])+'Map[n + set_size * '+str(int(idxs[g_m]))+'];')

      #whatever didn't come up and is opt
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and ((not (optflags[g_m]*nargs+mapinds[g_m]) in k) and (not mapinds[g_m] in      k)):
          k = k + [(optflags[g_m]*nargs+mapinds[g_m])]
          if optflags[g_m]==1:
            IF('optflags & 1<<'+str(optidxs[g_m]))

          code('map'+str(mapinds[g_m])+'idx = opDat'+str(invmapinds[inds[g_m]-1])+'Map[n + set_size * '+str(int(idxs[g_m]))+'];')
          if optflags[g_m]==1:
            ENDIF()

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
              if soaflags[g_m]:
                line = line + indent + ' &ind_arg'+str(inds[first]-1)+'[map'+str(mapinds[g_m+k])+'idx],\n'
              else:
                line = line + indent + ' &ind_arg'+str(inds[first]-1)+'[DIM * map'+str(mapinds[g_m+k])+'idx],\n'
            line = line[:-2]+'};'
            code(line)




#
# simple alternative when no indirection
#
    else:
      code('')
      comm('process set elements')
      FOR_INC('n','threadIdx.x+blockIdx.x*blockDim.x','set_size','blockDim.x*gridDim.x')

#
# kernel call
#
    code('')
    comm('user-supplied kernel call')
    line = name+'_gpu('
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
      elif maps[m]==OP_MAP and  accs[m]==OP_INC and not op_color2:
        if vectorised[m]:
          if m+1 in unique_args:
            line += rep(indent+'ARG_vec,\n',m)
        else:
          line += rep(indent+'ARG_l,\n',m)
        a =a+1
      elif maps[m]==OP_MAP:
        if vectorised[m]:
          if m+1 in unique_args:
            line += rep(indent+'ARG_vec,\n',m)
        else:
          if soaflags[m]:
            line += rep(indent+'ind_arg'+str(inds[m]-1)+'+map'+str(mapinds[m])+'idx,'+'\n',m)
          else:
            line += rep(indent+'ind_arg'+str(inds[m]-1)+'+map'+str(mapinds[m])+'idx*DIM,'+'\n',m)
        a =a+1
      elif maps[m]==OP_ID:
        if ninds>0 and not op_color2:
          if soaflags[m]:
            line += rep(indent+'ARG+(n+offset_b),\n',m)
          else:
            line += rep(indent+'ARG+(n+offset_b)*DIM,\n',m)
          a =a+1
        else:
          if soaflags[m]:
            line += rep(indent+'ARG+n,\n',m)
          else:
            line += rep(indent+'ARG+n*DIM,\n',m)
          a =a+1
      else:
        print 'internal error 1 '

    code(line[0:-2]+');') #remove final ',' and \n

#
# updating for indirect kernels ...
#
    if ninds>0 and not op_color2:
      if ind_inc:
        code('col2 = colors[n+offset_b];')
        ENDIF()
        code('')
        comm('store local variables')
        code('')
        if inc_stage==1:
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

        if inc_stage==1:
          for g_m in range(0,nargs):
            if maps[g_m] == OP_MAP and accs[g_m] == OP_INC:
              if optflags[g_m]==1:
                IF('optflags & 1<<'+str(optidxs[g_m]))
              for d in range(0,int(dims[g_m])):
                if soaflags[g_m]:
                  code('ARG_l['+str(d)+'] += ind_arg'+str(inds[g_m]-1)+'_s[ARG_map+'+str(d)+'*ind_arg'+str(inds[g_m]-1)+'_size];')
                else:
                  code('ARG_l['+str(d)+'] += ind_arg'+str(inds[g_m]-1)+'_s['+str(d)+'+ARG_map*DIM];')
#          for g_m in range(0,nargs):
#            if maps[g_m] == OP_MAP and accs[g_m] == OP_INC:
              for d in range(0,int(dims[g_m])):
                if soaflags[g_m]:
                  code('ind_arg'+str(inds[g_m]-1)+'_s[ARG_map+'+str(d)+'*ind_arg'+str(inds[g_m]-1)+'_size] = ARG_l['+str(d)+'];')
                else:
                  code('ind_arg'+str(inds[g_m]-1)+'_s['+str(d)+'+ARG_map*DIM] = ARG_l['+str(d)+'];')
                
              if optflags[g_m]==1:
                ENDIF()
        else:
          for g_m in range(0,nargs):
            if maps[g_m] == OP_MAP and accs[g_m] == OP_INC:
              if optflags[g_m]==1:
                IF('optflags & 1<<'+str(optidxs[g_m]))
              for d in range(0,int(dims[g_m])):
                if soaflags[g_m]:
                  code('ARG_l['+str(d)+'] += ind_arg'+str(inds[g_m]-1)+'['+str(d)+'*'+op2_gen_common.get_stride_string(g_m,maps,mapnames,name)+'+map'+str(mapinds[g_m])+'idx];')
                else:
                  code('ARG_l['+str(d)+'] += ind_arg'+str(inds[g_m]-1)+'['+str(d)+'+map'+str(mapinds[g_m])+'idx*DIM];')
#          for g_m in range(0,nargs):
#            if maps[g_m] == OP_MAP and accs[g_m] == OP_INC:
              for d in range(0,int(dims[g_m])):
                if soaflags[g_m]:
                  code('ind_arg'+str(inds[g_m]-1)+'['+str(d)+'*'+op2_gen_common.get_stride_string(g_m,maps,mapnames,name)+'+map'+str(mapinds[g_m])+'idx] = ARG_l['+str(d)+'];')
                else:
                  code('ind_arg'+str(inds[g_m]-1)+'['+str(d)+'+map'+str(mapinds[g_m])+'idx*DIM] = ARG_l['+str(d)+'];')
              if optflags[g_m]==1:
                ENDIF()

        ENDFOR()
        code('__syncthreads();')
        ENDFOR()
    if ninds == 0 or not op_color2:
      ENDFOR()

    if inc_stage:
      for g_m in range(0,ninds):
        if indaccs[g_m]==OP_INC:
          if indopts[g_m] > 0:
            IF('optflags & 1<<'+str(optidxs[indopts[g_m-1]]))
          if soaflags[invinds[g_m]]:
            FOR_INC('n','threadIdx.x','INDARG_size','blockDim.x')
            for d in range(0,int(dims[invinds[g_m]])):
              code('arg'+str(invinds[g_m])+'_l['+str(d)+'] = INDARG_s[n+'+str(d)+'*INDARG_size] + INDARG[INDARG_map[n]+'+str(d)+'*'+op2_gen_common.get_stride_string(g_m,maps,mapnames,name)+'];')
            for d in range(0,int(dims[invinds[g_m]])):
              code('INDARG[INDARG_map[n]+'+str(d)+'*'+op2_gen_common.get_stride_string(g_m,maps,mapnames,name)+'] = arg'+str(invinds[g_m])+'_l['+str(d)+'];')
            ENDFOR()
          else:
            FOR_INC('n','threadIdx.x','INDARG_size*INDDIM','blockDim.x')
            code('INDARG[n%INDDIM+INDARG_map[n/INDDIM]*INDDIM] += INDARG_s[n];')
            ENDFOR()
          if indopts[g_m] > 0:
            ENDIF()

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

    if nopts>0:
      code('int optflags = 0;')
      for i in range(0,nargs):
        if optflags[i] == 1:
          IF('args['+str(i)+'].opt')
          code('optflags |= 1<<'+str(optidxs[i])+';')
          ENDIF()
    if nopts > 30:
      print 'ERROR: too many optional arguments to store flags in an integer'
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
      code('')
      code('int    ninds   = '+str(ninds)+';')
      line = 'int    inds['+str(nargs)+'] = {'
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

    IF('set->size > 0')
    code('')

#
# kernel call for indirect version
#
    if ninds>0:
      if inc_stage==1 and ind_inc:
        code('op_plan *Plan = op_plan_get_stage(name,set,part_size,nargs,args,ninds,inds,OP_STAGE_INC);')
      elif op_color2:
        code('op_plan *Plan = op_plan_get_stage(name,set,part_size,nargs,args,ninds,inds,OP_COLOR2);')
      else:
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
        if maps[m]==OP_GBL and (accs[m]==OP_READ or accs[m] == OP_WRITE):
          code('consts_bytes += ROUND_UP(DIM*sizeof(TYP));')

      code('reallocConstArrays(consts_bytes);')
      code('consts_bytes = 0;')

      for m in range(0,nargs):
        if maps[m]==OP_GBL and (accs[m] == OP_READ or accs[m] == OP_WRITE):
          g_m = m
          code('ARG.data   = OP_consts_h + consts_bytes;')
          code('ARG.data_d = OP_consts_d + consts_bytes;')
          FOR('d','0','DIM')
          code('((TYP *)ARG.data)[d] = ARGh[d];')
          ENDFOR()
          code('consts_bytes += ROUND_UP(DIM*sizeof(TYP));')
      code('mvConstArraysToDevice(consts_bytes);')
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
            code('cudaMemcpyToSymbol(opDat'+str(invinds[inds[g_m]-1])+'_'+name+'_stride_OP2CONSTANT, &opDat'+str(invinds[inds[g_m]-1])+'_'+name+'_stride_OP2HOST,sizeof(int));')
            ENDIF()
      if dir_soa<>-1:
          IF('(OP_kernels[' +str(nk)+ '].count==1) || (direct_'+name+'_stride_OP2HOST != getSetSizeFromOpArg(&arg'+str(dir_soa)+'))')
          code('direct_'+name+'_stride_OP2HOST = getSetSizeFromOpArg(&arg'+str(dir_soa)+');')
          code('cudaMemcpyToSymbol(direct_'+name+'_stride_OP2CONSTANT,&direct_'+name+'_stride_OP2HOST,sizeof(int));')
          ENDIF()

#
# transfer global reduction initial data
#

    if ninds == 0:
      comm('set CUDA execution parameters')
      code('#ifdef OP_BLOCK_SIZE_'+str(nk))
      code('  int nthread = OP_BLOCK_SIZE_'+str(nk)+';')
      code('#else')
      code('  int nthread = OP_block_size;')
      comm('  int nthread = 128;')
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
      if not op_color2:
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
      if op_color2:
        code('int start = Plan->col_offsets[0][col];')
        code('int end = Plan->col_offsets[0][col+1];')
        code('int nblocks = (end - start - 1)/nthread + 1;')
      else:
        code('dim3 nblocks = dim3(Plan->ncolblk[col] >= (1<<16) ? 65535 : Plan->ncolblk[col],')
        code('Plan->ncolblk[col] >= (1<<16) ? (Plan->ncolblk[col]-1)/65535+1: 1, 1);')
        IF('Plan->ncolblk[col] > 0')

      if reduct or (inc_stage==1 and ind_inc):
        if reduct and inc_stage==1:
          code('int nshared = MAX(Plan->nshared,reduct_size*nthread);')
        elif reduct:
          code('int nshared = reduct_size*nthread;')
        else:
          code('int nshared = Plan->nsharedCol[col];')
        code('op_cuda_'+name+'<<<nblocks,nthread,nshared>>>(')
      else:
        code('op_cuda_'+name+'<<<nblocks,nthread>>>(')

      if nopts > 0:
        code('optflags,')
      for m in range(1,ninds+1):
        g_m = invinds[m-1]
        code('(TYP *)ARG.data_d,')
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
            k = k + [mapnames[g_m]]
            code('arg'+str(g_m)+'.map_data_d, ')
      for g_m in range(0,nargs):
        if inds[g_m]==0:
          code('(TYP*)ARG.data_d,')

      if inc_stage==1 and ind_inc:
        code('Plan->ind_map,')
        code('Plan->loc_map,')
        code('Plan->ind_sizes,')
        code('Plan->ind_offs,')
      if op_color2:
        code('start,')
        code('end,')
        code('Plan->col_reord,')
      else:
        code('block_offset,')
        code('Plan->blkmap,')
        code('Plan->offset,')
        code('Plan->nelems,')
        code('Plan->nthrcol,')
        code('Plan->thrcol,')
        code('Plan->ncolblk[col],')
      code('set->size+set->exec_size);')
      code('')
      if reduct:
        comm('transfer global reduction data back to CPU')
        IF('col == Plan->ncolors_owned-1')
        code('mvReductArraysToHost(reduct_bytes);')
        ENDIF()
      if not op_color2:
        ENDFOR() #TODO sztem ez forditva van...
        code('block_offset += Plan->ncolblk[col];')
      ENDIF()
#
# kernel call for direct version
#
    else:
      if reduct:
        code('int nshared = reduct_size*nthread;')
        code('op_cuda_'+name+'<<<nblocks,nthread,nshared>>>(')
      else:
        code('op_cuda_'+name+'<<<nblocks,nthread>>>(')

      indent = '  '#*(len(name)+42)
      if nopts > 0:
        code(indent+'optflags,')
      for g_m in range(0,nargs):
        if g_m > 0:
          code(indent+'(TYP *) ARG.data_d,')
        else:
          code(indent+'(TYP *) ARG.data_d,')

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
        if maps[m]==OP_GBL and accs[m]<>OP_READ and accs[m] <> OP_WRITE:
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
        code('')
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

    code('cutilSafeCall(cudaDeviceSynchronize());')
    comm('update kernel record')
    code('op_timers_core(&cpu_t2, &wall_t2);')
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
    depth = depth - 2
    code('}')


##########################################################################
#  output individual kernel file
##########################################################################
    if not os.path.exists('cuda'):
        os.makedirs('cuda')
    fid = open('cuda/'+name+'_kernel.cu','w')
    date = datetime.datetime.now()
    fid.write('//\n// auto-generated by op2.py\n//\n\n')
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
  code('void op_decl_const_char(int dim, char const *type,')
  code('int size, char *dat, char const *name){')
  depth = depth + 2

  code('if (!OP_hybrid_gpu) return;')
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
  fid.write('//\n// auto-generated by op2.py\n//\n\n')
  fid.write(file_text)
  fid.close()

