##########################################################################
#
# CUDA JIT code generator
#
# This routine is called by op2 which parses the input files
#
# It produces files xxx_kernel.cu & xx_kernel_rec.cu for each kernel,
# plus a master kernel file
#
##########################################################################

##TODO
# iterating over const array: macro type and local list?
# benchmark

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
    code('#include "op_cuda_rt_support.h"')
    code('#include "op_cuda_reduction.h"')
    comm('global_constants - values #defined by JIT')
    code('#include "jit_const.h"')
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

    # old_file = file_text
    # file_text = ''
    #
    # depth+=2
    # # Check constant values
    # IF('blockIdx.x == 0 && threadIdx.x == 0')
    #
    # for nc in range (0,len(consts)):
    #     if consts[nc]['dim']==1:
    #         comm(consts[nc]['name'])
    #         code('printf("'+name+'-'+consts[nc]['name']+': %1.17e\\n",'+ consts[nc]['name']+');')
    #     else:
    #         FOR('i','0',consts[nc]['dim'])
    #         code('printf("  %f\\n", '+consts[nc]['name']+'[i]);')
    #         ENDFOR()
    # ENDIF()
    # depth-=2
    #
    # body_text += file_text
    # file_text = old_file

    user_function = file_text + "//user function\n" + "__device__ void " + name + "_gpu( " + signature_text + ")\n{" + body_text + "}\n"

    file_text = ''

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
          code('const int* __restrict opDat'+str(invinds[inds[g_m]-1])+'Map, ')

    for g_m in range(0,nargs):
      is_const = ''
      if accs[g_m] == OP_READ:
        is_const = 'const '
      if maps[g_m] == OP_ID:
        code(is_const + 'TYP* __restrict ARG,')
      elif maps[g_m] == OP_GBL:
        code(is_const + 'TYP* ARG,')

    if ninds > 0:
      code('int start,')
      code('int end,')

    code('int set_size)')
    depth = 0
    code('{')

    #Function Body
    depth = 2

    for g_m in range(0,nargs):
      if maps[g_m]==OP_GBL and accs[g_m]<>OP_READ and accs[g_m]<>OP_WRITE:
        code('TYP ARG_1[DIM];');
        FOR('d','0','DIM');
        if accs[g_m]==OP_INC:
          code('ARG_1[d]=ZERO_TYP');
        else:
          code('ARG_1[d]=ARG[d+blockIdx.x*DIM];');
        ENDFOR();
    code('');

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
          code('TYP ARG_1[DIM];')
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

      code('')

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
        if accs[g_m] == OP_READ or accs[g_m] == OP_WRITE:
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
          if accs[g_m]   == OP_INC:
            op = 'OP_INC'
          elif accs[g_m] == OP_MIN:
            op = 'OP_MIN'
          elif accs[g_m] == OP_MAX:
            op = 'OP_MAX'
          else:
            print 'internal error: invalid reduction option'
            sys.exit(2)

          FOR('d','0','DIM')
          code('op_reduction<'+op+'>(&ARG[d+blockIdx.x*DIM],ARG_1[d]);')
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
#  Host stub
##########################################################################

    decl     = 'void op_par_loop_'+name+'_execute(op_kernel_descriptor* desc)'

##########################################################################
#  Recompiled Host stub decl
##########################################################################

    code('extern "C" {')
    code(decl + ';')
    code('')

    comm('Recompiled host stub function')
    code(decl)
    code('{')

    host_decl_rec = file_text
    file_text = ''

##########################################################################
#  Host stub decl
##########################################################################

    comm('Host stub function')
    code(decl)
    code('{')

##########################################################################
#  JIT compile call
##########################################################################

    depth += 2
    code('#ifdef OP2_JIT')
    depth += 2
    IF("!jit_compiled")
    code('jit_compile();')
    ENDIF()
    code('(*'+name+'_function)(desc);')
    code('return;')
    depth -= 2
    code('#endif')
    code('')

    host_decl = file_text
    file_text = ''

##########################################################################
#  Common function body
##########################################################################

    code('op_set set = desc->set;')
    #code('char const* name = desc->name;')
    code('int nargs = '+str(nargs)+';')

    code('')

    for g_m in range(0,nargs):
      u = [i for i in range(0,len(unique_args)) if unique_args[i]-1 == g_m]
      if len(u) > 0 and vectorised[g_m] > 0:
        code('ARG.idx = 0;')
        code('args['+str(g_m)+'] = ARG;')

        v = [int(vectorised[i] == vectorised[g_m]) for i in range(0,len(vectorised))]
        first = [i for i in range(0,len(v)) if v[i] == 1]
        first = first[0]

        FOR('v','1',str(sum(v)))
        code('args['+str(g_m)+' + v] = op_arg_dat(arg'+str(first)+'.dat, v, arg'+str(first)+'.map, DIM, "TYP", '+accstring[accs[g_m]]+');')
        ENDFOR()
        code('')
      elif vectorised[g_m] > 0:
        pass
      else:
        code('op_arg ARG = desc->args['+str(g_m)+'];')

    code('')
    start = 'op_arg args['+str(nargs)+'] = {'
    end = ','
    prefix = ' '*len(start)
    for m in unique_args:
      g_m = m - 1
      if g_m != 0:
        start = prefix
      elif m == unique_args[len(unique_args)-1]:
        end = ''

      code(start + 'ARG' + end)

    code('};')
    code('')

    for g_m in range(0,nargs):
      if maps[g_m]==OP_GBL:
        code('TYP *ARGh = (TYP *)ARG.data;')
    code('')

#
# start timing
#
    comm('initialise timers')
    code('double cpu_t1, cpu_t2, wall_t1, wall_t2;')
    code('op_timing_realloc('+str(nk)+');')
    code('op_timers_core(&cpu_t1, &wall_t1);')
    code('')
#
# diagnostics
#
    IF('OP_diags > 2')
    if ninds > 0:
      code('printf(" kernel routine with indirection: '+name+'\\n");')
    else:
      code('printf(" kernel routine without indirection: '+name+'\\n");')
    ENDIF()
#
# Halo exchange
#
    code('')
    code('int set_size = op_mpi_halo_exchanges_cuda(set, nargs, args);')
#
# Kernel calls
#
    code('')
    IF('set->size > 0')
    code('')

#
# transfer global reduction initial data
#

    comm('set CUDA execution parameters')
    envStr = 'OP_BLOCK_SIZE_' + str(nk)
    code('#ifdef ' + envStr)
    depth += 2
    code('int nthread = ' + envStr + ';')
    depth -= 2

    code('#else')
    depth += 2
    code('int nthread = OP_block_size;')
    depth -=2
    code('#endif')
    code('')
    if ninds == 0:
      code('int nblocks = 200;')
      code('')

    if reduct:
      comm('transfer global reduction data to GPU')
      if ninds > 0:
        code('int maxblocks = (MAX(set->core_size, set->size+set->exec_size-set->core_size)-1/nthread+1;')
      else:
        code('int maxblocks = nblocks;')

      code('int reduct_bytes = 0;')
      code('int reduct_size = 0;')

      for g_m in range (0,nargs):
        if maps[g_m]==OP_GBL and accs[g_m]<>OP_READ and accs[g_m]<>OP_WRITE:
          code('reduct_bytes += ROUND_UP(maxblocks*DIM*sizeof(TYP));')
          code('reduct_size   = MAX(reduct_size, sizeof(TYP));')

      code('reallocReductArrays(reduct_bytes);')
      code('');
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
 # indirect
 #
    if ninds>0:
      if reduct:
        FOR('round','0','3')
      else:
        FOR('round','0','2')
      IF('round==1')
      code('op_mpi_wait_all_cuda(nargs, args);')
      ENDIF()
      if reduct:
        code('int start = round==0 ? 0 : (round==1 ? set->core_size : set->size);')
        code('int end = round==0 ? set->core_size : (round==1? set->size :  set->size + set->exec_size);')
      else:
        code('int start = round==0 ? 0 : set->core_size;')
        code('int end = round==0 ? set->core_size : set->size + set->exec_size;')
      IF('end - start>0')
      code('int nblocks = (end-start-1)/nthread+1;')

#      code('printf("'+name+' blocks: %d, nthread: %d, total: %d\\n", nblocks, nthread, nblocks*nthread);')
      if reduct:
        code('int nshared = reduct_size*nthread;')
        code('op_cuda_'+name+'<<<nblocks,nthread,nshared>>>(')
      else:
        code('op_cuda_'+name+'<<<nblocks,nthread>>>(')

      depth += 2
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
      code('start,end,set->size+set->exec_size);')
      depth -= 2;
      code('cudaError_t err = cudaGetLastError();')
      IF('err != cudaSuccess')
      code('printf("CUDA error: %s\\n", cudaGetErrorString(err));')
      code('exit(1);')
      ENDIF()

      ENDIF()
      if reduct:
        code('if (round==1) mvReductArraysToHost(reduct_bytes);')
      ENDFOR()

 #
 # direct
 #
    else:
#      code('printf("'+name+' blocks: %d, nthread: %d, total: %d\\n", nblocks, nthread, nblocks*nthread);')
      if reduct:
        code('int nshared = reduct_size*nthread;')
        func_name = 'op_cuda_'+name+'<<<nblocks,nthread,nshared>>>('
      else:
        func_name = 'op_cuda_'+name+'<<<nblocks,nthread>>>('

      indent = ' '*(len(func_name))
      if nopts > 0:
        code (func_name + 'optflags,')
      for g_m in range(0,nargs):
        start = indent
        end = ','
        if nopts == 0 and g_m == 0:
          start = func_name
        code(start+'(TYP*) ARG.data_d'+end)

      code(indent+'set->size')
      code(');')
      code('cudaError_t err = cudaGetLastError();')
      IF('err != cudaSuccess')
      code('printf("CUDA error: %s\\n", cudaGetErrorString(err));')
      code('exit(1);')
      ENDIF()

#
# transfer global reduction initial data
#

    if reduct:
      if ninds == 0:
        comm('transfer global reduction data back to CPU')
        code('mvReductArraysToHost(reduct_bytes);');

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
        FOR('d','0','DIM')
        code('ARGh[d] = ((TYP *)ARG.data)[d];')
        ENDFOR()
        code('ARG.data = (char *)ARGh;')
        code('op_mpi_reduce(&ARG,ARGh);')

    ENDIF()
    code('op_mpi_set_dirtybit_cuda(nargs, args);')
    code('')

#
# update kernel record
#

    code('cutilSafeCall(cudaDeviceSynchronize());')
    comm(' update kernel record')
    code('op_timers_core(&cpu_t2, &wall_t2);')
    code('OP_kernels[' +str(nk)+ '].name      = desc->name;')
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

#
# Output
#
    code('')
    body = file_text
    file_text = ''

##########################################################################
#  Function called from modified source
##########################################################################

    comm('Function called from modified source')
    end = ','
    prefix = ' '*len("void ")
    code('void op_par_loop_'+name+'(char const* name, op_set set,')
    for m in unique_args:
      g_m = m - 1
      if m == unique_args[len(unique_args)-1]:
        end = ')'

      code(prefix + 'op_arg ARG' + end)

    code('{')
    code('')
    depth += 2

    code('int nargs = '+str(nargs)+';')
    code('op_arg args['+str(nargs)+'];')
    code('')

    code('op_kernel_descriptor *desc = ')
    code('(op_kernel_descriptor *)malloc(sizeof(op_kernel_descriptor));')
    code('desc->name = name;')
    code('desc->set = set;')
    code('desc->device = 1;')
    code('desc->index = '+str(nk)+';')
    code('desc->hash = 5381;')
    code('desc->hash = ((desc->hash << 5) + desc->hash) + '+str(nk)+';')
    code('')

    comm('save the arguments')
    code('desc->nargs = '+str(nargs)+';')
    code('desc->args = (op_arg *)malloc('+str(nargs)+' * sizeof(op_arg));')

    declared = 0
    for n in range (0, nargs):
      code('desc->args['+str(n)+'] = arg'+str(n)+';')
      if maps[n] <> OP_GBL:
        code('desc->hash = ((desc->hash << 5) + desc->hash) + arg'+str(n)+'.dat->index;')
      if maps[n] == OP_GBL and accs[n] == OP_READ:
        if declared == 0:
          code('char *tmp = (char*)malloc('+dims[n]+'*sizeof('+typs[n]+'));')
          declared = 1
        else:
          code('tmp = (char*)malloc('+dims[n]+'*sizeof('+typs[n]+'));')
        code('memcpy(tmp, arg'+str(n)+'.data,'+dims[n]+'*sizeof('+typs[n]+'));')
        code('desc->args['+str(n)+'].data = tmp;')
    code('desc->function = op_par_loop_'+name+'_execute;')

    code('')
    code('op_enqueue_kernel(desc);')
    depth -= 2
    code('}')

    body_end = file_text

##########################################################################
#  output individual kernel file
##########################################################################
    if not os.path.exists('cuda'):
        os.makedirs('cuda')
    fid = open('cuda/'+name+'_kernel.cu','w')
    fid.write('//\n// auto-generated by op2.py\n//\n\n')

    # Handle constants:
    #  - AOT: append _cuda for version copied to device mem
    aot_user_function = user_function
    for nc in range(0,len(consts)):
      varname = consts[nc]['name']
      aot_user_function = re.sub('\\b'+varname+'\\b',varname+'_cuda',aot_user_function)

    outfile = aot_user_function   +\
              kernel_function +\
              host_decl       +\
              body            +\
              body_end

    fid.write(outfile)
    fid.close()

##########################################################################
#  output individual kernel file - JIT compiled version
##########################################################################
    if not os.path.exists('cuda'):
        os.makedirs('cuda')
    fid = open('cuda/'+name+'_kernel_rec.cu','w')
    fid.write('//\n// auto-generated by op2.py\n//\n\n')


    # Handle constants:
    #  - JIT: Leave dim 1 constants
    #  -      transform multi-value constants
    jit_user_function = user_function
    for nc in range(0,len(consts)):
      varname = consts[nc]['name']
      if consts[nc]['dim'] != 1:
        jit_user_function = re.sub('\\b'+varname+'\[([0-9]+)\]','op_const_'+varname+'_\g<1>', jit_user_function)
        jit_user_function, numFound = re.subn('\\b'+varname+'\[', 'op_const_'+varname+'[', jit_user_function)

        if (numFound > 0):
            if CPP:
                #Line start
                codeline = '__constant__ '+consts[nc]['type'][1:-1]+' op_const_'+varname+'['+consts[nc]['dim']+'] = {'

                #Add each value to line
                for i in range(0,int(consts[nc]['dim'])):
                    codeline += "op_const_"+varname+"_"+str(i)+", "
                codeline = codeline[:-2] + "};"

                #Add list definition at top of function
                jit_user_function = codeline+'\n\n'+jit_user_function

    outfile = jit_include       +\
              jit_user_function     +\
              kernel_function   +\
              host_decl_rec     +\
              body              +\
              '} //end extern c'

    outfile = re.sub('_'+name,'_'+name+'_rec', outfile)
    fid.write(outfile)
    fid.close()

# end of main kernel call loop


##########################################################################
#  output one master kernel file
##########################################################################

  file_text =''
  comm(' header')
  if os.path.exists('./user_types.h'):
    code('#include "../user_types.h"')
  code('#include "op_lib_cpp.h"       ')
  code('#include "op_cuda_rt_support.h"')
  code('#include "op_cuda_reduction.h"')
  code('')
  comm(' global constants')

  for nc in range (0,len(consts)):
    if consts[nc]['dim']==1:
      # __constant__ [type] [name]_cuda;
      code('__constant__ '+consts[nc]['type'][1:-1]+' '+consts[nc]['name']+'_cuda;')
    else:
      if consts[nc]['dim'] > 0:
        num = str(consts[nc]['dim'])
      else:
        num = 'MAX_CONST_SIZE'

      # __constant__ [type] [name]_cuda[ [dim] ];
      code('__constant__ '+consts[nc]['type'][1:-1]+' '+consts[nc]['name']+'_cuda'+'['+num+'];')

  #
  # JIT DISABLED
  #

  code('\n#ifndef OP2_JIT');
  code('');
  # op_decl_const_char
  code('void op_decl_const_char(int dim, char const *type,')
  code('                        int size, char *dat,')
  code('                        char const *name)')
  code('{')
  depth += 2;
  comm(' copy value to device constant')

  for nc in range (0, len(consts)):
    IF('!strcmp(name,"'+consts[nc]['name']+'")')
    if(consts[nc]['dim'] < 0):
      IF('size>MAX_CONST_SIZE')
      code('printf("error: MAX_CONST_SIZE not big enough\\n");')
      code('exit(1);');
      ENDIF()
    code('cutilSafeCall(cudaMemcpyToSymbol('+consts[nc]['name']+'_cuda, dat, dim*size));')
    ENDIF()
    code('else')
  code('{')
  depth += 2
  code('printf("error: unknown const name\\n");')
  code('exit(1);')
  ENDIF()

  depth -= 2
  code('}')

  #
  # JIT ENABLED
  #

  code('\n#else')
  code('');

  # op_decl_const_char
  code('void op_decl_const_char(int dim, char const *type,')
  code('                        int size, char *dat,')
  code('                        char const *name)')
  code('{')
  depth += 2;
  IF('dim == 1')
  code('op_lazy_const(dim, type, size, dat, name);')
  ENDIF()
  code('else {')
  depth += 2
  FOR('d','0','dim')
  code('char name2[32];')
  code('sprintf(name2, "op_const_%s_%d\\0", name, d);')
  code('op_lazy_const(1, type, size, dat+(d*size), name2);')
  ENDFOR()
  ENDIF()

  depth -= 2;
  code('}')

  code('')
  comm(' pointers to recompiled functions')
  for nk in range (0,len(kernels)):
    name  = kernels[nk]['name']
    code('void (*'+name+'_function)(struct op_kernel_descriptor *desc) = NULL;')

  # do compiliation and linking using makefile
  code('')
  code('void jit_compile() {')
  depth += 2
  code('op_printf("JIT compiling op_par_loops\\n");')
  code('')

#
# start timing
#
  comm('initialise timers')
  code('double cpu_t1, cpu_t2, wall_t1, wall_t2;')
  code('op_timing_realloc('+str(nk)+');')
  code('op_timers_core(&cpu_t1, &wall_t1);')
  code('')

  comm('Write constants to header file')
  IF('op_is_root()')
  IF('system("make -j '+master.split('.')[0]+'_cuda_rec &> jit_compile.log")')
  comm(' 0 indicates success')
  code('printf("Error: JIT compile failed. \\n - see jit_compile.log for details\\n");')
  code('exit(1);')
  ENDIF()
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

  code('op_timers_core(&cpu_t2, &wall_t2);')
  code('op_printf(" Completed: %fs\\n", wall_t2 - wall_t1);')

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
