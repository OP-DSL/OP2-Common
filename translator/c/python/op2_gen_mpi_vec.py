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

def op2_gen_mpi_vec(master, date, consts, kernels):

  global dims, idxs, typs, indtyps, inddims
  global FORTRAN, CPP, g_m, file_text, depth

  OP_ID   = 1;  OP_GBL   = 2;  OP_MAP = 3;

  OP_READ = 1;  OP_WRITE = 2;  OP_RW  = 3;
  OP_INC  = 4;  OP_MAX   = 5;  OP_MIN = 6;

  accsstring = ['OP_READ','OP_WRITE','OP_RW','OP_INC','OP_MAX','OP_MIN' ]

  grouped = 0

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
      if maps[i] == OP_GBL and accs[i] != OP_READ:
        j = i
    reduct = j >= 0

    j = -1
    for i in range(0,nargs):
      if maps[i] == OP_MAP :
        j = i
    indirect_kernel = j >= 0

    if nargs != nargs_novec:
      return
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
    file_name = decl_filepath

    f = open(file_name, 'r')
    kernel_text = f.read()
    file_text += kernel_text
    f.close()

    ## Clang compiler can struggle to vectorize a loop if it uses a mix of
    ## Python-generated simd arrays for indirect data AND pointers to direct
    ## data. Fix by also generating simd arrays for direct data:
    do_gen_direct_simd_arrays = True

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
      head_text = signature_text[0:l] #save function name
      m = op2_gen_common.para_parse(signature_text, 0, '(', ')')
      signature_text = signature_text[l+1:m]
      body_text = kernel_text[i+j+1:k]

      ## Replace occurrences of '#include "<FILE>"' within loop with the contents of <FILE>:
      body_text = op2_gen_common.replace_local_includes_with_file_contents(body_text, os.path.dirname(master))


      # check for number of arguments
      nargs_actual = len(signature_text.split(','))
      if nargs_actual != nargs:
          print(('Error parsing user kernel({0}): must have {1} arguments (instead it has {2})'.format(name, nargs, nargs_actual)))
          return

      new_signature_text = ''
      for i in range(0,nargs):
        var = signature_text.split(',')[i].strip()

        if do_gen_direct_simd_arrays:
          do_gen_simd_array_arg = maps[i] != OP_GBL
        else:
          do_gen_simd_array_arg = maps[i] != OP_GBL and maps[i] != OP_ID
        if do_gen_simd_array_arg:
          #remove * and add [*][SIMD_VEC]
          var = var.replace('*','')
          #locate var in body and replace by adding [idx]
          length = len(re.compile('\\s+\\b').split(var))
          var2 = re.compile('\\s+\\b').split(var)[length-1].strip()

          #print var2

          body_text = re.sub('\*\\b'+var2+'\\b\\s*(?!\[)', var2+'[0]', body_text)
          array_access_pattern = '\[[\w\(\)\+\-\*\s\\\\]*\]'

          ## It has been observed that vectorisation can fail on loops with increments,
          ## but replacing them with writes succeeds.
          ## For example with Clang on particular loops, vectorisation fails with message:
          ##   "loop not vectorized: loop control flow is not understood by vectorizer"
          ## replacing increments with writes solves this.
          ## Replacement is data-safe due to use of local/intermediate SIMD arrays.
          ## Hopefully the regex is matching all increments.
          ## And for loops that were being vectorised, this change can give a small perf boost.
          if maps[i] == OP_MAP and accs[i] == OP_INC:
            ## Replace 'var' increments with writes:
            body_text = re.sub(r'('+var2+array_access_pattern+'\s*'+')'+re.escape("+="), r'\1'+'=', body_text)

          ## Append vector array access:
          body_text = re.sub(r'('+var2+array_access_pattern+')', r'\1'+'[idx]', body_text)

          var = var + '[][SIMD_VEC]'
          #var = var + '[restrict][SIMD_VEC]'
        new_signature_text +=  var+', '


      #add ( , idx and )
      signature_text = "#if defined __clang__ || defined __GNUC__\n"
      signature_text += "__attribute__((always_inline))\n"
      signature_text += "#endif\n"
      signature_text += "inline " + head_text + '( '+new_signature_text + 'int idx ) {'
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
        code('op_arg <ARG>){');
        code('')
      else:
        code('op_arg <ARG>,')

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

#
# create aligned pointers
#
    comm('create aligned pointers for dats')
    for g_m in range (0,nargs):
        if maps[g_m] != OP_GBL:
          if (accs[g_m] == OP_INC or accs[g_m] == OP_RW or accs[g_m] == OP_WRITE):
            code('ALIGNED_<TYP>       <TYP> * __restrict__ ptr'+\
            str(g_m)+' = (<TYP> *) arg'+str(g_m)+'.data;')
            #code('<TYP>* __restrict__ __attribute__((align_value (<TYP>_ALIGN)))  ptr'+\
            #str(g_m)+' = (<TYP> *) arg'+str(g_m)+'.data;')
            code('DECLARE_PTR_ALIGNED(ptr'+str(g_m)+',<TYP>_ALIGN);')

          else:
            code('ALIGNED_<TYP> const <TYP> * __restrict__ ptr'+\
            str(g_m)+' = (<TYP> *) arg'+str(g_m)+'.data;')
            code('DECLARE_PTR_ALIGNED(ptr'+str(g_m)+',<TYP>_ALIGN);')
            #code('const <TYP>* __restrict__ __attribute__((align_value (<TYP>_ALIGN)))  ptr'+\
            #str(g_m)+' = (<TYP> *) arg'+str(g_m)+'.data;')



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
    if grouped:
      code('int exec_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 1);')
    else:
      code('int exec_size = op_mpi_halo_exchanges(set, nargs, args);')

    code('')
    IF('exec_size >0')
    code('')

#
# kernel call for indirect version
#
    if ninds>0:
      code('#ifdef VECTORIZE')

      code('#pragma novector')
      FOR2('n','0','(exec_size/SIMD_VEC)*SIMD_VEC','SIMD_VEC')
      #initialize globals
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL:
          code('<TYP> dat{0}[SIMD_VEC];'.format(g_m))
          FOR('i','0','SIMD_VEC')
          if accs[g_m] == OP_INC:
            code('dat{0}[i] = 0.0;'.format(g_m))
          elif accs[g_m] == OP_MAX:
            code('dat{0}[i] = -INFINITY;'.format(g_m))
          elif accs[g_m] == OP_MIN:
            code('dat{0}[i] = INFINITY;'.format(g_m))
          elif accs[g_m] == OP_READ:
            code('dat{0}[i] = *((<TYP>*)arg{0}.data);'.format(g_m))
          ENDFOR()

      code('if (n<set->core_size && n>0 && n % OP_mpi_test_frequency == 0)')
      code('  op_mpi_test_all(nargs,args);')
      IF('(n+SIMD_VEC >= set->core_size) && (n+SIMD_VEC-set->core_size < SIMD_VEC)')
      if grouped:
        code('op_mpi_wait_all_grouped(nargs, args, 1);')
      else:
        code('op_mpi_wait_all(nargs, args);')
      ENDIF()
      for g_m in range(0,nargs):
        if do_gen_direct_simd_arrays:
          if (maps[g_m] in [OP_MAP, OP_ID]) and (accs[g_m] in [OP_READ, OP_RW, OP_WRITE, OP_INC]):
            code('ALIGNED_<TYP> <TYP> dat'+str(g_m)+'[<DIM>][SIMD_VEC];')
        else:
          if maps[g_m] == OP_MAP and (accs[g_m] in [OP_READ, OP_RW, OP_WRITE, OP_INC]):
            code('ALIGNED_<TYP> <TYP> dat'+str(g_m)+'[<DIM>][SIMD_VEC];')

      #setup gathers
      idx_map_template = "int idx{0}_<DIM> = <DIM> * arg{1}.map_data[(n+i) * arg{1}.map->dim + {2}];"
      idx_id_template  = "int idx{0}_<DIM> = <DIM> * (n+i);"
      code('#pragma omp simd simdlen(SIMD_VEC)')
      FOR('i','0','SIMD_VEC')
      if nmaps > 0:
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP :
            if (accs[g_m] in [OP_READ, OP_RW, OP_WRITE]):#and (not mapinds[g_m] in k):
              code(idx_map_template.format(g_m, invmapinds[inds[g_m]-1], idxs[g_m]))
          elif do_gen_direct_simd_arrays and maps[g_m] == OP_ID :
            code(idx_id_template.format(g_m))
      code('')

      init_dat_template = "dat{0}[{1}][i] = (ptr{0})[idx{0}_<DIM> + {1}];"
      zero_dat_template = "dat{0}[{1}][i] = 0.0;"
      for g_m in range(0,nargs):
        if do_gen_direct_simd_arrays:
          ## also 'gather' directly-accessed data, because SOME compilers
          ## struggle to vectorise otherwise (e.g. Clang).
          if maps[g_m] != OP_GBL :
            if accs[g_m] in [OP_READ, OP_RW]:
              for d in range(0,int(dims[g_m])):
                code(init_dat_template.format(g_m, d))
              code('')
            elif accs[g_m] == OP_INC:
              for d in range(0,int(dims[g_m])):
                code(zero_dat_template.format(g_m, d))
              code('')
        else:
          if maps[g_m] == OP_MAP :
            if accs[g_m] in [OP_READ, OP_RW]:#and (not mapinds[g_m] in k):
              for d in range(0,int(dims[g_m])):
                init_dat_str = init_dat_template.format(g_m, d)
                code(init_dat_str)
              code('')
            elif (accs[g_m] == OP_INC):
              for d in range(0,int(dims[g_m])):
                zero_dat_str = zero_dat_template.format(g_m, d)
                code(zero_dat_str)
              code('')
          else: #globals
            if (accs[g_m] == OP_INC):
              # for d in range(0,int(dims[g_m])):
              #   code('dat'+str(g_m)+'[i] = 0.0;')
              # code('')
              pass

      ENDFOR()
      #kernel call
      code('#pragma omp simd simdlen(SIMD_VEC)')
      FOR('i','0','SIMD_VEC')
      line = name+'_vec('
      indent = '\n'+' '*(depth+2)
      for g_m in range(0,nargs):
        if (not do_gen_direct_simd_arrays) and maps[g_m] == OP_ID:
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
            if (accs[g_m] in [OP_INC, OP_RW, OP_WRITE]):#and (not mapinds[g_m] in k):
              code(idx_map_template.format(g_m, invmapinds[inds[g_m]-1], idxs[g_m]))
          elif do_gen_direct_simd_arrays and maps[g_m] == OP_ID :
            if (accs[g_m] in [OP_INC, OP_RW, OP_WRITE]):
              code(idx_id_template.format(g_m))
      code('')
      dat_scatter_inc_template = "(ptr{0})[idx{0}_<DIM> + {1}] += dat{0}[{1}][i];"
      dat_scatter_wr_template  = "(ptr{0})[idx{0}_<DIM> + {1}] = dat{0}[{1}][i];"
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP :
          if (accs[g_m] == OP_INC ):
            for d in range(0,int(dims[g_m])):
              code(dat_scatter_inc_template.format(g_m, d))
            code('')
          elif accs[g_m] in [OP_WRITE, OP_RW]:
            for d in range(0,int(dims[g_m])):
              code(dat_scatter_wr_template.format(g_m, d))
            code('')
        elif do_gen_direct_simd_arrays and maps[g_m] == OP_ID:
          ## also scatter directly-written data
          if (accs[g_m] == OP_INC ):
            for d in range(0,int(dims[g_m])):
              code(dat_scatter_inc_template.format(g_m, d))
          elif accs[g_m] in [OP_WRITE, OP_RW]:
            for d in range(0,int(dims[g_m])):
              code(dat_scatter_wr_template.format(g_m, d))
            code('')
      ENDFOR()

      #do reductions
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL:
          FOR('i','0','SIMD_VEC')
          if accs[g_m] == OP_INC:
            code('*(<TYP>*)arg'+str(g_m)+'.data += dat'+str(g_m)+'[i];')
          elif accs[g_m] == OP_MAX:
            code('*(<TYP>*)arg'+str(g_m)+'.data = MAX(*(<TYP>*)arg'+str(g_m)+'.data,dat'+str(g_m)+'[i]);')
          elif accs[g_m] == OP_MIN:
            code('*(<TYP>*)arg'+str(g_m)+'.data = MIN(*(<TYP>*)arg'+str(g_m)+'.data,dat'+str(g_m)+'[i]);')
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
      if grouped:
        code('op_mpi_wait_all_grouped(nargs, args, 1);')
      else:
        code('op_mpi_wait_all(nargs, args);')
      ENDIF()
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
            code('map'+str(mapinds[g_m])+'idx = arg'+str(invmapinds[inds[g_m]-1])+'.map_data[n * arg'+str(invmapinds[inds[g_m]-1])+'.map->dim + '+str(idxs[g_m])+'];')
      #do optional ones
      if nmaps > 0:
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapinds[g_m] in k):
            if optflags[g_m]:
              IF('<ARG>.opt')
            else:
              k = k + [mapinds[g_m]]
            code('map'+str(mapinds[g_m])+'idx = arg'+str(invmapinds[inds[g_m]-1])+'.map_data[n * arg'+str(invmapinds[inds[g_m]-1])+'.map->dim + '+str(idxs[g_m])+'];')
            if optflags[g_m]:
              ENDIF()

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
          code('<TYP> dat{0}[SIMD_VEC];'.format(g_m))
          FOR('i','0','SIMD_VEC')
          if accs[g_m] == OP_INC:
            code('dat{0}[i] = 0.0;'.format(g_m))
          elif accs[g_m] == OP_MAX:
            code('dat{0}[i] = -INFINITY;'.format(g_m))
          elif accs[g_m] == OP_MIN:
            code('dat{0}[i] = INFINITY;'.format(g_m))
          elif accs[g_m] == OP_READ:
            code('dat{0}[i] = *((<TYP>*)arg{0}.data);'.format(g_m))
          ENDFOR()

      code('#pragma omp simd simdlen(SIMD_VEC)')
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
            code('*(<TYP>*)arg'+str(g_m)+'.data += dat'+str(g_m)+'[i];')
          elif accs[g_m] == OP_MAX:
            code('*(<TYP>*)arg'+str(g_m)+'.data = MAX(*(<TYP>*)arg'+str(g_m)+'.data,dat'+str(g_m)+'[i]);')
          elif accs[g_m] == OP_MIN:
            code('*(<TYP>*)arg'+str(g_m)+'.data = MIN(*(<TYP>*)arg'+str(g_m)+'.data,dat'+str(g_m)+'[i]);')
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
      if grouped:
        code('op_mpi_wait_all_grouped(nargs, args, 1);')
      else:
        code('op_mpi_wait_all(nargs, args);')
      ENDIF()

#
# combine reduction data from multiple OpenMP threads
#
    comm(' combine reduction data')
    for g_m in range(0,nargs):
      if maps[g_m]==OP_GBL and accs[g_m]!=OP_READ:
        code('op_mpi_reduce(&<ARG>,('+typs[g_m]+'*)<ARG>.data);')

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
        if maps[g_m]!=OP_GBL:
          if accs[g_m]==OP_READ:
            code(line+' <ARG>.size;')
          else:
            code(line+' <ARG>.size * 2.0f;')
    else:
      names = []
      for g_m in range(0,ninds):
        mult=''
        if indaccs[g_m] != OP_WRITE and indaccs[g_m] != OP_READ:
          mult = ' * 2.0f'
        if not var[invinds[g_m]] in names:
          code('OP_kernels['+str(nk)+'].transfer += (float)set->size * arg'+str(invinds[g_m])+'.size'+mult+';')
          names = names + [var[invinds[g_m]]]
      for g_m in range(0,nargs):
        mult=''
        if accs[g_m] != OP_WRITE and accs[g_m] != OP_READ:
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
    if not os.path.exists('vec'):
        os.makedirs('vec')
    fid = open('vec/'+name+'_veckernel.cpp','w')
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

  code('#define double_ALIGN 128')
  code('#define float_ALIGN 64')
  code('#define int_ALIGN 64')
  code('#ifdef VECTORIZE')
  code('#define SIMD_VEC 4')
  code('#define ALIGNED_double __attribute__((aligned(double_ALIGN)))')
  code('#define ALIGNED_float __attribute__((aligned(float_ALIGN)))')
  code('#define ALIGNED_int __attribute__((aligned(int_ALIGN)))')
  code('  #ifdef __ICC')
  code('    #define DECLARE_PTR_ALIGNED(X, Y) __assume_aligned(X, Y)')
  code('  #else')
  code('    #define DECLARE_PTR_ALIGNED(X, Y)')
  code('  #endif')
  code('#else')
  code('#define ALIGNED_double')
  code('#define ALIGNED_float')
  code('#define ALIGNED_int')
  code('#define DECLARE_PTR_ALIGNED(X, Y)')
  code('#endif')
  code('')

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
  code('#include "op_lib_cpp.h"')
  code('')

  comm(' user kernel files')

  for nk in range(0,len(kernels)):
    code('#include "'+kernels[nk]['name']+'_veckernel.cpp"')
  master = master.split('.')[0]
  fid = open('vec/'+master.split('.')[0]+'_veckernels.cpp','w')
  fid.write('//\n// auto-generated by op2.py\n//\n\n')
  fid.write(file_text)
  fid.close()
