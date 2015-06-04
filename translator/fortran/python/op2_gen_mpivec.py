##########################################################################
#
# MPI+Vectorized seq code generator
#
# This routine is called by op2_fortran which parses the input files
#
# It produces a file xxx_veckernel.F90 for each kernel,
# plus a master kernel file
#
##########################################################################

import re
import datetime
import os
import glob

def comm(line):
  global file_text, FORTRAN, CPP
  global depth
  if len(line) == 0:
    prefix = ''
  else:
    prefix = ' '*depth
  if len(line) == 0:
    file_text +='\n'
  elif FORTRAN:
    file_text +='! '+line+'\n'
  elif CPP:
    file_text +=prefix+'//'+line+'\n'

def rep(line,m):
  global dims, idxs, typs, indtyps, inddims

  if FORTRAN:
    if m < len(inddims):
      line = re.sub('INDDIM',str(inddims[m]),line)
      line = re.sub('INDTYP',str(indtyps[m]),line)

    line = re.sub('INDARG','ind_arg'+str(m+1),line)
    line = re.sub('DIMS',str(dims[m]),line)
    line = re.sub('ARG','arg'+str(m+1),line)
    line = re.sub('TYP',typs[m],line)
    line = re.sub('IDX',str(int(idxs[m])),line)
  elif CPP:
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
  if len(text) == 0:
    file_text += '\n'
    return
  if len(text) == 0:
    prefix = ''
  else:
    prefix = ' '*depth
  if FORTRAN:
    file_text += prefix+rep(text,g_m)+'\n'
  elif CPP:
    file_text += prefix+rep(text,g_m)+'\n'

def code_pre(text):
  global file_text, FORTRAN, CPP, g_m
  if FORTRAN:
    file_text += rep(text,g_m)+'\n'
  elif CPP:
    file_text += rep(text,g_m)+'\n'

def DO(i,start,finish):
  global file_text, FORTRAN, CPP, g_m
  global depth
  if FORTRAN:
    code('DO '+i+' = '+start+', '+finish+'-1, 1')
  elif CPP:
    code('for ( int '+i+'='+start+'; '+i+'<'+finish+'; '+i+'++ ){')
  depth += 2

def DO2(i,start,finish,step):
  global file_text, FORTRAN, CPP, g_m
  global depth
  if FORTRAN:
    code('DO '+i+' = '+start+', '+finish+'-1, '+step)
  elif CPP:
    code('for ( int '+i+'='+start+'; '+i+'<'+finish+'; '+i+'++ ){')
  depth += 2

def FOR(i,start,finish):
  global file_text, FORTRAN, CPP, g_m
  global depth
  if FORTRAN:
    code('DO '+i+' = '+start+', '+finish+'-1')
  elif CPP:
    code('for ( int '+i+'='+start+'; '+i+'<'+finish+'; '+i+'++ ){')
  depth += 2

def ENDDO():
  global file_text, FORTRAN, CPP, g_m
  global depth
  depth -= 2
  if FORTRAN:
    code('END DO')
  elif CPP:
    code('}')

def ENDFOR():
  global file_text, FORTRAN, CPP, g_m
  global depth
  depth -= 2
  if FORTRAN:
    code('END DO')
  elif CPP:
    code('}')

def IF(line):
  global file_text, FORTRAN, CPP, g_m
  global depth
  if FORTRAN:
    code('IF ('+line+') THEN')
  elif CPP:
    code('if ('+ line + ') {')
  depth += 2

def ELSE():
  global file_text, FORTRAN, CPP, g_m
  global depth
  depth -= 2
  if FORTRAN:
    code('ELSE')
  elif CPP:
    code('else {')
  depth += 2

def ENDIF():
  global file_text, FORTRAN, CPP, g_m
  global depth
  depth -= 2
  if FORTRAN:
    code('END IF')
  elif CPP:
    code('}')

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

def op2_gen_mpivec(master, date, consts, kernels, hydra):

  global dims, idxs, typs, indtyps, inddims
  global FORTRAN, CPP, g_m, file_text, depth

  OP_ID   = 1;  OP_GBL   = 2;  OP_MAP = 3;

  OP_READ = 1;  OP_WRITE = 2;  OP_RW  = 3;
  OP_INC  = 4;  OP_MAX   = 5;  OP_MIN = 6;

  accsstring = ['OP_READ','OP_WRITE','OP_RW','OP_INC','OP_MAX','OP_MIN' ]
  typestrigs = ['INTEGER','INT','REAL','DOUBLE','CHAR','FLOAT' ]

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
    optflags = kernels[nk]['optflags']
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
# set three logicals
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

    j = 0
    for i in range(0,nargs):
      if maps[i] == OP_MAP :
        j = i
    indirect_kernel = j > 0

    FORTRAN = 1;
    CPP     = 0;
    g_m = 0;
    file_text = ''
    depth = 0

##########################################################################
#  Generate Header
##########################################################################
    if hydra:
      code('MODULE '+kernels[nk]['mod_file'][4:]+'_MODULE')
    else:
      code('MODULE '+name.upper()+'_MODULE')
    code('USE OP2_FORTRAN_DECLARATIONS')
    code('USE OP2_FORTRAN_RT_SUPPORT')
    code('USE ISO_C_BINDING')
    if hydra == 0:
      code('USE OP2_CONSTANTS')

    code('')

####################################################################################
#  generate the user kernel function - creating versions for vectorisation as needed
####################################################################################
    code('')
    code('CONTAINS')
    code('')
    if hydra == 0:
#
# First original version
#
      comm('user function')
      file_name = name+'.inc'
      f = open(file_name, 'r')
      kernel_text = f.read()
      file_text += kernel_text
      f.close()
      code('')

#
# Modified vectorisable version if its an indirect kernel
# - direct kernels can be vectorised without modification
#
      if indirect_kernel:
        code('#ifdef VECTORIZE')
        code('#define SIMD_VEC 4')
        comm('user function -- modified for vectorisation')
        f = open(file_name, 'r')
        kernel_text = f.read()
        f.close()

        p = re.compile('SUBROUTINE\\s+\\b'+name+'\\b')
        i = p.search(kernel_text).start()

        if(i < 0):
          print "\n********"
          print "Error: cannot locate user kernel function name: "+name+" - Aborting code generation"
          exit(2)
        i2 = i
        j = kernel_text[i:].find('(')
        k = para_parse(kernel_text, i+j, '(', ')')
        l = kernel_text[k:].find('END'+'\\s+\\b'+'SUBROUTINE')
        para = kernel_text[j+1:k].split(',')

        code('SUBROUTINE '+name+'_vec('+kernel_text[j+1:k]+')')
        depth = depth + 2
        code('!dir$ attributes vector :: '+name+'_vec')
        code('IMPLICIT NONE')
        for g_m in range(0,nargs):
          if maps[g_m] <> OP_GBL:
            if (accs[g_m] == OP_INC or accs[g_m] == OP_RW or accs[g_m] == OP_WRITE):
              code('TYP, DIMENSION(SIMD_VEC,DIMS) :: '+para[g_m])
            if (accs[g_m] == OP_READ):
              code('TYP, DIMENSION(SIMD_VEC,DIMS), INTENT(IN) :: '+para[g_m])
          else:
            code('TYP DIMENSION(DIMS) :: '+para[g_m])

        #locate and remove non-local parameters
        body_line = kernel_text[k+1:l].split('\n')
        depth = depth - 2
        kernel_line = ''
        for bl in range(0,len(body_line)-1):
          if not(any(word in body_line[bl] for word in para) and \
            any(word in body_line[bl].upper() for word in typestrigs)):
              if not('IMPLICIT' in body_line[bl]):
                temp = body_line[bl]
                for p in range(0,len(para)):
                  print para[p]
                  temp = re.sub(r'('+para[p]+'\s*'+'\('+')', r'\1'+'idx,', temp)
                  temp = re.sub(para[p]+'(?!\()', para[p]+'(idx,1)', temp)

                kernel_line = temp
              code(kernel_line)

        code('END SUBROUTINE')
        code('#endif')

    ###  Hydra specifics #### should not be in code generator ..
    else:
      file_text += '!DEC$ ATTRIBUTES FORCEINLINE :: ' + name + '\n'
      modfile = kernels[nk]['mod_file'][4:]
      print modfile
      modfile = modfile.replace('INIT_INIT','INIT')
      name2 = name.replace('INIT_INIT','INIT')
      print modfile
      filename = modfile.split('_')[1].lower() + '/' + modfile.split('_')[0].lower() + '/' + name2 + '.F95'
      if not os.path.isfile(filename):
        filename = modfile.split('_')[1].lower() + '/' + modfile.split('_')[0].lower() + '/' + name + '.F95'
      if not os.path.isfile(filename):
        filename = modfile.split('_')[1].lower() + '/' + modfile.split('_')[0].lower() + '/' + name2[:-1] + '.F95'
      fid = open(filename, 'r')
      text = fid.read()
      fid.close()
      text = text.replace('recursive subroutine','subroutine')
      text = text.replace('module','!module')
      text = text.replace('contains','!contains')
      text = text.replace('end !module','!end module')
      file_text += text
      #code(kernels[nk]['mod_file'])
    code('')

##########################################################################
#  Generate wrapper to iterate over set
##########################################################################

    code('SUBROUTINE op_wrap_'+name+'( &')
    depth = depth + 2
    for g_m in range(0,ninds):
      code('& opDat'+str(invinds[g_m]+1)+'Local, &')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code('& opDat'+str(g_m+1)+'Local, &')
      elif maps[g_m] == OP_GBL:
        code('& opDat'+str(g_m+1)+'Local, &')
    if nmaps > 0:
      k = []
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
          k = k + [mapnames[g_m]]
          code('& opDat'+str(invinds[inds[g_m]-1]+1)+'Map, &')
          code('& opDat'+str(invinds[inds[g_m]-1]+1)+'MapDim, &')
    code('& bottom,top)')

    for g_m in range(0,ninds):
      code(typs[invinds[g_m]]+' opDat'+str(invinds[g_m]+1)+'Local('+str(dims[invinds[g_m]])+',*)')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code(typs[g_m]+' opDat'+str(g_m+1)+'Local('+str(dims[g_m])+',*)')
      elif maps[g_m] == OP_GBL:
        code(typs[g_m]+' opDat'+str(g_m+1)+'Local('+str(dims[g_m])+')')
    if nmaps > 0:
      k = []
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
          k = k + [mapnames[g_m]]
          code('INTEGER(kind=4) opDat'+str(invinds[inds[g_m]-1]+1)+'Map(*)')
          code('INTEGER(kind=4) opDat'+str(invinds[inds[g_m]-1]+1)+'MapDim')

    code('INTEGER(kind=4) bottom,top,i1')
    if nmaps > 0:
      k = []
      line = 'INTEGER(kind=4) '
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (not mapinds[g_m] in k):
          k = k + [mapinds[g_m]]
          line += 'map'+str(mapinds[g_m]+1)+'idx, '
      code(line[:-2])
    code('')

#
# kernel call for indirect version
#
    #If indirect kernel then add vector gather/scatter variables
    if indirect_kernel:
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (accs[g_m] == OP_READ \
          or accs[g_m] == OP_RW or accs[g_m] == OP_INC):
          code('TYP dat'+str(g_m)+'(SIMD_VEC,DIMS)')
      code('')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (accs[g_m] == OP_READ \
          or accs[g_m] == OP_RW or accs[g_m] == OP_INC):
          code('!dir$ attributes align: 64:: dat'+str(g_m))
      code('')

      code('#ifdef VECTORIZE')
      DO2('i1','bottom','((top-1)/SIMD_VEC)*SIMD_VEC','SIMD_VEC')
      code('!DIR$ SIMD')
      DO('i2','1','SIMD_VEC')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP :
          if (accs[g_m] == OP_READ or accs[g_m] == OP_RW):#and (not mapinds[g_m] in k):
            code('map'+str(g_m+1)+'idx = opDat'+str(invmapinds[inds[g_m]-1]+1)+'Map(1 + i(i1+i2-1) * opDat'+str(invmapinds[inds[g_m]-1]+1)+'MapDim + '+str(int(idxs[g_m])-1)+') + 1')
      code('')

      #setup gathers
      code('')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP :
          if (accs[g_m] == OP_READ or accs[g_m] == OP_RW):#and (not mapinds[g_m] in k):
            for d in range(0,int(dims[g_m])):
              code('dat'+str(g_m+1)+'(i2,'+str(d+1)+') = opDat'+str(g_m+1)+'Local('+str(d+1)+',map'+str(g_m+1)+'idx)')
            code('')
          elif (accs[g_m] == OP_INC):
            for d in range(0,int(dims[g_m])):
              code('dat'+str(g_m+1)+'(i,'+str(d+1)+') = 0.0')
            code('')
      ENDDO()

      #vectorized kernel call
      code('!DIR$ SIMD')
      code('!DIR$ FORCEINLINE')
      DO('i2','1','SIMD_VEC')
      code('!vecotorized kernel call')
      line = 'CALL '+name+'( &'
      indent = '\n'+' '*depth
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          line = line + indent + '& opDat'+str(g_m+1)+'Local(1,(i1+i2-1)+1), &'
        if maps[g_m] == OP_MAP:
          line = line +indent + '& dat'+str(g_m+1)+', &'
        if maps[g_m] == OP_GBL:
          line = line + indent +'& dat'+str(g_m+1)+'(DIMS*(i2-1)+1:DIMS*(i2-1)+DIMS), &'
      line = line + indent +'& i2)'
      code(line)
      ENDDO()

      #do the scatters
      FOR('i','0','SIMD_VEC')
      if nmaps > 0:
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP :
            if (accs[g_m] == OP_INC ):
              code('map'+str(g_m+1)+'idx = opDat'+str(invmapinds[inds[g_m]-1]+1)+'Map(1 + i(i1+i2-1) * opDat'+str(invmapinds[inds[g_m]-1]+1)+'MapDim + '+str(int(idxs[g_m])-1)+') + 1')
              code('')
              for d in range(0,int(dims[g_m])):
                code('opDat'+str(g_m+1)+'Local('+str(d+1)+',map'+str(g_m+1)+'idx) = opDat'+str(g_m+1)+'Local('+str(d+1)+',map'+str(g_m+1)+'idx) + dat'+str(g_m+1)+'(i2,'+str(d+1)+')')
            if (accs[g_m] == OP_WRITE or accs[g_m] == OP_RW):
                for d in range(0,int(dims[g_m])):
                  code('opDat'+str(g_m+1)+'Local('+str(d+1)+',map'+str(g_m+1)+'idx) = dat'+str(g_m+1)+'(i2,'+str(d+1)+')')
                code('')



    #do reductions


    ENDDO()
    ENDDO()
    code('!remainder')
    DO('i1','((top-1)/SIMD_VEC)*SIMD_VEC','top')
    depth = depth - 2
    code('#else')
    DO('i1','bottom','top')
    depth = depth - 2
    code('#endif')
    depth = depth + 2
    k = []
    for g_m in range(0,nargs):
      if maps[g_m] == OP_MAP and (not mapinds[g_m] in k):
        k = k + [mapinds[g_m]]
        code('map'+str(mapinds[g_m]+1)+'idx = opDat'+str(invmapinds[inds[g_m]-1]+1)+'Map(1 + i1 * opDat'+str(invmapinds[inds[g_m]-1]+1)+'MapDim + '+str(int(idxs[g_m])-1)+')+1')
    code('!kernel call')
    line = '  CALL '+name+'( &'
    indent = '\n'+' '*depth
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        line = line + indent + '& opDat'+str(g_m+1)+'Local(1,i1+1)'
      if maps[g_m] == OP_MAP:
        line = line +indent + '& opDat'+str(invinds[inds[g_m]-1]+1)+'Local(1,map'+str(mapinds[g_m]+1)+'idx)'
      if maps[g_m] == OP_GBL:
        line = line + indent +'& opDat'+str(g_m+1)+'Local(1)'
      if g_m < nargs-1:
        line = line +', &'
      else:
         line = line +' &'
    depth = depth - 2
    code(line + indent + '& )')
    depth = depth + 2

    ENDDO()
    depth = depth - 2
    code('END SUBROUTINE')

##########################################################################
#  Generate SEQ host stub
##########################################################################
    code('SUBROUTINE '+name+'_host( userSubroutine, set, &'); depth = depth + 2
    for g_m in range(0,nargs):
      if g_m == nargs-1:
        code('& opArg'+str(g_m+1)+' )')
      else:
        code('& opArg'+str(g_m+1)+', &')

    code('')
    code('IMPLICIT NONE')
    code('character(kind=c_char,len=*), INTENT(IN) :: userSubroutine')
    code('type ( op_set ) , INTENT(IN) :: set')
    code('')

    for g_m in range(0,nargs):
      code('type ( op_arg ) , INTENT(IN) :: opArg'+str(g_m+1))
    code('')

    code('type ( op_arg ) , DIMENSION('+str(nargs)+') :: opArgArray')
    code('INTEGER(kind=4) :: numberOfOpDats')
    code('INTEGER(kind=4), DIMENSION(1:8) :: timeArrayStart')
    code('INTEGER(kind=4), DIMENSION(1:8) :: timeArrayEnd')
    code('REAL(kind=8) :: startTime')
    code('REAL(kind=8) :: endTime')
    code('INTEGER(kind=4) :: returnSetKernelTiming')
    code('INTEGER(kind=4) :: n_upper')
    code('type ( op_set_core ) , POINTER :: opSetCore')
    code('')
    for g_m in range(0,ninds):
      code('INTEGER(kind=4), POINTER, DIMENSION(:) :: opDat'+str(invinds[g_m]+1)+'Map')
      code('INTEGER(kind=4) :: opDat'+str(invinds[g_m]+1)+'MapDim')
      code(typs[invinds[g_m]]+', POINTER, DIMENSION(:) :: opDat'+str(invinds[g_m]+1)+'Local')
      code('INTEGER(kind=4) :: opDat'+str(invinds[g_m]+1)+'Cardinality')
      code('')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code(typs[g_m]+', POINTER, DIMENSION(:) :: opDat'+str(g_m+1)+'Local')
        code('INTEGER(kind=4) :: opDat'+str(g_m+1)+'Cardinality')
        code('')
      if maps[g_m] == OP_GBL:
        code(typs[g_m]+', POINTER, DIMENSION(:) :: opDat'+str(g_m+1)+'Local')

    for g_m in range(0,nargs):
      if maps[g_m] == OP_MAP and optflags[g_m]==1:
        code(typs[g_m]+', POINTER, DIMENSION(:) :: opDat'+str(g_m+1)+'OptPtr')


    code('')
    code('INTEGER(kind=4) :: i1')

    code('')
    code('numberOfOpDats = '+str(nargs))
    code('')

    for g_m in range(0,nargs):
      code('opArgArray('+str(g_m+1)+') = opArg'+str(g_m+1))
    code('')

    code('returnSetKernelTiming = setKernelTime('+str(nk)+' , userSubroutine//C_NULL_CHAR, &')
    code('& 0.d0, 0.00000,0.00000, 0)')

    code('call op_timers_core(startTime)')
    code('')
    #mpi halo exchange call
    code('n_upper = op_mpi_halo_exchanges(set%setCPtr,numberOfOpDats,opArgArray)')

    code('')
    code('opSetCore => set%setPtr')
    code('')
    for g_m in range(0,ninds):
      code('opDat'+str(invinds[g_m]+1)+'Cardinality = opArg'+str(invinds[g_m]+1)+'%dim * getSetSizeFromOpArg(opArg'+str(invinds[g_m]+1)+')')
      code('opDat'+str(invinds[g_m]+1)+'MapDim = getMapDimFromOpArg(opArg'+str(invinds[g_m]+1)+')')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code('opDat'+str(g_m+1)+'Cardinality = opArg'+str(g_m+1)+'%dim * getSetSizeFromOpArg(opArg'+str(g_m+1)+')')

    for g_m in range(0,ninds):
      code('CALL c_f_pointer(opArg'+str(invinds[g_m]+1)+'%data,opDat'+str(invinds[g_m]+1)+'Local,(/opDat'+str(invinds[g_m]+1)+'Cardinality/))')
      code('CALL c_f_pointer(opArg'+str(invinds[g_m]+1)+'%map_data,opDat'+str(invinds[g_m]+1)+'Map,(/opSetCore%size*opDat'+str(invinds[g_m]+1)+'MapDim/))')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code('CALL c_f_pointer(opArg'+str(g_m+1)+'%data,opDat'+str(g_m+1)+'Local,(/opDat'+str(g_m+1)+'Cardinality/))')
      elif maps[g_m] == OP_GBL:
        code('CALL c_f_pointer(opArg'+str(g_m+1)+'%data,opDat'+str(g_m+1)+'Local, (/opArg'+str(g_m+1)+'%dim/))')
    code('')

    code('')
    if 0:
      code('CALL op_wrap_'+name+'( &')
      for g_m in range(0,ninds):
        code('& opDat'+str(invinds[g_m]+1)+'Local, &')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          code('& opDat'+str(g_m+1)+'Local, &')
        elif maps[g_m] == OP_GBL:
          code('& opDat'+str(g_m+1)+'Local, &')
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
            k = k + [mapnames[g_m]]
            code('& opDat'+str(invinds[inds[g_m]-1]+1)+'Map, &')
            code('& opDat'+str(invinds[inds[g_m]-1]+1)+'MapDim, &')
      code('& 0, opSetCore%core_size)')
    code('CALL op_mpi_wait_all(numberOfOpDats,opArgArray)')
    code('CALL op_wrap_'+name+'( &')
    for g_m in range(0,ninds):
      code('& opDat'+str(invinds[g_m]+1)+'Local, &')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code('& opDat'+str(g_m+1)+'Local, &')
      elif maps[g_m] == OP_GBL:
        code('& opDat'+str(g_m+1)+'Local, &')
    if nmaps > 0:
      k = []
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
          k = k + [mapnames[g_m]]
          code('& opDat'+str(invinds[inds[g_m]-1]+1)+'Map, &')
          code('& opDat'+str(invinds[inds[g_m]-1]+1)+'MapDim, &')
    code('& 0, n_upper)')
#    code('& opSetCore%core_size, n_upper)')


#    IF('(n_upper .EQ. 0) .OR. (n_upper .EQ. opSetCore%core_size)')
#    code('CALL op_mpi_wait_all(numberOfOpDats,opArgArray)')
#    ENDIF()
#    code('')



    code('')
    code('CALL op_mpi_set_dirtybit(numberOfOpDats,opArgArray)')
    code('')

    #reductions
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and (accs[g_m] == OP_INC or accs[g_m] == OP_MIN or accs[g_m] == OP_MAX or accs[g_m] == OP_WRITE):
        if typs[g_m] == 'real(8)' or typs[g_m] == 'REAL(kind=8)':
          code('CALL op_mpi_reduce_double(opArg'+str(g_m+1)+',opArg'+str(g_m+1)+'%data)')
        elif typs[g_m] == 'real(4)' or typs[g_m] == 'REAL(kind=4)':
          code('CALL op_mpi_reduce_float(opArg'+str(g_m+1)+',opArg'+str(g_m+1)+'%data)')
        elif typs[g_m] == 'integer(4)' or typs[g_m] == 'INTEGER(kind=4)':
          code('CALL op_mpi_reduce_int(opArg'+str(g_m+1)+',opArg'+str(g_m+1)+'%data)')
        elif typs[g_m] == 'logical' or typs[g_m] == 'logical*1':
          code('CALL op_mpi_reduce_bool(opArg'+str(g_m+1)+',opArg'+str(g_m+1)+'%data)')
        code('')

    code('call op_timers_core(endTime)')
    code('')
    code('returnSetKernelTiming = setKernelTime('+str(nk)+' , userSubroutine//C_NULL_CHAR, &')
    code('& endTime-startTime,0.00000,0.00000, 1)')
    depth = depth - 2
    code('END SUBROUTINE')
    code('END MODULE')
    code('')

##########################################################################
#  output individual kernel file
##########################################################################
    if hydra:
      name = 'kernels/'+kernels[nk]['master_file']+'/'+name
      fid = open(name+'_veckernel.F95','w')
    else:
      fid = open(name+'_veckernel.F90','w')
    date = datetime.datetime.now()
    fid.write('!\n! auto-generated by op2.py on '+date.strftime("%Y-%m-%d %H:%M")+'\n!\n\n')
    fid.write(file_text.strip())
    fid.close()
