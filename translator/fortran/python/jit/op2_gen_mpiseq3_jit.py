##########################################################################
#
# OpenMP code generator
#
# This routine is called by op2 which parses the input files
#
# It produces a file xxx_kernel.F90 for each kernel,
# plus a master kernel file
#
##########################################################################

import re
import datetime
import os

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


def op2_gen_mpiseq3_jit(master, date, consts, kernels, hydra, bookleaf):

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

  specified_kernels = 'IFLUX_EDGEK'

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
# set two logicals
#
    j = -1
    for i in range(0,nargs):
      if maps[i] == OP_MAP and (accs[i] == OP_INC or accs[i] == OP_RW):
        j = i
    ind_inc = j >= 0

    j = -1
    for i in range(0,nargs):
      if maps[i] == OP_GBL and accs[i] <> OP_READ:
        j = i
    reduct = j >= 0


    FORTRAN = 1;
    CPP     = 0;
    g_m = 0;
    file_text = ''
    depth = 0

    needDimList = []
    for g_m in range(0,nargs):
      if (not dims[g_m].isdigit()) and not (dims[g_m] in ['NPDE','DNTQMU','DNFCROW','1*1']):
        needDimList = needDimList + [g_m]

##########################################################################
#  Generate Header - Non JIT
##########################################################################
    
    
    if hydra :
      if name not in specified_kernels:
        print "skipping unspecified kernel :", name
        continue

    if hydra:
      code('MODULE '+kernels[nk]['mod_file'][4:]+'_MODULE')
    else:
      code('MODULE '+name.upper()+'_MODULE')
    code('USE OP2_FORTRAN_DECLARATIONS')
    code('USE OP2_FORTRAN_RT_SUPPORT')
    code('USE OP2_FORTRAN_JIT')
    code('USE '+master.split('.')[0]+'_SEQ')
    code('USE ISO_C_BINDING')

    if hydra == 0 and bookleaf == 0:
      code('USE OP2_CONSTANTS')
    if bookleaf:
      code('USE kinds_mod,    ONLY: ink,rlk')
      code('USE parameters_mod,ONLY: LI')

    code('')

    code('')
    code('CONTAINS')
    code('')

    code('#ifndef OP2_JIT')
    code('')
    code('#include "'+name+'.inc"')
    code('')

    #
    # save non-jit header
    #
    header = file_text
    file_text = ''


##########################################################################
#  Generate Header - JIT
##########################################################################
    if hydra:
      code('MODULE '+kernels[nk]['mod_file'][4:]+'_MODULE_EXECUTE')
    else:
      code('MODULE '+name.upper()+'_MODULE_EXECUTE')
    code('USE OP2_FORTRAN_DECLARATIONS')
    code('USE OP2_FORTRAN_RT_SUPPORT')
    code('USE ISO_C_BINDING')

    if hydra == 0 and bookleaf == 0:
      code('USE OP2_CONSTANTS')
    if bookleaf:
      code('USE kinds_mod,    ONLY: ink,rlk')
      code('USE parameters_mod,ONLY: LI')

    code('')

    code('')
    code('CONTAINS')
    code('')

    code('#include "jit_const.h"')
    code('')


    #
    # save jit header
    #
    header_jit = file_text
    file_text = ''



##########################################################################
#  Inline user kernel function
##########################################################################

    if hydra == 1:
      file_text += '!DEC$ ATTRIBUTES FORCEINLINE :: ' + name + '\n'
      modfile = kernels[nk]['mod_file'][4:]
      modfile = modfile.replace('INIT_INIT','INIT')
      name2 = name.replace('INIT_INIT','INIT')
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

      #
      # substitute npdes with DNPDE
      #
      using_npdes = 0
      for g_m in range(0,nargs):
        if var[g_m] == 'npdes':
          using_npdes = 1
      if using_npdes:
        i = re.search('\\bnpdes\\b',text)
        j = i.start()
        i = re.search('\\bnpdes\\b',text[j:])
        j = j + i.start()+5
        i = re.search('\\bnpdes\\b',text[j:])
        j = j + i.start()+5
        text = text[1:j] + re.sub('\\bnpdes\\b','NPDE',text[j:])

      file_text += text
      #code(kernels[nk]['mod_file'])
    elif bookleaf == 1:
      file_text += '!DEC$ ATTRIBUTES FORCEINLINE :: ' + name + '\n'
      modfile = kernels[nk]['mod_file']
      prefixes=['./','ale/','utils/','io/','eos/','hydro/','mods/']
      prefix_i=0
      while (prefix_i<7 and (not os.path.exists(prefixes[prefix_i]+modfile))):
        prefix_i=prefix_i+1
      fid = open(prefixes[prefix_i]+modfile, 'r')
      text = fid.read()
      i = re.search('SUBROUTINE '+name+'\\b',text).start() #text.find('SUBROUTINE '+name)
      j = i + 10 + text[i+10:].find('SUBROUTINE '+name) + 11 + len(name)
      file_text += text[i:j]+'\n\n'
    else:
      comm('user function')
      code('#include "'+name+'.inc"')
      code('')

    code('')

    #
    # save user kernel
    #
    user_kernel = file_text
    file_text = ''

##########################################################################
#  Generate wrapper to iterate over set
##########################################################################

    code('SUBROUTINE op_wrap_'+name+'( &')
    depth = depth + 2
    for g_m in range(0,ninds):
      if invinds[g_m] in needDimList:
        code('& opDat'+str(invinds[g_m]+1)+'Dim, &')
      code('& opDat'+str(invinds[g_m]+1)+'Local, &')
    for g_m in range(0,nargs):
      if maps[g_m] <> OP_MAP:
        if g_m in needDimList:
          code('& opDat'+str(g_m+1)+'Dim, &')
        code('& opDat'+str(g_m+1)+'Local, &')
    if nmaps > 0:
      k = []
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
          k = k + [mapnames[g_m]]
          code('& opDat'+str(invinds[inds[g_m]-1]+1)+'Map, &')
          code('& opDat'+str(invinds[inds[g_m]-1]+1)+'MapDim, &')
    code('& bottom,top)')
    code('implicit none')
    for g_m in range(0,ninds):
      if invinds[g_m] in needDimList:
        code('INTEGER(kind=4) opDat'+str(invinds[g_m]+1)+'Dim')
        code(typs[invinds[g_m]]+' opDat'+str(invinds[g_m]+1)+'Local(opDat'+str(invinds[g_m]+1)+'Dim,*)')
      else:
        code(typs[invinds[g_m]]+' opDat'+str(invinds[g_m]+1)+'Local('+str(dims[invinds[g_m]])+',*)')
    for g_m in range(0,nargs):
      if maps[g_m] <> OP_MAP:
        if g_m in needDimList:
          code('INTEGER(kind=4) opDat'+str(g_m+1)+'Dim')
      if maps[g_m] == OP_ID:
        if g_m in needDimList:
          code(typs[g_m]+' opDat'+str(g_m+1)+'Local(opDat'+str(g_m+1)+'Dim,*)')
        else:
          code(typs[g_m]+' opDat'+str(g_m+1)+'Local('+str(dims[g_m])+',*)')
      elif maps[g_m] == OP_GBL:
        if g_m in needDimList:
          code(typs[g_m]+' opDat'+str(g_m+1)+'Local(opDat'+str(g_m+1)+'Dim)')
        else:
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
#    if ind_inc == 0 and reduct == 0:
#      code('!DIR$ simd')
    DO('i1','bottom','top')
    k = []
    for g_m in range(0,nargs):
      if maps[g_m] == OP_MAP and (not mapinds[g_m] in k):
        k = k + [mapinds[g_m]]
        code('map'+str(mapinds[g_m]+1)+'idx = opDat'+str(invmapinds[inds[g_m]-1]+1)+'Map(1 + i1 * opDat'+str(invmapinds[inds[g_m]-1]+1)+'MapDim + '+str(int(idxs[g_m])-1)+')+1')
    comm('kernel call')
    line = 'CALL '+name+'( &'
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
    code('')

    #
    # save user kernel
    #
    wrapper = file_text
    file_text = ''

##########################################################################
#  Generate SEQ host stub header - non JIT version
##########################################################################
    code('SUBROUTINE '+name+'_host( userSubroutine, set, &')
    #
    # save user kernel
    #
    sig = file_text
    file_text = ''

##########################################################################
#  Generate SEQ host stub header - JIT version
##########################################################################
    code('SUBROUTINE '+name+'_host_rec( userSubroutine, set, &')
    #
    # save user kernel
    #
    sig_JIT = file_text
    file_text = ''


##########################################################################
#  Generate SEQ host stub - common body
##########################################################################

    depth = depth + 2
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
    code('REAL(kind=4) :: dataTransfer')

    code('')
    code('numberOfOpDats = '+str(nargs))
    code('')

    for g_m in range(0,nargs):
      code('opArgArray('+str(g_m+1)+') = opArg'+str(g_m+1))
    code('')

    code('returnSetKernelTiming = setKernelTime('+str(nk)+' , userSubroutine//C_NULL_CHAR, &')
    code('& 0.0_8, 0.00000_4,0.00000_4, 0)')

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
    if 1:
      code('CALL op_wrap_'+name+'( &')
      for g_m in range(0,ninds):
        if invinds[g_m] in needDimList:
          code('& opArg'+str(invinds[g_m]+1)+'%dim, &')
        code('& opDat'+str(invinds[g_m]+1)+'Local, &')
      for g_m in range(0,nargs):
        if maps[g_m] <> OP_MAP:
          if g_m in needDimList:
            code('& opArg'+str(g_m+1)+'%dim, &')
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
      if invinds[g_m] in needDimList:
          code('& opArg'+str(invinds[g_m]+1)+'%dim, &')
      code('& opDat'+str(invinds[g_m]+1)+'Local, &')
    for g_m in range(0,nargs):
      if maps[g_m] <> OP_MAP:
        if g_m in needDimList:
            code('& opArg'+str(g_m+1)+'%dim, &')
        code('& opDat'+str(g_m+1)+'Local, &')

    if nmaps > 0:
      k = []
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
          k = k + [mapnames[g_m]]
          code('& opDat'+str(invinds[inds[g_m]-1]+1)+'Map, &')
          code('& opDat'+str(invinds[inds[g_m]-1]+1)+'MapDim, &')
    #code('& 0, n_upper)')
    code('& opSetCore%core_size, n_upper)')


    IF('(n_upper .EQ. 0) .OR. (n_upper .EQ. opSetCore%core_size)')
    code('CALL op_mpi_wait_all(numberOfOpDats,opArgArray)')
    ENDIF()
    code('')



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
    code('dataTransfer = 0.0')
    if ninds == 0:
      for g_m in range(0,nargs):
        if optflags[g_m] == 1:
              IF('opArg'+str(g_m+1)+'%opt == 1')
        if accs[g_m] == OP_READ or accs[g_m] == OP_WRITE:
          if maps[g_m] == OP_GBL:
            code('dataTransfer = dataTransfer + opArg'+str(g_m+1)+'%size')
          else:
            code('dataTransfer = dataTransfer + opArg'+str(g_m+1)+'%size * opSetCore%size')
        else:
          if maps[g_m] == OP_GBL:
            code('dataTransfer = dataTransfer + opArg'+str(g_m+1)+'%size * 2.d0')
          else:
            code('dataTransfer = dataTransfer + opArg'+str(g_m+1)+'%size * opSetCore%size * 2.d0')
        if optflags[g_m] == 1:
          ENDIF()
    else:
      names = []
      for g_m in range(0,ninds):
        mult=''
        if indaccs[g_m] <> OP_WRITE and indaccs[g_m] <> OP_READ:
          mult = ' * 2.d0'
        if not var[invinds[g_m]] in names:
          if optflags[invinds[g_m]] == 1:
            IF('opArg'+str(g_m+1)+'%opt == 1')
          code('dataTransfer = dataTransfer + opArg'+str(invinds[g_m]+1)+'%size * MIN(n_upper,getSetSizeFromOpArg(opArg'+str(invinds[g_m]+1)+'))'+mult)
          names = names + [var[invinds[g_m]]]
          if optflags[invinds[g_m]] == 1:
            ENDIF()
      for g_m in range(0,nargs):
        mult=''
        if accs[g_m] <> OP_WRITE and accs[g_m] <> OP_READ:
          mult = ' * 2.d0'
        if not var[g_m] in names:
          if optflags[g_m] == 1:
            IF('opArg'+str(g_m+1)+'%opt == 1')
          names = names + [var[invinds[g_m]]]
          if maps[g_m] == OP_ID:
            code('dataTransfer = dataTransfer + opArg'+str(g_m+1)+'%size * MIN(n_upper,getSetSizeFromOpArg(opArg'+str(g_m+1)+'))'+mult)
          elif maps[g_m] == OP_GBL:
            code('dataTransfer = dataTransfer + opArg'+str(g_m+1)+'%size'+mult)
          if optflags[g_m] == 1:
            ENDIF()
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
            k = k + [mapnames[g_m]]
            code('dataTransfer = dataTransfer + n_upper * opDat'+str(invinds[inds[g_m]-1]+1)+'MapDim * 4.d0')

    code('returnSetKernelTiming = setKernelTime('+str(nk)+' , userSubroutine//C_NULL_CHAR, &')
    code('& endTime-startTime, dataTransfer, 0.00000_4, 1)')
    #code('returnSetKernelTiming = setKernelTime('+str(nk)+' , userSubroutine//C_NULL_CHAR, &')
    #code('& endTime-startTime,0.00000,0.00000, 1)')
    depth = depth - 2
    code('END SUBROUTINE')
    code('END MODULE')
    code('')


    #
    # Save body
    #
    body_common = file_text
    file_text = ''

    code('')
    code('#else !OP2_JIT defined')
    code('')


################################################################################
#  Generate SEQ host stub - routine that gets called to compile the JIT version
################################################################################
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

    comm('Define interface of call-back routine.')
    code('abstract interface')
    code('subroutine called_proc (userSubroutine, set, &')
    for g_m in range(0,nargs):
      if g_m == nargs-1:
        code('& opArg'+str(g_m+1)+' ) bind(c)')
      else:
        code('& opArg'+str(g_m+1)+', &')
    depth = depth + 2

    code('')
    code('USE, intrinsic :: iso_c_binding')
    code('USE OP2_FORTRAN_DECLARATIONS')
    code('USE OP2_FORTRAN_RT_SUPPORT')
    code('USE OP2_CONSTANTS')

    code('character (kind=c_char), dimension(*), intent(in) :: userSubroutine')
    code('type ( op_set ) , INTENT(IN) :: set')
    for g_m in range(0,nargs):
      code('type ( op_arg ) , INTENT(IN) :: opArg'+str(g_m+1))
    code('')
    depth = depth - 2
    code('end subroutine called_proc')
    code('end interface')
    comm('End interface of call-back routine')
    code('')

    code('procedure(called_proc), bind(c), pointer :: proc_'+name)
    code('')
    IF('.not. JIT_COMPILED')
    code('call jit_compile()')
    ENDIF()

    code('call c_f_procpointer( proc_addr_'+name+', proc_'+name+' )')
    code('')

    comm('call/execute dynamically loaded procedure with the parameters from '+name+'_host() signature')
    code('call proc_'+name+'( userSubroutine, set, &')
    for g_m in range(0,nargs):
      if g_m == nargs-1:
        code('& opArg'+str(g_m+1)+' )')
      else:
        code('& opArg'+str(g_m+1)+', &')

    code('')
    depth = depth - 2
    code('END SUBROUTINE')
    code('')
    code('END MODULE')
    code('#endif !OP2_JIT')
    code('')


##########################################################################
#  output individual kernel file - non JIT version
##########################################################################
    if hydra:
      new_name = 'kernels/'+kernels[nk]['master_file']+'/'+name
      fid = open(new_name+'_seqkernel.F95','w')
    elif bookleaf:
      fid = open(prefixes[prefix_i]+name+'_seqkernel.f90','w')
    else:
      if not os.path.exists('seq'):
        os.makedirs('seq')
      fid = open('seq/'+name+'_seqkernel.F90','w')
    date = datetime.datetime.now()
    fid.write('!\n! auto-generated by op2.py\n!\n\n')
    fid.write(header+ wrapper + sig + \
      body_common+file_text.strip())
    fid.close()


##########################################################################
#  output individual kernel file - JIT version
##########################################################################
    if hydra:
      new_name = 'kernels/'+kernels[nk]['master_file']+'/'+name
      fid = open(new_name+'_seqkernel_rec.F90','w')
    elif bookleaf:
      fid = open(prefixes[prefix_i]+name+'_seqkernel_rec.f90','w')
    else:
      if not os.path.exists('seq'):
        os.makedirs('seq')
      fid = open('seq/'+name+'_seqkernel_rec.F90','w')
    date = datetime.datetime.now()
    fid.write('!\n! auto-generated by op2.py\n!\n\n')
    fid.write(header_jit+ user_kernel + wrapper + sig_JIT + \
      body_common.strip())
    fid.close()



##########################################################################
#  output one master kernel file
##########################################################################

  file_text =''
  code('module '+master.split('.')[0]+'_seq')

  code('use OP2_Fortran_RT_Support')
  code('USE OP2_FORTRAN_JIT')
  code('use OP2_CONSTANTS')
  code('use, intrinsic :: iso_c_binding')
  code('')

  comm('JIT function pointers')
  for nk in range(0,len(kernels)):
    code('type(c_funptr) :: proc_addr_'+kernels[nk]['name'])
  code('')

  code('type(c_ptr) :: handle')
  code('')

  code('contains')
  code('')

  comm('Function to compile functions in JIT')
  code('subroutine jit_compile ()')
  code('use, intrinsic :: iso_c_binding')

  code('IMPLICIT NONE')
  code('')
  depth += 2
  code('integer(c_int), parameter :: RTLD_LAZY=1 ! value extracte from the C header file')
  code('integer STATUS')
  code('')

  IF('op_is_root() .eq. 1')
  comm('compile *_seqkernel_rec.F90 using system command')
  code('write(*,*) "JIT compiling op_par_loops"')
  code('call execute_command_line ("make -j genseq_jit", exitstat=STATUS)')
  ENDIF()
  code('call op_mpi_barrier()')
  code('')

  comm('dynamically load '+master.split('.')[0]+'_seqkernel_rec.so')
  code('handle=dlopen("./'+master.split('.')[0]+'_seqkernel_rec.so"//c_null_char, RTLD_LAZY)')
  IF('.not. c_associated(handle)')
  code('WRITE(*,*)"Unable to load DLL ./'+master.split('.')[0]+'_seqkernel_rec.so: ", CToFortranString(DLError())')
  code('stop')
  ENDIF()
  code('')

  for nk in range(0,len(kernels)):
    if hydra :
      name = kernels[nk]['mod_file'][4:].lower()+'_module_execute_mp_'+kernels[nk]['name'].lower()+'_host_rec_'
      if kernels[nk]['name'] not in specified_kernels:
        continue
    else:
      name = kernels[nk]['name']+'_module_execute_mp_'+kernels[nk]['name']+'_host_rec_'

    IF('.not. c_associated(proc_addr_'+kernels[nk]['name']+')')
    code('proc_addr_'+kernels[nk]['name']+'=dlsym(handle, "'+name+'"//c_null_char)')
    IF('.not. c_associated(proc_addr_'+kernels[nk]['name']+')')
    code('write(*,*) "Unable to load the procedure '+name+'"')
    code('stop')
    ENDIF()
    ENDIF()
    code('')

  IF('op_is_root() .eq. 1')
  code('write(*,*) "Sucessfully Loaded JIT Compiled Modules"')
  ENDIF()
  code('JIT_COMPILED = .true.')
  code('call op_mpi_barrier()')
  code('')

  depth -= 2
  code('end subroutine jit_compile')

  code('END MODULE '+master.split('.')[0]+'_seq')

  master = master.split('.')[0]
  fid = open('seq/'+master.split('.')[0]+'_seqkernels.F90','w')
  fid.write('!\n! auto-generated by op2.py\n!\n\n')
  fid.write(file_text)
  fid.close()