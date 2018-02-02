##########################################################################
#
# OpenMP code generator
#
# This routine is called by op2_fortran which parses the input files
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


def op2_gen_openmp(master, date, consts, kernels, hydra):

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
    optflags = kernels[nk]['optflags']
    ninds   = kernels[nk]['ninds']
    inddims = kernels[nk]['inddims']
    indaccs = kernels[nk]['indaccs']
    indtyps = kernels[nk]['indtyps']
    invinds = kernels[nk]['invinds']

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
      if maps[i] == OP_GBL and accs[i] <> OP_READ:
        j = i
    reduct = j >= 0


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
    code('#ifdef _OPENMP'); depth = depth + 2
    code('USE OMP_LIB'); depth = depth - 2
    code('#endif')

##########################################################################
#  Variable declarations
##########################################################################
    code('')
    comm('variable declarations')

    code('')

    if ninds > 0: #if indirect loop
      code('LOGICAL :: firstTime_'+name+' = .TRUE.')
      code('type ( c_ptr )  :: planRet_'+name)
      code('type ( op_plan ) , POINTER :: actualPlan_'+name)
      code('type ( c_ptr ) , POINTER, DIMENSION(:) :: ind_maps_'+name)
      code('type ( c_ptr ) , POINTER, DIMENSION(:) :: mappingArray_'+name)
      code('')
      for g_m in range(0,ninds):
        code('INTEGER(kind=4), POINTER, DIMENSION(:) :: ind_maps'+str(invinds[g_m]+1)+'_'+name)
      code('')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP:
          code('INTEGER(kind=2), POINTER, DIMENSION(:) :: mappingArray'+str(g_m+1)+'_'+name)
      code('')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP:
          code('INTEGER(kind=4) :: mappingArray'+str(g_m+1)+'Size_'+name)
      code('')
      code('INTEGER(kind=4), POINTER, DIMENSION(:) :: blkmap_'+name)
      code('INTEGER(kind=4) :: blkmapSize_'+name)
      code('INTEGER(kind=4), POINTER, DIMENSION(:) :: ind_offs_'+name)
      code('INTEGER(kind=4) :: ind_offsSize_'+name)
      code('INTEGER(kind=4), POINTER, DIMENSION(:) :: ind_sizes_'+name)
      code('INTEGER(kind=4) :: ind_sizesSize_'+name)
      code('INTEGER(kind=4), POINTER, DIMENSION(:) :: nelems_'+name)
      code('INTEGER(kind=4) :: nelemsSize_'+name)
      code('INTEGER(kind=4), POINTER, DIMENSION(:) :: nthrcol_'+name)
      code('INTEGER(kind=4) :: nthrcolSize_'+name)
      code('INTEGER(kind=4), POINTER, DIMENSION(:) :: offset_'+name)
      code('INTEGER(kind=4) :: offsetSize_'+name)
      code('INTEGER(kind=4), POINTER, DIMENSION(:) :: thrcol_'+name)
      code('INTEGER(kind=4) :: thrcolSize_'+name)
      code('INTEGER(kind=4), POINTER, DIMENSION(:) :: ncolblk_'+name)
      code('INTEGER(kind=4), POINTER, DIMENSION(:) :: pnindirect_'+name)

##########################################################################
#  Inline user kernel function
##########################################################################
    code('')
    code('CONTAINS')
    code('')
    if hydra == 0:
      comm('user function')
      code('#include "'+name+'.inc"')
      code('')
    else:
      modfile = kernels[nk]['mod_file'][4:]
      filename = modfile.split('_')[1].lower() + '/' + modfile.split('_')[0].lower() + '/' + name + '.F95'
      if not os.path.isfile(filename):
        filename = modfile.split('_')[1].lower() + '/' + modfile.split('_')[0].lower() + '/' + name[:-1] + '.F95'
      fid = open(filename, 'r')
      text = fid.read()
      fid.close()
      text = text.replace('module','!module')
      text = text.replace('contains','!contains')
      text = text.replace('end !module','!end module')
      file_text += text
      #code(kernels[nk]['mod_file'])
    code('')

##########################################################################
#  Generate OpenMP kernel function
##########################################################################
    comm('x86 kernel function')
    code('SUBROUTINE op_x86_'+name+'( &'); depth = depth + 2
    if nopts>0:
      code('&  optflags,        &')
    for g_m in range(0,ninds):
      code('&  opDat'+str(invinds[g_m]+1)+',   &')

    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code('&  opDat'+str(g_m+1)+',   &')
      elif maps[g_m] == OP_GBL:
        code('&  opDat'+str(g_m+1)+',   &')

    if ninds > 0: #indirect loop
      for g_m in range(0,ninds):
        code('&  ind_maps'+str(invinds[g_m]+1)+', &')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP:
          code('&  mappingArray'+str(g_m+1)+', &')
      code('&  ind_sizes, &')
      code('&  ind_offs,  &')
      code('&  blkmap,        &')
      code('&  offset,        &')
      code('&  nelems,        &')
      code('&  nthrcol,       &')
      code('&  thrcol,        &')
      code('&  blockOffset,   &')
      code('&  blockID )')
      code('')
    else: #direct loop
      code('& sliceStart, &')
      code('& sliceEnd )')
      code('')


    code('IMPLICIT NONE')
    code('')

##########################################################################
#  Declare local variables
##########################################################################
    comm('local variables')
    if nopts>0:
      code('INTEGER(kind=4) :: optflags')
    if ninds > 0: #indirect loop
      for g_m in range(0,ninds):
        code(typs[invinds[g_m]]+', DIMENSION(0:*) :: opDat'+str(invinds[g_m]+1))

      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          code(typs[g_m]+', DIMENSION(0:*) :: opDat'+str(g_m+1))
        elif maps[g_m] == OP_GBL:
          if dims[g_m].isdigit() and int(dims[g_m]) == 1:
            code(typs[g_m]+' :: opDat'+str(g_m+1))
            #code(typs[g_m]+', DIMENSION(1) :: opDat'+str(g_m+1))
          else:
            code(typs[g_m]+', DIMENSION(0:'+dims[g_m]+'-1) :: opDat'+str(g_m+1))

      code('')
      for g_m in range(0,ninds):
        code('INTEGER(kind=4), DIMENSION(0:), target :: ind_maps'+str(invinds[g_m]+1))
      code('')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP:
          code('INTEGER(kind=2), DIMENSION(0:*) :: mappingArray'+str(g_m+1))
      code('')
      code('INTEGER(kind=4), DIMENSION(0:*) :: ind_sizes')
      code('INTEGER(kind=4), DIMENSION(0:*) :: ind_offs')
      code('INTEGER(kind=4), DIMENSION(0:*) :: blkmap')
      code('INTEGER(kind=4), DIMENSION(0:*) :: offset')
      code('INTEGER(kind=4), DIMENSION(0:*) :: nelems')
      code('INTEGER(kind=4), DIMENSION(0:*) :: nthrcol')
      code('INTEGER(kind=4), DIMENSION(0:*) :: thrcol')
      code('INTEGER(kind=4) :: blockOffset')
      code('INTEGER(kind=4) :: blockID')
      code('INTEGER(kind=4) :: threadBlockOffset')
      code('INTEGER(kind=4) :: threadBlockID')
      code('INTEGER(kind=4) :: numberOfActiveThreads')
      code('INTEGER(kind=4) :: i1')
      code('INTEGER(kind=4) :: i2')
      add_real = 0
      add_int = 0
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP:
          if 'real' in typs[g_m].lower():
            add_real = 1
          elif 'integer' in typs[g_m].lower():
            add_int = 1
      if add_real:
        code('REAL(kind=8), DIMENSION(0:128000 - 1), target :: sharedFloat8')
      if add_int:
        code('INTEGER(kind=4), DIMENSION(0:128000 - 1), target :: sharedInt8')
      code('')

      for g_m in range(0,ninds):
        code('INTEGER(kind=4), POINTER, DIMENSION(:) :: opDat'+str(invinds[g_m]+1)+'IndirectionMap')
        code(typs[invinds[g_m]]+', POINTER, DIMENSION(:) :: opDat'+str(invinds[g_m]+1)+'SharedIndirection')

# for indirect OP_READ, we would pass in a pointer to shared, offset by map, but if opt, then map may not exist, thus we need a separate pointer
      for g_m in range(0,nargs):
        if (accs[g_m] == OP_READ or accs[g_m] == OP_RW or accs[g_m] == OP_WRITE) and maps[g_m] == OP_MAP and optflags[g_m]==1:
          code(typs[g_m]+', POINTER, DIMENSION(:) :: opDat'+str(g_m+1)+'OptPtr')

      for g_m in range(0,ninds):
        code('INTEGER(kind=4) :: opDat'+str(invinds[g_m]+1)+'nBytes')

      for g_m in range(0,ninds):
        code('INTEGER(kind=4) :: opDat'+str(invinds[g_m]+1)+'RoundUp')
      for g_m in range(0,ninds):
        code('INTEGER(kind=4) :: opDat'+str(invinds[g_m]+1)+'SharedIndirectionSize')

      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (accs[g_m] == OP_INC):
          code('REAL(kind=8), DIMENSION(0:'+dims[g_m]+'-1) :: opDat'+str(g_m+1)+'Local')
          code('INTEGER(kind=4) :: opDat'+str(g_m+1)+'Map')


      code('INTEGER(kind=4) :: numOfColours')
      code('INTEGER(kind=4) :: numberOfActiveThreadsCeiling')
      code('INTEGER(kind=4) :: colour1')
      code('INTEGER(kind=4) :: colour2')
      code('')
      code('threadBlockID = blkmap(blockID + blockOffset)')
      code('numberOfActiveThreads = nelems(threadBlockID)')
      code('threadBlockOffset = offset(threadBlockID)')
      code('numberOfActiveThreadsCeiling = numberOfActiveThreads')
      code('numOfColours = nthrcol(threadBlockID)')
      code('')

      for g_m in range(0,ninds):
        code('opDat'+str(invinds[g_m]+1)+'SharedIndirectionSize = ind_sizes('+str(g_m)+' + threadBlockID * '+str(ninds)+')')

      for g_m in range(0,ninds):
        code('opDat'+str(invinds[g_m]+1)+'IndirectionMap => ind_maps'+str(invinds[g_m]+1)+'(ind_offs('+str(g_m)+' + threadBlockID * '+str(ninds)+'):)')

      for g_m in range(0,ninds):
        code('opDat'+str(invinds[g_m]+1)+'RoundUp = opDat'+str(invinds[g_m]+1)+'SharedIndirectionSize * ('+inddims[g_m]+')')
        code('opDat'+str(invinds[g_m]+1)+'RoundUp = opDat'+str(invinds[g_m]+1)+'RoundUp + MOD(opDat'+str(invinds[g_m]+1)+'RoundUp,2)')

      for g_m in range(0,ninds):
        if g_m>0 and indopts[g_m-1] >= 0:
          IF('BTEST(optflags,'+str(optidxs[indopts[g_m-1]])+')')
        if g_m == 0:
          code('opDat'+str(invinds[g_m]+1)+'nBytes = 0')
        else:
          prev_size = 0
          if 'real' in typs[invinds[g_m-1]].lower():
            prev_size = 8
          elif 'integer' in typs[invinds[g_m-1]].lower():
            prev_size = 4
          this_size = 0
          if 'real' in typs[invinds[g_m]].lower():
            this_size = 8
          elif 'integer' in typs[invinds[g_m]].lower():
            this_size = 4
          if this_size == 0 or prev_size == 0:
            print "ERROR: Unrecognized type"
          code('opDat'+str(invinds[g_m]+1)+'nBytes = opDat'+str(invinds[g_m-1]+1)+'nBytes * '+str(prev_size)+\
          ' / '+str(this_size)+' + opDat'+str(invinds[g_m-1]+1)+'RoundUp * '+str(prev_size)+' / '+str(this_size))
        if g_m>0 and indopts[g_m-1] >= 0:
          ELSE()
          if g_m==0:
            code('opDat'+str(invinds[g_m]+1)+'nBytes = 0')
          else:
            code('opDat'+str(invinds[g_m]+1)+'nBytes = opDat'+str(invinds[g_m-1]+1)+'nBytes * '+str(prev_size)+\
            ' / '+str(this_size))
          ENDIF()

      for g_m in range(0,ninds):
        if 'REAL' in typs[invinds[g_m]].upper():
          code('opDat'+str(invinds[g_m]+1)+'SharedIndirection => sharedFloat8(opDat'+str(invinds[g_m]+1)+'nBytes:)')
        if 'INTEGER' in typs[invinds[g_m]].upper():
          code('opDat'+str(invinds[g_m]+1)+'SharedIndirection => sharedInt8(opDat'+str(invinds[g_m]+1)+'nBytes:)')
      code('')
      for g_m in range(0,ninds):
        if indopts[g_m]>=0:
          IF('BTEST(optflags,'+str(optidxs[indopts[g_m]])+')')
        DO('i1','0','opDat'+str(invinds[g_m]+1)+'SharedIndirectionSize')
        DO('i2','0', inddims[g_m])
        if accs[invinds[g_m]] == OP_READ or accs[invinds[g_m]] == OP_RW or accs[invinds[g_m]] == OP_WRITE:
          code('opDat'+str(invinds[g_m]+1)+'SharedIndirection(i2 + i1 * ('+inddims[g_m]+\
          ') + 1) = opDat'+str(invinds[g_m]+1)+'(i2 + opDat'+str(invinds[g_m]+1)+\
          'IndirectionMap(i1 + 1) * ('+inddims[g_m]+'))')
        elif accs[invinds[g_m]] == OP_INC:
          code('opDat'+str(invinds[g_m]+1)+'SharedIndirection(i2 + i1 * ('+inddims[g_m]+\
          ') + 1) = 0')
        ENDDO()
        ENDDO()
        if indopts[g_m]>=0:
          ENDIF()
        code('')

      DO('i1','0','numberOfActiveThreadsCeiling')
      code('  colour2 = -1')
      IF('i1 < numberOfActiveThreads')

      for g_m in range(0,nargs):
        if accs[g_m] == OP_INC and maps[g_m] == OP_MAP:
          DO('i2','0',dims[g_m])
          code('opDat'+str(g_m+1)+'Local(i2) = 0')
          ENDDO()

      for g_m in range(0,nargs):
        if (accs[g_m] == OP_READ or accs[g_m] == OP_RW or accs[g_m] == OP_WRITE) and maps[g_m] == OP_MAP and optflags[g_m]==1:
          IF('BTEST(optflags,'+str(optidxs[g_m])+')')
          if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
            code('opDat'+str(g_m+1)+'OptPtr => opDat'+str(invinds[inds[g_m]-1]+1)+'SharedIndirection(1+mappingArray'+str(g_m+1)+'(i1 + threadBlockOffset) * ('+dims[g_m]+'):)')
          else:
            code('opDat'+str(g_m+1)+'OptPtr => opDat'+str(invinds[inds[g_m]-1]+1)+'SharedIndirection(1+mappingArray'+str(g_m+1)+'(i1 + threadBlockOffset) * 1:)')
          ELSE()
          code('opDat'+str(g_m+1)+'OptPtr => opDat'+str(invinds[inds[g_m]-1]+1)+'SharedIndirection(0:)')
          ENDIF()


    else: #direct loop
      for g_m in range(0,nargs):
        if maps[g_m] <> OP_GBL:
          code(typs[g_m]+', DIMENSION(0:*) :: opDat'+str(g_m+1))
        else: #global arg
          if dims[g_m].isdigit() and int(dims[g_m]) == 1:
            code(typs[g_m]+' :: opDat'+str(g_m+1))
            #code(typs[g_m]+', DIMENSION(1) :: opDat'+str(g_m+1))
          else:
            code(typs[g_m]+', DIMENSION(0:'+dims[g_m]+'-1) :: opDat'+str(g_m+1))

      code('INTEGER(kind=4) :: sliceStart')
      code('INTEGER(kind=4) :: sliceEnd')
      code('INTEGER(kind=4) :: i1')


##########################################################################
#  x86 kernel call
##########################################################################

    if ninds > 0: #indirect kernel call
      code('')
      comm('kernel call')
      line = 'CALL '+name+'( &'
      indent = '\n'+' '*depth
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
            line = line + indent + '& opDat'+str(g_m+1)+'((i1 + threadBlockOffset) * ('+dims[g_m]+'):(i1 + threadBlockOffset) * ('+dims[g_m]+') + '+dims[g_m]+' - 1)'
          else:
            line = line + indent + '& opDat'+str(g_m+1)+'((i1 + threadBlockOffset) * 1)'
        if maps[g_m] == OP_MAP and (accs[g_m] == OP_READ or accs[g_m] == OP_RW or accs[g_m] == OP_WRITE) and optflags[g_m]==0:
          if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
            line = line +indent + '& opDat'+str(invinds[inds[g_m]-1]+1)+'SharedIndirection(1 + mappingArray'+str(g_m+1)+'(i1 + threadBlockOffset) * ('+dims[g_m]+'):1 + mappingArray'+str(g_m+1)+'(i1 + threadBlockOffset) * ('+dims[g_m]+') + '+dims[g_m]+' - 1)'
          else:
            line = line +indent + '& opDat'+str(invinds[inds[g_m]-1]+1)+'SharedIndirection(1 + mappingArray'+str(g_m+1)+'(i1 + threadBlockOffset) * 1)'
        elif maps[g_m] == OP_MAP and (accs[g_m] == OP_READ or accs[g_m] == OP_RW or accs[g_m] == OP_WRITE) and optflags[g_m]==1:
          if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
            line = line +indent + '& opDat'+str(g_m+1)+'OptPtr(1:'+dims[g_m]+')'
          else:
            line = line +indent + '& opDat'+str(g_m+1)+'OptPtr(1)'
        elif maps[g_m] == OP_MAP and accs[g_m] == OP_INC:
          if dims[g_m].isdigit() and int(dims[g_m])==1:
            line = line +indent + '& opDat'+str(g_m+1)+'Local(0)'
          else:
            line = line +indent + '& opDat'+str(g_m+1)+'Local'
        if maps[g_m] == OP_GBL:
          line = line + indent +'& opDat'+str(g_m+1)
        if g_m < nargs-1:
          line = line +', &'
        else:
           line = line +' &'
      depth = depth - 2
      code(line + indent + '& )')
      depth = depth + 2
      code('colour2 = thrcol(i1 + threadBlockOffset)')
      ENDIF()

      code('')
      for g_m in range(0,nargs):
        if accs[g_m] == OP_INC and maps[g_m] == OP_MAP:
          if optflags[g_m]:
            IF('BTEST(optflags,'+str(optidxs[g_m])+')')
          code('opDat'+str(g_m+1)+'Map = mappingArray'+str(g_m+1)+'(i1 + threadBlockOffset)')
          if optflags[g_m]:
            ENDIF()

      code('')
      DO('colour1','0','numOfColours')
      IF('colour2 .EQ. colour1')
      for g_m in range(0,nargs):
        if optflags[g_m]==1 and maps[g_m]==OP_MAP and accs[g_m] == OP_INC:
          IF('BTEST(optflags,'+str(optidxs[g_m])+')')
        if accs[g_m] == OP_INC and maps[g_m] == OP_MAP:
          DO('i2','0',dims[g_m])
          code('opDat'+str(invinds[inds[g_m]-1]+1)+'SharedIndirection(1 + (i2 + opDat'+str(g_m+1)+'Map * ('+dims[g_m]+'))) = opDat'+str(invinds[inds[g_m]-1]+1)+'SharedIndirection(1 + (i2 + opDat'+str(g_m+1)+'Map * ('+dims[g_m]+'))) + opDat'+str(g_m+1)+'Local(i2)')
          ENDDO()
        if optflags[g_m]==1 and maps[g_m]==OP_MAP and (accs[g_m] == OP_INC):
          ENDIF()
        if maps[g_m]==OP_MAP and accs[g_m] == OP_INC:
          code('')
      ENDIF()
      ENDDO()
      ENDDO()
      code('')
      for g_m in range(0,ninds):
        if indopts[g_m]>=0 and (accs[invinds[g_m]]==OP_INC or accs[invinds[g_m]]==OP_WRITE or accs[invinds[g_m]]==OP_RW):
          IF('BTEST(optflags,'+str(optidxs[indopts[g_m]])+')')
        if accs[invinds[g_m]] == OP_INC:
          DO('i1','0','opDat'+str(invinds[g_m]+1)+'SharedIndirectionSize')
          DO('i2','0',inddims[g_m])
          code('opDat'+str(invinds[g_m]+1)+'(i2 + opDat'+str(invinds[g_m]+1)+'IndirectionMap(i1 + 1) * ('+inddims[g_m]+')) = opDat'+str(invinds[g_m]+1)+'(i2 + opDat'+str(invinds[g_m]+1)+'IndirectionMap(i1 + 1) * ('+inddims[g_m]+')) + opDat'+str(invinds[g_m]+1)+'SharedIndirection(1 + (i2 + i1 * ('+inddims[g_m]+')))')
          ENDDO()
          ENDDO()
        if accs[invinds[g_m]] == OP_RW or accs[invinds[g_m]] == OP_WRITE:
          DO('i1','0','opDat'+str(invinds[g_m]+1)+'SharedIndirectionSize')
          DO('i2','0',inddims[g_m])
          code('opDat'+str(invinds[g_m]+1)+'(i2 + opDat'+str(invinds[g_m]+1)+'IndirectionMap(i1 + 1) * ('+inddims[g_m]+')) = opDat'+str(invinds[g_m]+1)+'SharedIndirection(1 + (i2 + i1 * ('+inddims[g_m]+')))')
          ENDDO()
          ENDDO()
        if indopts[g_m]>=0 and (accs[invinds[g_m]]==OP_INC or accs[invinds[g_m]]==OP_WRITE or accs[invinds[g_m]]==OP_RW):
          ENDIF()

    else: #direct kernel call
      code('')
      comm('kernel call')
      DO('i1','sliceStart', 'sliceEnd')
      line = 'CALL '+name+'( &'
      indent = '\n'+' '*depth
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL:
          line = line + indent +'& opDat'+str(g_m+1)
        else:
          if dims[g_m].isdigit() and int(dims[g_m]) == 1:
            line = line + indent +'& opDat'+str(g_m+1)+'(i1 * ('+dims[g_m]+'))'
          else:
            line = line + indent +'& opDat'+str(g_m+1)+'(i1 * ('+dims[g_m]+'):i1 * ('+dims[g_m]+') + '+dims[g_m]+' - 1)'
        if g_m < nargs-1:
          line = line + ', &'
        else:
           line = line + ' &'
      depth = depth - 2
      code(line + indent +  '& )')
      depth = depth + 2
      ENDDO()

    depth = depth - 2
    code('END SUBROUTINE')
    code('')

##########################################################################
#  Generate OpenMP hust stub
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
    code('INTEGER(kind=4) :: n_upper')
    code('INTEGER(kind=4), DIMENSION(1:8) :: timeArrayStart')
    code('INTEGER(kind=4), DIMENSION(1:8) :: timeArrayEnd')
    code('REAL(kind=8) :: startTime')
    code('REAL(kind=8) :: endTime')
    code('INTEGER(kind=4) :: returnSetKernelTiming')
    code('type ( op_set_core ) , POINTER :: opSetCore')
    code('')

    for g_m in range(0,ninds):
      code('type ( op_set_core ) , POINTER :: opSet'+str(invinds[g_m]+1)+'Core')
      code(typs[invinds[g_m]]+', POINTER, DIMENSION(:) :: opDat'+str(invinds[g_m]+1)+'Local')
      code('INTEGER(kind=4) :: opDat'+str(invinds[g_m]+1)+'Cardinality')
      code('')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code('type ( op_set_core ) , POINTER :: opSet'+str(g_m+1)+'Core')
        code(typs[g_m]+', POINTER, DIMENSION(:) :: opDat'+str(g_m+1)+'Local')
        code('INTEGER(kind=4) :: opDat'+str(g_m+1)+'Cardinality')
        code('')
      if maps[g_m] == OP_GBL:
        if (dims[g_m].isdigit()) and (int(dims[g_m]) == 1):
          code(typs[g_m]+', POINTER :: opDat'+str(g_m+1)+'Local')
        else:
          code(typs[g_m]+', POINTER, DIMENSION(:) :: opDat'+str(g_m+1)+'Local')
          code('INTEGER(kind=4) :: opDat'+str(g_m+1)+'Cardinality')

    code('')
    for g_m in range(0,nargs):
      code('type ( op_dat_core ) , POINTER :: opDat'+str(g_m+1)+'Core')
    code('')

    if ninds > 0:
      for g_m in range(0,nargs):
        code('type ( op_map_core ) , POINTER :: opMap'+str(g_m+1)+'Core')
      code('')

      code('INTEGER(kind=4) :: threadID')
      code('INTEGER(kind=4) :: numberOfThreads')
      code('INTEGER(kind=4) :: partitionSize')
      code('INTEGER(kind=4), DIMENSION(1:'+str(nargs)+') :: opDatArray')
      code('INTEGER(kind=4), DIMENSION(1:'+str(nargs)+') :: mappingIndicesArray')
      code('INTEGER(kind=4), DIMENSION(1:'+str(nargs)+') :: mappingArray')
      code('INTEGER(kind=4), DIMENSION(1:'+str(nargs)+') :: accessDescriptorArray')
      code('INTEGER(kind=4), DIMENSION(1:'+str(nargs)+') :: indirectionDescriptorArray')
      code('INTEGER(kind=4), DIMENSION(1:'+str(nargs)+') :: opDatTypesArray')
      code('INTEGER(kind=4) :: numberOfIndirectOpDats')
      code('INTEGER(kind=4) :: blockOffset')
      code('INTEGER(kind=4) :: nblocks')
      code('INTEGER(kind=4) :: i1')
      code('INTEGER(kind=4) :: i2')
      code('')

    else:
      code('INTEGER(kind=4) :: threadID')
      code('INTEGER(kind=4) :: numberOfThreads')
      code('INTEGER(kind=4) :: sliceStart')
      code('INTEGER(kind=4) :: sliceEnd')
      code('INTEGER(kind=4) :: partitionSize')
      code('INTEGER(kind=4) :: i1')
      code('INTEGER(kind=4) :: i10')
      code('INTEGER(kind=4) :: i11')
      code('REAL(kind=4) :: dataTransfer')

    code('')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and accs[g_m] == OP_INC:
        code(typs[g_m]+', DIMENSION(:), ALLOCATABLE :: reductionArrayHost'+str(g_m+1))

    if nopts>0:
      code('INTEGER(kind=4) :: optflags')
      code('optflags = 0')
      for i in range(0,nargs):
        if optflags[i] == 1:
          IF('opArg'+str(i+1)+'%opt == 1')
          code('optflags = IBSET(optflags,'+str(optidxs[i])+')')
          ENDIF()
    if nopts > 30:
      print 'ERROR: too many optional arguments to store flags in an integer'

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

    depth = depth - 2

    if ninds > 0:
      code_pre('#ifdef OP_PART_SIZE_1')
      code_pre('  partitionSize = OP_PART_SIZE_1')
      code_pre('#else')
      code_pre('  partitionSize = 0')
      code_pre('#endif')

    code('')
    code_pre('#ifdef _OPENMP')
    code_pre('  numberOfThreads = omp_get_max_threads()')
    code_pre('#else')
    code_pre('  numberOfThreads = 1')
    code_pre('#endif')
    depth = depth + 2


    if ninds > 0:
      for g_m in range(0,nargs):
        code('indirectionDescriptorArray('+str(g_m+1)+') = '+str(inds[g_m]-1))
      code('')

      code('numberOfIndirectOpDats = '+str(ninds))
      code('')
      code('planRet_'+name+' = FortranPlanCaller( &')
      code('& userSubroutine//C_NULL_CHAR, &')
      code('& set%setCPtr, &')
      code('& partitionSize, &')
      code('& numberOfOpDats, &')
      code('& opArgArray, &')
      code('& numberOfIndirectOpDats, &')
      code('& indirectionDescriptorArray, 2)')
      code('')
      code('CALL c_f_pointer(planRet_'+name+',actualPlan_'+name+')')
      code('CALL c_f_pointer(actualPlan_'+name+'%nindirect,pnindirect_'+name+',(/numberOfIndirectOpDats/))')
      code('CALL c_f_pointer(actualPlan_'+name+'%ind_maps,ind_maps_'+name+',(/numberOfIndirectOpDats/))')
      code('CALL c_f_pointer(actualPlan_'+name+'%maps,mappingArray_'+name+',(/numberOfOpDats/))')
      code('CALL c_f_pointer(actualPlan_'+name+'%ncolblk,ncolblk_'+name+',(/actualPlan_'+name+'%ncolors_core/))')
      code('CALL c_f_pointer(actualPlan_'+name+'%ind_sizes,ind_sizes_'+name+',(/actualPlan_'+name+'%nblocks * numberOfIndirectOpDats/))')
      code('CALL c_f_pointer(actualPlan_'+name+'%ind_offs,ind_offs_'+name+',(/actualPlan_'+name+'%nblocks * numberOfIndirectOpDats/))')
      code('CALL c_f_pointer(actualPlan_'+name+'%blkmap,blkmap_'+name+',(/actualPlan_'+name+'%nblocks/))')
      code('CALL c_f_pointer(actualPlan_'+name+'%offset,offset_'+name+',(/actualPlan_'+name+'%nblocks/))')
      code('CALL c_f_pointer(actualPlan_'+name+'%nelems,nelems_'+name+',(/actualPlan_'+name+'%nblocks/))')
      code('CALL c_f_pointer(actualPlan_'+name+'%nthrcol,nthrcol_'+name+',(/actualPlan_'+name+'%nblocks/))')
      code('CALL c_f_pointer(actualPlan_'+name+'%thrcol,thrcol_'+name+',(/set%setPtr%size/))')
      code('')
      for g_m in range(0,ninds):
        code('CALL c_f_pointer(ind_maps_'+name+'('+str(g_m+1)+'),ind_maps'+str(invinds[g_m]+1)+'_'+name+',(/pnindirect_'+name+'('+str(g_m+1)+')/))')
      code('')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP:
          IF('indirectionDescriptorArray('+str(g_m+1)+') >= 0')
          code('CALL c_f_pointer(mappingArray_'+name+'('+str(g_m+1)+'),mappingArray'+str(g_m+1)+'_'+name+',(/set%setPtr%size/))')
          ENDIF()
          code('')


    code('')
    code('opSetCore => set%setPtr')
    code('')
    for g_m in range(0,ninds):
      code('opDat'+str(invinds[g_m]+1)+'Cardinality = opArg'+str(invinds[g_m]+1)+'%dim * getSetSizeFromOpArg(opArg'+str(invinds[g_m]+1)+')')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code('opDat'+str(g_m+1)+'Cardinality = opArg'+str(g_m+1)+'%dim * getSetSizeFromOpArg(opArg'+str(g_m+1)+')')
      elif maps[g_m] == OP_GBL:
        if (dims[g_m].isdigit() == 0) or (int(dims[g_m]) > 1):
          code('opDat'+str(g_m+1)+'Cardinality = opArg'+str(g_m+1)+'%dim')

    code('')
    for g_m in range(0,ninds):
      code('CALL c_f_pointer(opArg'+str(invinds[g_m]+1)+'%data,opDat'+str(invinds[g_m]+1)+'Local,(/opDat'+str(invinds[g_m]+1)+'Cardinality/))')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code('CALL c_f_pointer(opArg'+str(g_m+1)+'%data,opDat'+str(g_m+1)+'Local,(/opDat'+str(g_m+1)+'Cardinality/))')
      elif maps[g_m] == OP_GBL:
        if dims[g_m].isdigit() and int(dims[g_m]) == 1:
          code('CALL c_f_pointer(opArg'+str(g_m+1)+'%data,opDat'+str(g_m+1)+'Local)')
        else:
          code('CALL c_f_pointer(opArg'+str(g_m+1)+'%data,opDat'+str(g_m+1)+'Local, (/opDat'+str(g_m+1)+'Cardinality/))')
    code('')

    #reductions
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and accs[g_m] == OP_INC:
        code('allocate( reductionArrayHost'+str(g_m+1)+'(numberOfThreads * (('+dims[g_m]+'-1)/64+1)*64) )')
        DO('i10','1','numberOfThreads+1')
        DO('i11','1',dims[g_m]+'+1')
        code('reductionArrayHost'+str(g_m+1)+'((i10 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + i11) = 0')
        ENDDO()
        ENDDO()

    code('')
    if ninds > 0: #indirect loop host stub call
      code('blockOffset = 0')
      code('')
      DO('i1','0','actualPlan_'+name+'%ncolors')

      IF('i1 .EQ. actualPlan_'+name+'%ncolors_core')
      code('CALL op_mpi_wait_all(numberOfOpDats,opArgArray)')
      ENDIF()
      code('')

      code('nblocks = ncolblk_'+name+'(i1 + 1)')
      code('!$OMP PARALLEL DO private (threadID)')
      DO('i2','0','nblocks')
      code('threadID = omp_get_thread_num()')
      code('CALL op_x86_'+name+'( &')
      if nopts>0:
        code('& optflags, &')
      for g_m in range(0,ninds):
        code('& opDat'+str(invinds[g_m]+1)+'Local, &')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          code('& opDat'+str(g_m+1)+'Local, &')
        elif maps[g_m] == OP_GBL and accs[g_m] == OP_INC:
          code('& reductionArrayHost'+str(g_m+1)+'(threadID * (('+dims[g_m]+'-1)/64+1)*64 + 1), &')
        elif maps[g_m] == OP_GBL and (accs[g_m] == OP_READ or accs[g_m] == OP_WRITE):
          code('& opDat'+str(g_m+1)+'Local, &')

      for g_m in range(0,ninds):
        code('& ind_maps'+str(invinds[g_m]+1)+'_'+name+', &')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP:
          code('& mappingArray'+str(g_m+1)+'_'+name+', &')

      code('& ind_sizes_'+name+', &')
      code('& ind_offs_'+name+', &')
      code('& blkmap_'+name+', &')
      code('& offset_'+name+', &')
      code('& nelems_'+name+', &')
      code('& nthrcol_'+name+', &')
      code('& thrcol_'+name+', &')
      code('& blockOffset,i2)')

      ENDDO()
      code('!$OMP END PARALLEL DO')
      code('blockOffset = blockOffset + nblocks')
      ENDDO()
      code('')


    else: #direct loop host stub call
      code('!$OMP PARALLEL DO private (sliceStart,sliceEnd,i1,threadID)')
      DO('i1','0','numberOfThreads')
      code('sliceStart = opSetCore%size * i1 / numberOfThreads')
      code('sliceEnd = opSetCore%size * (i1 + 1) / numberOfThreads')
      code('threadID = omp_get_thread_num()')
      code('CALL op_x86_'+name+'( &')
      if nopts>0:
        code('& optflags, &')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          code('& opDat'+str(g_m+1)+'Local, &')
        if maps[g_m] == OP_GBL and accs[g_m] == OP_INC:
          code('& reductionArrayHost'+str(g_m+1)+'(threadID * (('+dims[g_m]+'-1)/64+1)*64 + 1), &')
        elif maps[g_m] == OP_GBL and (accs[g_m] == OP_READ or accs[g_m] == OP_WRITE):
          code('& opDat'+str(g_m+1)+'Local, &')
      code('& sliceStart, &')
      code('& sliceEnd)')
      ENDDO()
      code('!$OMP END PARALLEL DO')

    code('')
    IF('(n_upper .EQ. 0) .OR. (n_upper .EQ. opSetCore%core_size)')
    code('CALL op_mpi_wait_all(numberOfOpDats,opArgArray)')
    ENDIF()
    code('')
    code('')
    code('CALL op_mpi_set_dirtybit(numberOfOpDats,opArgArray)')
    code('')

    #reductions
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and (accs[g_m] == OP_INC or accs[g_m] == OP_MIN or accs[g_m] == OP_MAX):
        DO('i10','1','numberOfThreads+1')
        if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
          DO('i11','1',dims[g_m]+'+1')
          code('opDat'+str(g_m+1)+'Local(i11) = opDat'+str(g_m+1)+'Local(i11) + reductionArrayHost'+str(g_m+1)+'((i10 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + i11)')
          ENDDO()
        else:
          code('opDat'+str(g_m+1)+'Local = opDat'+str(g_m+1)+'Local + reductionArrayHost'+str(g_m+1)+'((i10 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + 1)')
        ENDDO()
        code('')
        code('deallocate( reductionArrayHost'+str(g_m+1)+' )')
        code('')
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
    if ninds == 0:
      code('dataTransfer = 0.0')
      for g_m in range(0,nargs):
        if accs[g_m] == OP_READ:
          if maps[g_m] == OP_GBL:
            code('dataTransfer = dataTransfer + opArg'+str(g_m+1)+'%size')
          else:
            code('dataTransfer = dataTransfer + opArg'+str(g_m+1)+'%size * getSetSizeFromOpArg(opArg'+str(g_m+1)+')')
        else:
          if maps[g_m] == OP_GBL:
            code('dataTransfer = dataTransfer + opArg'+str(g_m+1)+'%size * 2.d0')
          else:
            code('dataTransfer = dataTransfer + opArg'+str(g_m+1)+'%size * getSetSizeFromOpArg(opArg'+str(g_m+1)+') * 2.d0')

    code('returnSetKernelTiming = setKernelTime('+str(nk)+' , userSubroutine//C_NULL_CHAR, &')

    if ninds > 0:
      code('& endTime-startTime, actualPlan_'+name+'%transfer,actualPlan_'+name+'%transfer2, 1)')
    else:
      code('& endTime-startTime, dataTransfer, 0.00000, 1)')


    depth = depth - 2
    code('END SUBROUTINE')
    code('END MODULE')
    code('')

##########################################################################
#  output individual kernel file
##########################################################################
    if hydra:
      name = 'kernels/'+kernels[nk]['master_file']+'/'+name
      fid = open(name+'_kernel.F95','w')
    else:
      fid = open(name+'_kernel.F90','w')
    date = datetime.datetime.now()
    fid.write('!\n! auto-generated by op2.py\n!\n\n')
    fid.write(file_text.strip())
    fid.close()