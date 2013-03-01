##########################################################################
#
# CUDA code generator
#
# This routine is called by op2 which parses the input files
#
# It produces a file xxx_kernel.CUF for each kernel,
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
    line = re.sub('TYPS',typs[m],line)
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
    code('DO '+i+' = '+start+', '+finish+' - 1, 1')
  elif CPP:
    code('for ( int '+i+'='+start+'; '+i+'<'+finish+'; '+i+'++ ){')
  depth += 2

def DO_STEP(i,start,finish,step):
  global file_text, FORTRAN, CPP, g_m
  global depth
  if FORTRAN:
    code('DO '+i+' = '+start+', '+finish+' - 1, '+step)
  elif CPP:
    code('for ( int '+i+'='+start+'; '+i+'<'+finish+'; '+i+' = '+i+' + '+step+' ){')
  depth += 2

def DOWHILE(line):
  global file_text, FORTRAN, CPP, g_m
  global depth
  if FORTRAN:
    code('DO WHILE ('+line+' )')
  elif CPP:
    code('while ('+ line+ ' )')
  depth += 2

def FOR(i,start,finish):
  global file_text, FORTRAN, CPP, g_m
  global depth
  if FORTRAN:
    code('DO '+i+' = '+start+', '+finish+' - 1')
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

def ENDIF():
  global file_text, FORTRAN, CPP, g_m
  global depth
  depth -= 2
  if FORTRAN:
    code('END IF')
  elif CPP:
    code('}')


def op2_gen_cuda(master, date, consts, kernels):

  global dims, idxs, typs, indtyps, inddims
  global file_format, cont, comment
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
    ninds   = kernels[nk]['ninds']
    inddims = kernels[nk]['inddims']
    indaccs = kernels[nk]['indaccs']
    indtyps = kernels[nk]['indtyps']
    invinds = kernels[nk]['invinds']

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


    FORTRAN = 1;
    CPP     = 0;
    g_m = 0;
    file_text = ''
    depth = 0

##########################################################################
#  Generate Header
##########################################################################

    code('MODULE '+name.upper()+'_MODULE')
    code('USE OP2_FORTRAN_DECLARATIONS')
    code('USE OP2_FORTRAN_RT_SUPPORT')
    code('USE ISO_C_BINDING')
    code('USE OP2_CONSTANTS')
    code('USE CUDAFOR')
    code('USE CUDACONFIGURATIONPARAMS')
    code('')

##########################################################################
#  Variable declarations
##########################################################################
    code('')
    comm('variable declarations')

    code('TYPE  :: '+name+'_opDatDimensions')
    depth = depth + 2
    for g_m in range(0,nargs):
      if maps[g_m] <> OP_GBL:
        code('INTEGER(kind=4) :: opDat'+str(g_m+1)+'Dimension')
    depth = depth - 2
    code('END TYPE '+name+'_opDatDimensions')
    code('')

    code('TYPE  :: '+name+'_opDatCardinalities')
    depth = depth + 2
    for g_m in range(0,ninds):
      code('INTEGER(kind=4) :: opDat'+str(invinds[g_m]+1)+'Cardinality')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code('INTEGER(kind=4) :: opDat'+str(g_m+1)+'Cardinality')
      elif maps[g_m] == OP_GBL:
        code('INTEGER(kind=4) :: opDat'+str(g_m+1)+'Cardinality')

    for g_m in range(0,ninds):
      code('INTEGER(kind=4) :: ind_maps'+str(invinds[g_m]+1)+'Size')

    if ninds > 0:
      for g_m in range(0,nargs):
        if maps[g_m] <> OP_GBL:
          code('INTEGER(kind=4) :: mappingArray'+str(g_m+1)+'Size')

      code('INTEGER(kind=4) :: pblkMapSize')
      code('INTEGER(kind=4) :: pindOffsSize')
      code('INTEGER(kind=4) :: pindSizesSize')
      code('INTEGER(kind=4) :: pnelemsSize')
      code('INTEGER(kind=4) :: pnthrcolSize')
      code('INTEGER(kind=4) :: poffsetSize')
      code('INTEGER(kind=4) :: pthrcolSize')

    depth = depth - 2
    code('END TYPE '+name+'_opDatCardinalities')
    code('')
    code('REAL(kind=4) :: loopTimeHost'+name)
    code('REAL(kind=4) :: loopTimeKernel'+name)
    code('INTEGER(kind=4) :: numberCalled'+name)
    code('')
    for g_m in range(0,ninds):
      code(typs[g_m]+', DIMENSION(:), DEVICE, ALLOCATABLE :: opDat'+str(invinds[g_m]+1)+'Device'+name)
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code(typs[g_m]+', DIMENSION(:), DEVICE, ALLOCATABLE :: opDat'+str(g_m+1)+'Device'+name)

    if ninds > 0:
      code('TYPE ( c_ptr )  :: planRet_'+name)
      for g_m in range(0,nargs):
        if maps[g_m] <> OP_GBL:
          code('INTEGER(kind=4), DIMENSION(:), DEVICE, ALLOCATABLE :: ind_maps'+str(g_m+1)+'_'+name)
      for g_m in range(0,nargs):
        if maps[g_m] <> OP_GBL:
          code('INTEGER(kind=2), DIMENSION(:), DEVICE, ALLOCATABLE :: mappingArray'+str(g_m+1)+'_'+name)

##########################################################################
#  Inline user kernel function
##########################################################################
    code('')
    code('CONTAINS')
    code('')
    comm('user function')
    code('attributes (device) &')
    code('#include "'+name+'.inc"')
    code('')
    code('')

##########################################################################
#  Reduction kernel function - if an OP_GBL exists
##########################################################################
    if reduct > 0:
      comm('Reduction cuda kernel'); depth = depth +2;
      code('attributes (device) SUBROUTINE ReductionFloat8(reductionResult,inputValue,reductionOperation)')
      code('REAL(kind=8), DIMENSION(:), DEVICE :: reductionResult')
      code('REAL(kind=8), DIMENSION(1:1) :: inputValue')
      code('INTEGER(kind=4), VALUE :: reductionOperation')
      code('REAL(kind=8), DIMENSION(0:*), SHARED :: sharedDouble8')
      code('INTEGER(kind=4) :: i1')
      code('INTEGER(kind=4) :: threadID')
      code('threadID = threadIdx%x - 1')
      code('i1 = ishft(blockDim%x,-1)')
      code('CALL syncthreads()')
      code('sharedDouble8(threadID) = inputValue(1)')

      DOWHILE('i1 > 0')
      code('CALL syncthreads()')
      IF('threadID < i1')
      code('SELECT CASE(reductionOperation)')
      code('CASE (0)')
      code('sharedDouble8(threadID) = sharedDouble8(threadID) + sharedDouble8(threadID + i1)')
      code('CASE (1)')
      IF('sharedDouble8(threadID + i1) < sharedDouble8(threadID)')
      code('sharedDouble8(threadID) = sharedDouble8(threadID + i1)')
      ENDIF()
      code('CASE (2)')
      IF('sharedDouble8(threadID + i1) > sharedDouble8(threadID)')
      code('sharedDouble8(threadID) = sharedDouble8(threadID + i1)')
      ENDIF()
      code('END SELECT')
      ENDIF()
      code('i1 = ishft(i1,-1)')
      ENDDO()

      code('CALL syncthreads()')

      IF('threadID .EQ. 0')
      code('SELECT CASE(reductionOperation)')
      code('CASE (0)')
      code('reductionResult(1) = reductionResult(1) + sharedDouble8(0)')
      code('CASE (1)')
      IF('sharedDouble8(0) < reductionResult(1)')
      code('reductionResult(1) = sharedDouble8(0)')
      ENDIF()
      code('CASE (2)')
      IF('sharedDouble8(0) > reductionResult(1)')
      code('reductionResult(1) = sharedDouble8(0)')
      ENDIF()
      code('END SELECT')
      ENDIF()

      code('CALL syncthreads()')
      code('END SUBROUTINE')
      code('')

##########################################################################
#  Generate CUDA kernel function
##########################################################################
    comm('CUDA kernel function')
    code('attributes (global) SUBROUTINE op_cuda_'+name+'( &'); depth = depth + 2
    code('& opDatDimensions, &')
    code('& opDatCardinalities, &')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL:
        code('& reductionArrayDevice'+str(g_m+1)+',   &')

    if ninds > 0: #indirect loop
      code('& pindSizes, &')
      code('& pindOffs, &')
      code('& pblkMap, &')
      code('& poffset, &')
      code('& pnelems, &')
      code('& pnthrcol, &')
      code('& pthrcol, &')
      code('& blockOffset)')
    else: #direct loop
      code('& setSize, &')
      code('& warpSize, &')
      code('& sharedMemoryOffset)')

    code('')
    code('IMPLICIT NONE')
    code('')

##########################################################################
#  Declare local variables
##########################################################################
    comm('local variables')
    if ninds > 0: #indirect loop
      code('TYPE ( '+name+'_opDatDimensions ) , DEVICE :: opDatDimensions')
      code('TYPE ( '+name+'_opDatCardinalities ) , DEVICE :: opDatCardinalities')
      code('INTEGER(kind=4), DIMENSION(0:opDatCardinalities%pindSizesSize - 1), DEVICE :: pindSizes')
      code('INTEGER(kind=4), DIMENSION(0:opDatCardinalities%pindOffsSize - 1), DEVICE :: pindOffs')
      code('INTEGER(kind=4), DIMENSION(0:opDatCardinalities%pblkMapSize - 1), DEVICE :: pblkMap')
      code('INTEGER(kind=4), DIMENSION(0:opDatCardinalities%poffsetSize - 1), DEVICE :: poffset')
      code('INTEGER(kind=4), DIMENSION(0:opDatCardinalities%pnelemsSize - 1), DEVICE :: pnelems')
      code('INTEGER(kind=4), DIMENSION(0:opDatCardinalities%pnthrcolSize - 1), DEVICE :: pnthrcol')
      code('INTEGER(kind=4), DIMENSION(0:opDatCardinalities%pthrcolSize - 1), DEVICE :: pthrcol')
      code('INTEGER(kind=4), VALUE :: blockOffset')
      code('')
      for g_m in range(0,ninds):
        if accs[invinds[g_m]] == OP_INC:
          for m in range (0,int(idxs[g_m])):
            code('REAL(kind=8), DIMENSION(0:'+dims[g_m]+'-1) :: opDat'+str(invinds[g_m]+1+m)+'Local')
            code('INTEGER(kind=4) :: opDat'+str(invinds[g_m]+1+m)+'Map')

      code('')
      for g_m in range(0,ninds):
        code('INTEGER(kind=4) :: opDat'+str(invinds[g_m]+1)+'nBytes')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          code('INTEGER(kind=4) :: opDat'+str(g_m+1)+'nBytes')
        elif maps[g_m] == OP_GBL:
          code('INTEGER(kind=4) :: opDat'+str(g_m+1)+'nBytes')

      code('')
      for g_m in range(0,ninds):
        code('INTEGER(kind=4) :: opDat'+str(invinds[g_m]+1)+'RoundUp')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          code('INTEGER(kind=4) :: opDat'+str(g_m+1)+'RoundUp')
        elif maps[g_m] == OP_GBL:
          code('INTEGER(kind=4) :: opDat'+str(g_m+1)+'RoundUp')

      code('')
      for g_m in range(0,ninds):
        code('INTEGER(kind=4), SHARED :: opDat'+str(invinds[g_m]+1)+'SharedIndirectionSize')
      code('')
      code('REAL(kind=8), DIMENSION(0:*), SHARED :: sharedFloat8')
      code('INTEGER(kind=4) :: sharedOffsetFloat8')
      code('INTEGER(kind=4), SHARED :: numOfColours')
      code('INTEGER(kind=4), SHARED :: numberOfActiveThreadsCeiling')
      code('INTEGER(kind=4), SHARED :: sharedMemoryOffset')
      code('INTEGER(kind=4), SHARED :: blockID')
      code('INTEGER(kind=4), SHARED :: numberOfActiveThreads')
      code('INTEGER(kind=4) :: moduloResult')
      code('INTEGER(kind=4) :: nbytes')
      code('INTEGER(kind=4) :: colour1')
      code('INTEGER(kind=4) :: colour2')
      code('INTEGER(kind=4) :: n1')
      code('INTEGER(kind=4) :: i1')
      code('INTEGER(kind=4) :: i2')

      IF('threadIdx%x - 1 .EQ. 0')
      code('blockID = pblkMap(blockIdx%x - 1 + blockOffset)')
      code('numberOfActiveThreads = pnelems(blockID)')
      code('numberOfActiveThreadsCeiling = blockDim%x * (1 + (numberOfActiveThreads - 1) / blockDim%x)')
      code('numOfColours = pnthrcol(blockID)')
      code('sharedMemoryOffset = poffset(blockID)')
      code('')
      for g_m in range(0,ninds):
        code('opDat'+str(invinds[g_m]+1)+'SharedIndirectionSize = pindSizes('+str(g_m)+' + blockID * '+str(ninds)+')')
      ENDIF()
      code('')
      code('CALL syncthreads()')
      code('')
      for g_m in range(0,ninds):
        code('opDat'+str(invinds[g_m]+1)+'RoundUp = opDat'+str(invinds[g_m]+1)+'SharedIndirectionSize * opDatDimensions%opDat'+str(invinds[g_m]+1)+'Dimension')
      code('')
      for g_m in range(0,ninds):
        if g_m == 0:
          code('opDat'+str(invinds[g_m]+1)+'nBytes = 0')
        else:
          code('opDat'+str(invinds[g_m]+1)+'nBytes = opDat'+str(invinds[g_m-1]+1)+'nBytes * 8 / 8 + opDat'+str(invinds[g_m-1]+1)+'RoundUp * 8 / 8')
      code('')

      for g_m in range(0,ninds):
        code('i1 = threadIdx%x - 1')
        code('n1 = opDat'+str(invinds[g_m]+1)+'SharedIndirectionSize * opDatDimensions%opDat'+str(invinds[g_m]+1)+'Dimension')
        if accs[invinds[g_m]] == OP_READ:
          DOWHILE('i1 < n1')
          code('moduloResult = mod(i1,opDatDimensions%opDat'+str(invinds[g_m]+1)+'Dimension)')
          code('sharedFloat8(opDat'+str(invinds[g_m]+1)+'nBytes + i1) = opDat'+str(invinds[g_m]+1)+'Device'+name+'( &')
          code('& moduloResult + ind_maps'+str(invinds[g_m]+1)+'_'+name+'(0 + (pindOffs('+str(g_m)+' + blockID * '+str(ninds)+') + i1 / &')
          code('& opDatDimensions%opDat'+str(invinds[g_m]+1)+'Dimension) + 1) * &')
          code('& opDatDimensions%opDat'+str(invinds[g_m]+1)+'Dimension + 1)')
          code('i1 = i1 + blockDim%x')
          ENDDO()
        elif accs[invinds[g_m]] == OP_INC:
          DOWHILE('i1 < n1')
          code('sharedFloat8(opDat'+str(invinds[g_m]+1)+'nBytes + i1) = 0')
          code('i1 = i1 + blockDim%x')
          ENDDO()
        code('')

      code('CALL syncthreads()')
      code('i1 = threadIdx%x - 1')
      code('')


      DOWHILE('i1 < numberOfActiveThreadsCeiling')
      code('colour2 = -1')
      IF('i1 < numberOfActiveThreads')
      for g_m in range(0,ninds):
        if accs[invinds[g_m]] == OP_INC:
          for m in range (0,int(idxs[g_m])):
            DO('i2','0','opDatDimensions%opDat'+str(invinds[g_m]+1+m)+'Dimension')
            code('opDat'+str(invinds[g_m]+1+m)+'Local(i2) = 0')
            ENDDO()

    else: #direct loop
      code('TYPE ( '+name+'_opDatDimensions ) , DEVICE :: opDatDimensions')
      code('TYPE ( '+name+'_opDatCardinalities ) , DEVICE :: opDatCardinalities')
      for g_m in range(0,nargs):
        if maps[g_m] <> OP_GBL:
          code(typs[g_m]+', DIMENSION(0:'+dims[g_m]+'-1) :: opDat'+str(g_m+1)+'Local')
        else: #global arg
          code(typs[g_m]+', DIMENSION(0:'+dims[g_m]+'-1) :: opDat'+str(g_m+1)+'Local')
          code(typs[g_m]+', DIMENSION(:), DEVICE :: reductionArrayDevice'+str(g_m+1))

      code('INTEGER(kind=4), VALUE :: setSize')
      code('INTEGER(kind=4), VALUE :: warpSize')
      code('INTEGER(kind=4), VALUE :: sharedMemoryOffset')
      code('REAL(kind=8), DIMENSION(0:*), SHARED :: sharedFloat8')
      code('INTEGER(kind=4) :: sharedOffsetFloat8')
      code('INTEGER(kind=4) :: numberOfActiveThreads')
      code('INTEGER(kind=4) :: localOffset')
      code('INTEGER(kind=4) :: threadID')
      code('INTEGER(kind=4) :: i1')
      code('INTEGER(kind=4) :: i2')


##########################################################################
#  CUDA kernel call
##########################################################################
    if ninds > 0: #indirect kernel call
      code('')
      comm('kernel call')
      line = '  CALL '+name+'( &'
      indent = '\n'+' '*depth
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          if int(dims[g_m]) > 1:
            line = line + indent + '& opDat'+str(g_m+1)+'Device'+name+ \
            '((i1 + sharedMemoryOffset) * opDatDimensions%opDat'+str(g_m+1)+ \
            'Dimension + 1:(i1 + sharedMemoryOffset) * opDatDimensions%opDat'+ \
            str(g_m+1)+'Dimension + opDatDimensions%opDat'+str(g_m+1)+\
            'Dimension + 1 + 1)'
          else:
            line = line + indent + '& opDat'+str(g_m+1)+'Device'+ \
            name+'((i1 + sharedMemoryOffset) * opDatDimensions%opDat'+str(g_m+1)+ \
            'Dimension + 1)'
        if maps[g_m] == OP_MAP and accs[g_m] == OP_READ:
          line = line + indent + '& sharedFloat8(opDat'+str(invinds[inds[g_m]-1]+1)+ \
          'nBytes + mappingArray'+str(g_m+1)+'_'+name+ \
          '(i1 + sharedMemoryOffset + 1) * opDatDimensions%opDat'+str(g_m+1)+'Dimension)'
        elif maps[g_m] == OP_MAP and (accs[g_m] == OP_INC or accs[g_m] == OP_RW):
          line = line +indent + '& opDat'+str(g_m+1)+'Local'
        if g_m < nargs-1:
          line = line +', &'
        else:
           line = line +' &'
      depth = depth - 2
      code(line + indent + '& )')
      depth = depth + 2
      code('colour2 = pthrcol(i1 + sharedMemoryOffset)')
      ENDIF()

      code('')
      for g_m in range(0,ninds):
        if accs[invinds[g_m]] == OP_INC:
          for m in range (0,int(idxs[g_m])):
            code('opDat'+str(invinds[g_m]+1+m)+'Map = mappingArray'+str(invinds[g_m]+1+m)+'_'+name+'(i1 + sharedMemoryOffset + 1)')
      code('')

      DO('colour1','0','numOfColours')
      IF('colour2 .EQ. colour1')
      for g_m in range(0,ninds):
        if accs[invinds[g_m]] == OP_INC:
          for m in range (0,int(idxs[g_m])):
            DO('i2','0', 'opDatDimensions%opDat'+str(invinds[g_m]+1+m)+'Dimension')
            code('sharedFloat8(opDat'+str(invinds[g_m]+1)+'nBytes + (i2 + opDat'+str(invinds[g_m]+1+m)+'Map * opDatDimensions%opDat'+str(invinds[g_m]+1+m)+'Dimension)) = &')
            code('& sharedFloat8(opDat'+str(invinds[g_m]+1)+'nBytes + (i2 + opDat'+str(invinds[g_m]+1+m)+'Map * opDatDimensions%opDat'+str(invinds[g_m]+1+m)+'Dimension)) + opDat'+str(invinds[g_m]+1+m)+'Local(i2)')
            ENDDO()
            code('')
      ENDIF()
      code('CALL syncthreads()')
      ENDDO()
      code('i1 = i1 + blockDim%x')
      ENDDO()
      code('')
      code('CALL syncthreads()')
      code('i1 = threadIdx%x - 1')
      code('')
      for g_m in range(0,ninds):
        if accs[invinds[g_m]] == OP_INC:
          DOWHILE('i1 < opDat'+str(invinds[g_m]+1)+'SharedIndirectionSize * opDatDimensions%opDat'+str(invinds[g_m]+1)+'Dimension')
          code('moduloResult = mod(i1,opDatDimensions%opDat'+str(invinds[g_m]+1)+'Dimension)')
          code('opDat'+str(invinds[g_m]+1)+'Device'+name+'(moduloResult + ind_maps'+str(invinds[g_m]+1)+'_'+name+' &')
          code('& (0 + (pindOffs(3 + blockID * 4) + i1 / opDatDimensions%opDat'+str(invinds[g_m]+1)+'Dimension) + 1) * &')
          code('& opDatDimensions%opDat'+str(invinds[g_m]+1)+'Dimension + 1) = &')
          code('& opDat'+str(invinds[g_m]+1)+'Device'+name+'(moduloResult + ind_maps'+str(invinds[g_m]+1)+'_'+name+' &')
          code('& (0 + (pindOffs(3 + blockID * 4) + i1 / opDatDimensions%opDat'+str(invinds[g_m]+1)+'Dimension) + 1) * &')
          code('& opDatDimensions%opDat'+str(invinds[g_m]+1)+'Dimension + 1) + &')
          code('& sharedFloat8(opDat'+str(invinds[g_m]+1)+'nBytes + i1)')
          code('i1 = i1 + blockDim%x')
          ENDDO()

    else: #direct kernel call
      code('')
      comm('kernel call')
      code('threadID = mod(threadIdx%x - 1,warpSize)')
      code('sharedOffsetFloat8 = sharedMemoryOffset * ((threadIdx%x - 1) / warpSize) / 8')
      code('')
      DO_STEP('i1','threadIdx%x - 1 + (blockIdx%x - 1) * blockDim%x','setSize','blockDim%x * gridDim%x')
      code('localOffset = i1 - threadID')
      code('numberOfActiveThreads = min(warpSize,setSize - localOffset)')
      for g_m in range(0,nargs):
        if int(dims[g_m]) <> 1 and (accs[g_m] == OP_READ or accs[g_m] == OP_RW):
          DO('i2','0','opDatDimensions%opDat'+str(g_m+1)+'Dimension')
          code('sharedFloat8(sharedOffsetFloat8 + (threadID + i2 * numberOfActiveThreads)) = &')
          code('& opDat'+str(g_m+1)+'Device'+name+'(threadID + (i2 * numberOfActiveThreads + localOffset &')
          code('& * opDatDimensions%opDat'+str(g_m+1)+'Dimension) + 1)')
          ENDDO()
          code('')
          DO('i2','0','opDatDimensions%opDat'+str(g_m+1)+'Dimension')
          code('opDat'+str(g_m+1)+'Local(i2) = sharedFloat8(sharedOffsetFloat8 + (i2 + threadID * opDatDimensions%opDat'+str(g_m+1)+'Dimension))')
          ENDDO()
          code('')
      code('')
      line = '  CALL '+name+'( &'
      indent = '\n'+' '*depth
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL:
          line = line + indent +'& opDat'+str(g_m+1)+'Local'
        else:
          if int(dims[g_m]) == 1:
            line = line + indent +'& opDat'+str(g_m+1)+'Device'+name+'(i1 + 1)'
          else:
            line = line + indent +'& opDat'+str(g_m+1)+'Local'

        if g_m < nargs-1:
          line = line + ', &'
        else:
          line = line + ' &'
      depth = depth - 2
      code(line + indent +  '& )')
      depth = depth + 2
      code('')
      for g_m in range(0,nargs):
        if int(dims[g_m]) <> 1 and (accs[g_m] == OP_WRITE or accs[g_m] == OP_RW):
          DO('i2','0','opDatDimensions%opDat'+str(g_m+1)+'Dimension')
          code('sharedFloat8(sharedOffsetFloat8 + (i2 + threadID * opDatDimensions%opDat'+str(g_m+1)+'Dimension)) = opDat'+str(g_m+1)+'Local(i2)')
          ENDDO()
          code('')
          DO('i2','0','opDatDimensions%opDat'+str(g_m+1)+'Dimension')
          code('opDat'+str(g_m+1)+'Device'+name+'(threadID + (i2 * numberOfActiveThreads + localOffset * &')
          code('& opDatDimensions%opDat'+str(g_m+1)+'Dimension) + 1) = &')
          code('& sharedFloat8(sharedOffsetFloat8 + (threadID + i2 * numberOfActiveThreads))')
          ENDDO()
          code('')

      ENDDO()

    #call cuda reduction for each OP_GBL
    code('')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL:
        code('CALL ReductionFloat8(reductionArrayDevice'+str(g_m+1)+'(blockIdx%x - 1 + 1:),opDat'+str(g_m+1)+'Local,0)')
    code('')

    depth = depth - 2
    code('END SUBROUTINE')
    code('')

##########################################################################
#  Generate CUP hust stub
##########################################################################
    code('attributes (host) SUBROUTINE '+name+'_host( userSubroutine, set, &'); depth = depth + 2
    for g_m in range(0,nargs):
      if g_m == nargs-1:
        code('& opArg'+str(g_m+1)+' )')
      else:
        code('& opArg'+str(g_m+1)+', &')

    code('')
    code('IMPLICIT NONE')
    code('character(len='+str(len(name))+'), INTENT(IN) :: userSubroutine')
    code('TYPE ( op_set ) , INTENT(IN) :: set')
    code('')

    for g_m in range(0,nargs):
      code('TYPE ( op_arg ) , INTENT(IN) :: opArg'+str(g_m+1))
    code('')
    code('TYPE ( op_arg ) , DIMENSION('+str(nargs)+') :: opArgArray')
    code('INTEGER(kind=4) :: numberOfOpDats')
    code('INTEGER(kind=4) :: returnMPIHaloExchange')
    code('INTEGER(kind=4) :: returnSetKernelTiming')
    code('')
    code('TYPE ( '+name+'_opDatDimensions ) , DEVICE :: opDatDimensions')
    code('TYPE ( '+name+'_opDatCardinalities ) , DEVICE :: opDatCardinalities')
    code('')
    for g_m in range(0,ninds):
      code('INTEGER(kind=4) :: opDat'+str(invinds[g_m]+1)+'Cardinality')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code('INTEGER(kind=4) :: opDat'+str(g_m+1)+'Cardinality')
      elif maps[g_m] == OP_GBL:
        code('INTEGER(kind=4) :: opDat'+str(g_m+1)+'Cardinality')
    code('')

    if ninds > 0: #indirect loop
      code('TYPE ( op_plan ) , POINTER :: actualPlan_'+name+'')
      code('TYPE ( c_devptr ) , POINTER, DIMENSION(:) :: pindMaps')
      code('TYPE ( c_devptr ) , POINTER, DIMENSION(:) :: pmaps')
      code('')
      code('INTEGER(kind=4) :: pindMapsSize')
      code('INTEGER(kind=4) :: blocksPerGrid')
      code('INTEGER(kind=4) :: threadsPerBlock')
      code('INTEGER(kind=4) :: dynamicSharedMemorySize')
      code('INTEGER(kind=4) :: threadSynchRet')
      code('INTEGER(kind=4), DIMENSION(1:'+str(nargs)+') :: opDatArray')
      code('INTEGER(kind=4), DIMENSION(1:'+str(nargs)+') :: mappingIndicesArray')
      code('INTEGER(kind=4), DIMENSION(1:'+str(nargs)+') :: mappingArray')
      code('INTEGER(kind=4), DIMENSION(1:'+str(nargs)+') :: accessDescriptorArray')
      code('INTEGER(kind=4), DIMENSION(1:'+str(nargs)+') :: indirectionDescriptorArray')
      code('')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP:
          code('INTEGER(kind=4) :: mappingArray'+str(g_m+1)+'Size')
      code('')

      code('INTEGER(kind=4) :: numberOfIndirectOpDats')
      code('INTEGER(kind=4) :: blockOffset')
      code('INTEGER(kind=4) :: pindSizesSize')
      code('INTEGER(kind=4) :: pindOffsSize')
      code('INTEGER(kind=4) :: pblkMapSize')
      code('INTEGER(kind=4) :: poffsetSize')
      code('INTEGER(kind=4) :: pnelemsSize')
      code('INTEGER(kind=4) :: pnthrcolSize')
      code('INTEGER(kind=4) :: pthrcolSize')
      code('INTEGER(kind=4), POINTER, DIMENSION(:) :: ncolblk')
      code('INTEGER(kind=4), POINTER, DIMENSION(:) :: pnindirect')
      code('INTEGER(kind=4), DIMENSION(:), DEVICE, ALLOCATABLE :: pindSizes')
      code('INTEGER(kind=4), DIMENSION(:), DEVICE, ALLOCATABLE :: pindOffs')
      code('INTEGER(kind=4), DIMENSION(:), DEVICE, ALLOCATABLE :: pblkMap')
      code('INTEGER(kind=4), DIMENSION(:), DEVICE, ALLOCATABLE :: poffset')
      code('INTEGER(kind=4), DIMENSION(:), DEVICE, ALLOCATABLE :: pnelems')
      code('INTEGER(kind=4), DIMENSION(:), DEVICE, ALLOCATABLE :: pnthrcol')
      code('INTEGER(kind=4), DIMENSION(:), DEVICE, ALLOCATABLE :: pthrcol')
      code('INTEGER(kind=4) :: partitionSize')
      code('INTEGER(kind=4) :: blockSize')
      code('INTEGER(kind=4) :: i1')
      code('INTEGER(kind=4) :: i2')
      code('INTEGER(kind=4), SAVE :: calledTimes')
      code('INTEGER(kind=4) :: returnDumpOpDat')
      code('')

    else: #direct loop
      code('INTEGER(kind=4) :: blocksPerGrid')
      code('INTEGER(kind=4) :: threadsPerBlock')
      code('INTEGER(kind=4) :: dynamicSharedMemorySize')
      code('INTEGER(kind=4) :: threadSynchRet')
      code('INTEGER(kind=4) :: sharedMemoryOffset')
      code('INTEGER(kind=4) :: warpSize')
      code('INTEGER(kind=4), SAVE :: calledTimes')
      code('INTEGER(kind=4) :: returnDumpOpDat')
      code('INTEGER(kind=4) :: i1')
      code('INTEGER(kind=4) :: i2')
      code('INTEGER(kind=4) :: i10')
      code('INTEGER(kind=4) :: i20')
      code('')

    code('INTEGER(kind=4) :: istat')
    code('REAL(kind=4) :: accumulatorHostTime')
    code('REAL(kind=4) :: accumulatorKernelTime')
    code('REAL(kind=8) :: KT_double')
    code('TYPE ( cudaEvent )  :: startTimeHost')
    code('TYPE ( cudaEvent )  :: endTimeHost')
    code('TYPE ( cudaEvent )  :: startTimeKernel')
    code('TYPE ( cudaEvent )  :: endTimeKernel')

    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL:
        code(typs[g_m]+', DIMENSION(:), ALLOCATABLE :: reductionArrayHost'+str(g_m+1))
        code(typs[g_m]+', DIMENSION(:), DEVICE, ALLOCATABLE :: reductionArrayDevice'+str(g_m+1))
        code(typs[g_m]+', POINTER :: opDat'+str(g_m+1)+'Host')
        code('INTEGER(kind=4) :: reductionCardinality'+str(g_m+1))

    code('')
    code('numberOfOpDats = '+str(nargs))
    code('')

    for g_m in range(0,nargs):
      code('opArgArray('+str(g_m+1)+') = opArg'+str(g_m+1))
    code('')

    code('returnMPIHaloExchange = op_mpi_halo_exchanges(set%setCPtr,numberOfOpDats,opArgArray)')
    IF('returnMPIHaloExchange .EQ. 0')
    code('CALL op_mpi_wait_all(numberOfOpDats,opArgArray)')
    code('CALL op_mpi_set_dirtybit(numberOfOpDats,opArgArray)')
    code('RETURN')
    ENDIF()
    code('')

    code('istat = cudaEventCreate(startTimeHost)')
    code('istat = cudaEventCreate(endTimeHost)')
    code('istat = cudaEventCreate(startTimeKernel)')
    code('istat = cudaEventCreate(endTimeKernel)')
    code('')
    code('numberCalled'+name+' = numberCalled'+name+' + 1')
    code('istat = cudaEventRecord(startTimeHost,0)')
    code('')

    if ninds > 0:
      for g_m in range(0,nargs):
        code('indirectionDescriptorArray('+str(g_m+1)+') = '+str(inds[g_m]-1))
      code('')
      code('numberOfIndirectOpDats = '+str(ninds))
      code('')
      code('partitionSize = getPartitionSize(userSubroutine,set%setPtr%size)')
      code('')
      code('planRet_'+name+' = FortranPlanCaller( &')
      code('& userSubroutine, &')
      code('& set%setCPtr, &')
      code('& partitionSize, &')
      code('& numberOfOpDats, &')
      code('& opArgArray, &')
      code('& numberOfIndirectOpDats, &')
      code('& indirectionDescriptorArray)')
      code('')
    else:
      code('')
      code('blocksPerGrid = 200')
      code('threadsPerBlock = getBlockSize(userSubroutine,set%setPtr%size)')
      code('warpSize = OP_WARPSIZE')
      code('dynamicSharedMemorySize = 32')
      code('sharedMemoryOffset = dynamicSharedMemorySize * OP_WARPSIZE')
      code('dynamicSharedMemorySize = dynamicSharedMemorySize * threadsPerBlock')
      code('')


    for g_m in range(0,ninds):
      code('opDatCardinalities%opDat'+str(invinds[g_m]+1)+'Cardinality = opArg'+str(invinds[g_m]+1)+'%dim * getSetSizeFromOpArg(opArg'+str(invinds[g_m]+1)+')')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code('opDatCardinalities%opDat'+str(g_m+1)+'Cardinality = opArg'+str(g_m+1)+'%dim * getSetSizeFromOpArg(opArg'+str(g_m+1)+')')
      elif maps[g_m] == OP_GBL:
        code('opDatCardinalities%opDat'+str(g_m+1)+'Cardinality = set%setPtr%size')
    code('')

    for g_m in range(0,nargs):
      if maps[g_m] <> OP_GBL:
        code('opDatDimensions%opDat'+str(g_m+1)+'Dimension = opArg'+str(g_m+1)+'%dim')

    code('')
    for g_m in range(0,ninds):
      code('opDat'+str(invinds[g_m]+1)+'Cardinality = opArg'+str(invinds[g_m]+1)+'%dim * getSetSizeFromOpArg(opArg'+str(invinds[g_m]+1)+')')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code('opDat'+str(g_m+1)+'Cardinality = opArg'+str(g_m+1)+'%dim * getSetSizeFromOpArg(opArg'+str(g_m+1)+')')
      elif maps[g_m] == OP_GBL:
        code('opDat'+str(g_m+1)+'Cardinality = set%setPtr%size')

    code('')
    for g_m in range(0,ninds):
      code('CALL c_f_pointer(opArg'+str(invinds[g_m]+1)+'%data_d,opDat'+str(invinds[g_m]+1)+'Device'+name+',(/opDat'+str(invinds[g_m]+1)+'Cardinality/))')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code('CALL c_f_pointer(opArg'+str(g_m+1)+'%data_d,opDat'+str(g_m+1)+'Device'+name+',(/opDat'+str(g_m+1)+'Cardinality/))')
      elif maps[g_m] == OP_GBL:
        code('CALL c_f_pointer(opArg'+str(g_m+1)+'%data,opDat'+str(g_m+1)+'Host)')
    code('')

    if ninds > 0:
      code('CALL c_f_pointer(planRet_'+name+',actualPlan_'+name+')')
      code('CALL c_f_pointer(actualPlan_'+name+'%ind_maps,pindMaps,(/numberOfIndirectOpDats/))')
      code('CALL c_f_pointer(actualPlan_'+name+'%ncolblk,ncolblk,(/set%setPtr%size/))')
      code('')
      code('pindSizesSize = actualPlan_'+name+'%nblocks * numberOfIndirectOpDats')
      code('CALL c_f_pointer(actualPlan_'+name+'%ind_sizes,pindSizes,(/pindSizesSize/))')
      code('')
      code('pindOffsSize = pindSizesSize')
      code('CALL c_f_pointer(actualPlan_'+name+'%ind_offs,pindOffs,(/pindOffsSize/))')
      code('')
      code('pblkMapSize = actualPlan_'+name+'%nblocks')
      code('CALL c_f_pointer(actualPlan_'+name+'%blkmap,pblkMap,(/pblkMapSize/))')
      code('')
      code('poffsetSize = actualPlan_'+name+'%nblocks')
      code('CALL c_f_pointer(actualPlan_'+name+'%offset,poffset,(/poffsetSize/))')
      code('')
      code('pnelemsSize = actualPlan_'+name+'%nblocks')
      code('CALL c_f_pointer(actualPlan_'+name+'%nelems,pnelems,(/pnelemsSize/))')
      code('')
      code('pnthrcolSize = actualPlan_'+name+'%nblocks')
      code('CALL c_f_pointer(actualPlan_'+name+'%nthrcol,pnthrcol,(/pnthrcolSize/))')
      code('')
      code('pthrcolSize = set%setPtr%size')
      code('CALL c_f_pointer(actualPlan_'+name+'%thrcol,pthrcol,(/pthrcolSize/))')
      code('CALL c_f_pointer(actualPlan_'+name+'%nindirect,pnindirect,(/numberOfIndirectOpDats/))')
      code('')
      for g_m in range(0,ninds):
        code('CALL c_f_pointer(pindMaps('+str(g_m+1)+'),ind_maps'+str(invinds[g_m]+1)+'_'+name+',pnindirect('+str(g_m+1)+'))')
      code('CALL c_f_pointer(actualPlan_'+name+'%maps,pmaps,(/numberOfOpDats/))')
      code('')

      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP:
          IF('indirectionDescriptorArray('+str(g_m+1)+') >= 0')
          code('mappingArray'+str(g_m+1)+'Size = set%setPtr%size')
          code('CALL c_f_pointer(pmaps('+str(g_m+1)+'),mappingArray'+str(g_m+1)+'_'+name+',(/mappingArray'+str(g_m+1)+'Size/))')
          ENDIF()
          code('')

      for g_m in range(0,ninds):
        code('opDatCardinalities%ind_maps'+str(invinds[g_m]+1)+'Size = pnindirect('+str(g_m+1)+')')
      code('')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP:
          code('opDatCardinalities%mappingArray'+str(g_m+1)+'Size = mappingArray'+str(g_m+1)+'Size')
      code('')

      code('opDatCardinalities%pblkMapSize = pblkMapSize')
      code('opDatCardinalities%pindOffsSize = pindOffsSize')
      code('opDatCardinalities%pindSizesSize = pindSizesSize')
      code('opDatCardinalities%pnelemsSize = pnelemsSize')
      code('opDatCardinalities%pnthrcolSize = pnthrcolSize')
      code('opDatCardinalities%poffsetSize = poffsetSize')
      code('opDatCardinalities%pthrcolSize = pthrcolSize')
      code('')


    #setup for reduction
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL:
        code('reductionCardinality'+str(g_m+1)+' = blocksPerGrid * 1')
        code('allocate( reductionArrayHost'+str(g_m+1)+'(reductionCardinality'+str(g_m+1)+') )')
        code('allocate( reductionArrayDevice'+str(g_m+1)+'(reductionCardinality'+str(g_m+1)+') )')
        code('')
        DO('i10','1','reductionCardinality'+str(g_m+1)+'')
        code('reductionArrayHost'+str(g_m+1)+'(i10) = 0.00000')
        ENDDO()
        code('')
        code('reductionArrayDevice'+str(g_m+1)+' = reductionArrayHost'+str(g_m+1)+'')

    code('istat = cudaEventRecord(endTimeHost,0)')
    code('istat = cudaEventSynchronize(endTimeHost)')
    code('istat = cudaEventElapsedTime(accumulatorHostTime,startTimeHost,endTimeHost)')
    code('')
    code('loopTimeHost'+name+' = loopTimeHost'+name+' + accumulatorHostTime')
    code('istat = cudaEventRecord(startTimeKernel,0)')
    code('')

    #indirect loop host stub call
    if ninds > 0:
      code('blockOffset = 0')
      code('')
      code('threadsPerBlock = getBlockSize(userSubroutine,set%setPtr%size)')

      DO('i2','0','actualPlan_'+name+'%ncolors')
      code('blocksPerGrid = ncolblk(i2 + 1)')
      code('dynamicSharedMemorySize = actualPlan_'+name+'%nshared')
      code('')
      code('CALL op_cuda_'+name+' <<<blocksPerGrid,threadsPerBlock,dynamicSharedMemorySize>>> &')
      code('& (opDatDimensions,opDatCardinalities,pindSizes,pindOffs,pblkMap, &')
      code('& poffset,pnelems,pnthrcol,pthrcol,blockOffset)')
      code('')
      code('threadSynchRet = cudaThreadSynchronize()')
      code('blockOffset = blockOffset + blocksPerGrid')
      ENDDO()
      code('')
      code('istat = cudaEventRecord(endTimeKernel,0)')
      code('istat = cudaEventSynchronize(endTimeKernel)')
      code('istat = cudaEventElapsedTime(accumulatorKernelTime,startTimeKernel,endTimeKernel)')
      code('loopTimeKernel'+name+' = loopTimeKernel'+name+' + accumulatorKernelTime')
      code('')
      code('istat = cudaEventRecord(startTimeHost,0)')
      code('istat = cudaEventRecord(endTimeHost,0)')
      code('istat = cudaEventSynchronize(endTimeHost)')
      code('istat = cudaEventElapsedTime(accumulatorHostTime,startTimeHost,endTimeHost)')
      code('loopTimeHost'+name+' = loopTimeHost'+name+' + accumulatorHostTime')
      code('')
    else: #direct loop host stub call
      code('CALL op_cuda_'+name+' <<<blocksPerGrid,threadsPerBlock,dynamicSharedMemorySize>>> &')
      code('& (opDatDimensions,opDatCardinalities, &')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL:
          code('reductionArrayDevice'+str(g_m+1)+', &')
      code('set%setPtr%size, &')
      code('& warpSize,sharedMemoryOffset)')
      code('')
      code('threadSynchRet = cudaThreadSynchronize()')
      code('istat = cudaEventRecord(endTimeKernel,0)')
      code('istat = cudaEventSynchronize(endTimeKernel)')
      code('istat = cudaEventElapsedTime(accumulatorKernelTime,startTimeKernel,endTimeKernel)')
      code('loopTimeKernel'+name+' = loopTimeKernel'+name+' + accumulatorKernelTime')
      code('')

    #reduction
    #reductions
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and accs[g_m] == OP_INC:
        code('istat = cudaEventRecord(startTimeHost,0)') #timer for reduction
        code('reductionArrayHost'+str(g_m+1)+' = reductionArrayDevice'+str(g_m+1)+'')
        code('')
        DO('i10','0','reductionCardinality'+str(g_m+1)+'')
        code('opDat'+str(g_m+1)+'Host = reductionArrayHost'+str(g_m+1)+'(i10) + opDat'+str(g_m+1)+'Host')
        ENDDO()
        code('')
        code('deallocate( reductionArrayHost'+str(g_m+1)+' )')
        code('deallocate( reductionArrayDevice'+str(g_m+1)+' )')
        code('CALL op_mpi_reduce_double(opArg'+str(g_m+1)+',opArg'+str(g_m+1)+'%data)')
        code('')
        code('calledTimes = calledTimes + 1')
        code('istat = cudaEventRecord(endTimeHost,0)') #end timer for reduction
        code('istat = cudaEventSynchronize(endTimeHost)')
        code('istat = cudaEventElapsedTime(accumulatorHostTime,startTimeHost,endTimeHost)')
        code('loopTimeHost'+name+' = loopTimeHost'+name+' + accumulatorHostTime')

    code('KT_double = REAL(accumulatorKernelTime / 1000.00)')
    code('returnSetKernelTiming = setKernelTime('+str(nk)+' , userSubroutine//C_NULL_CHAR, &')

    if ninds > 0:
      code('& KT_double, actualPlan_'+name+'%transfer,actualPlan_'+name+'%transfer2)')
    else:
      code('& KT_double, 0.00000,0.00000)')

    depth = depth - 2
    code('END SUBROUTINE')
    code('END MODULE '+name.upper()+'_MODULE')
##########################################################################
#  output individual kernel file
##########################################################################
    fid = open(name+'_kernel.CUF','w')
    date = datetime.datetime.now()
    fid.write('!\n! auto-generated by op2.py on '+date.strftime("%Y-%m-%d %H:%M")+'\n!\n\n')
    fid.write(file_text)
    fid.close()
