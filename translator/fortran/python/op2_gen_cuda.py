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
import os

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

def op2_gen_cuda(master, date, consts, kernels, hydra):

  global dims, idxs, typs, indtyps, inddims
  global file_format, cont, comment
  global FORTRAN, CPP, g_m, file_text, depth, header_text, body_text

  header_text = ''
  body_text = ''

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
    j = 0
    ind_rw = 0
    for i in range(0,nargs):
      if maps[i] == OP_MAP and accs[i] == OP_INC:
        j = i
      if maps[i] == OP_MAP and accs[i] == OP_RW:
        ind_rw = 1
    ind_inc = j > 0

    j = 0
    reduct_mdim = 0
    reduct_1dim = 0
    for i in range(0,nargs):
      if maps[i] == OP_GBL and (accs[i] == OP_INC or accs[i] == OP_MAX or accs[i] == OP_MIN):
        j = i
        if (not dims[i].isdigit()) or int(dims[i])>1:
          reduct_mdim = 1
          if (accs[i] == OP_MAX or accs[i] == OP_MIN):
            print 'ERROR: Multidimensional MIN/MAX reduction not yet implemented'
        else:
          reduct_1dim = 1
      if maps[i] == OP_GBL and accs[i] == OP_WRITE:
        j = i
    reduct = j > 0

    is_soa = -1
    for i in range(0,nargs):
      if soaflags[i] == 1:
        is_soa = i
        break

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
      modfile = kernels[nk]['mod_file'][4:]
      filename = modfile.split('_')[1].lower() + '/' + modfile.split('_')[0].lower() + '/' + name + '.F95'
      if not os.path.isfile(filename):
        filename = modfile.split('_')[1].lower() + '/' + modfile.split('_')[0].lower() + '/' + name[:-1] + '.F95'
      fid = open(filename, 'r')
      text = fid.read()
      fid.close()
      code('USE HYDRA_CUDA_MODULE')
    else:
      code('MODULE '+name.upper()+'_MODULE')
      code('USE OP2_CONSTANTS')
    code('USE OP2_FORTRAN_DECLARATIONS')
    code('USE OP2_FORTRAN_RT_SUPPORT')
    code('USE ISO_C_BINDING')
    code('USE CUDAFOR')
    code('USE CUDACONFIGURATIONPARAMS')
    code('')
    code('')
    code('#ifdef _OPENMP'); depth = depth + 2
    code('USE OMP_LIB'); depth = depth - 2
    code('#endif')

##########################################################################
#  Variable declarations
##########################################################################
    code('')
    comm(name+'variable declarations')
    code('')

    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL:
        if (accs[g_m]== OP_INC or accs[g_m]== OP_MIN or accs[g_m]== OP_MAX):
          code(typs[g_m]+', DIMENSION(:), DEVICE, ALLOCATABLE :: reductionArrayDevice'+str(g_m+1)+name)
        if ((accs[g_m]==OP_READ and ((not dims[g_m].isdigit()) or int(dims[g_m]) > 1)) or accs[g_m]==OP_WRITE):
          code(typs[g_m]+', DIMENSION(:), DEVICE, ALLOCATABLE :: opGblDat'+str(g_m+1)+'Device'+name)

    code('REAL(kind=4) :: loopTimeHost'+name)
    code('REAL(kind=4) :: loopTimeKernel'+name)
    code('INTEGER(kind=4) :: numberCalled'+name)
    code('')

    if ninds > 0:
      code('TYPE ( c_ptr )  :: planRet_'+name)

    code('')
    code('CONTAINS')
    code('')

##########################################################################
#  Reduction kernel function - if an OP_GBL exists
##########################################################################
    if reduct_1dim:
      comm('Reduction cuda kernel'); depth = depth +2;
      code('attributes (device) SUBROUTINE ReductionFloat8(reductionResult,inputValue,reductionOperation)')
      code('REAL(kind=8), DIMENSION(:), DEVICE :: reductionResult')
      code('REAL(kind=8) :: inputValue')
      code('INTEGER(kind=4), VALUE :: reductionOperation')
      code('REAL(kind=8), DIMENSION(0:*), SHARED :: sharedDouble8')
      code('INTEGER(kind=4) :: i1')
      code('INTEGER(kind=4) :: threadID')
      code('threadID = threadIdx%x - 1')
      code('i1 = ishft(blockDim%x,-1)')
      code('CALL syncthreads()')
      code('sharedDouble8(threadID) = inputValue')

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

      code('attributes (device) SUBROUTINE ReductionInt4(reductionResult,inputValue,reductionOperation)')
      code('INTEGER(kind=4), DIMENSION(:), DEVICE :: reductionResult')
      code('INTEGER(kind=4) :: inputValue')
      code('INTEGER(kind=4), VALUE :: reductionOperation')
      code('INTEGER(kind=4), DIMENSION(0:*), SHARED :: sharedInt4')
      code('INTEGER(kind=4) :: i1')
      code('INTEGER(kind=4) :: threadID')
      code('threadID = threadIdx%x - 1')
      code('i1 = ishft(blockDim%x,-1)')
      code('CALL syncthreads()')
      code('sharedInt4(threadID) = inputValue')

      DOWHILE('i1 > 0')
      code('CALL syncthreads()')
      IF('threadID < i1')
      code('SELECT CASE(reductionOperation)')
      code('CASE (0)')
      code('sharedInt4(threadID) = sharedInt4(threadID) + sharedInt4(threadID + i1)')
      code('CASE (1)')
      IF('sharedInt4(threadID + i1) < sharedInt4(threadID)')
      code('sharedInt4(threadID) = sharedInt4(threadID + i1)')
      ENDIF()
      code('CASE (2)')
      IF('sharedInt4(threadID + i1) > sharedInt4(threadID)')
      code('sharedInt4(threadID) = sharedInt4(threadID + i1)')
      ENDIF()
      code('END SELECT')
      ENDIF()
      code('i1 = ishft(i1,-1)')
      ENDDO()

      code('CALL syncthreads()')

      IF('threadID .EQ. 0')
      code('SELECT CASE(reductionOperation)')
      code('CASE (0)')
      code('reductionResult(1) = reductionResult(1) + sharedInt4(0)')
      code('CASE (1)')
      IF('sharedInt4(0) < reductionResult(1)')
      code('reductionResult(1) = sharedInt4(0)')
      ENDIF()
      code('CASE (2)')
      IF('sharedInt4(0) > reductionResult(1)')
      code('reductionResult(1) = sharedInt4(0)')
      ENDIF()
      code('END SELECT')
      ENDIF()

      code('CALL syncthreads()')
      code('END SUBROUTINE')
      code('')
    if reduct_mdim:
      comm('Multidimensional reduction cuda kernel'); depth = depth +2;
      code('attributes (device) SUBROUTINE ReductionFloat8Mdim(reductionResult,inputValue,reductionOperation,dim)')
      code('REAL(kind=8), DIMENSION(:), DEVICE :: reductionResult')
      code('REAL(kind=8), DIMENSION(:) :: inputValue')
      code('INTEGER(kind=4), VALUE :: reductionOperation')
      code('INTEGER(kind=4), VALUE :: dim')
      code('REAL(kind=8), DIMENSION(0:*), SHARED :: sharedDouble8')
      code('INTEGER(kind=4) :: i1')
      code('INTEGER(kind=4) :: d')
      code('INTEGER(kind=4) :: threadID')
      code('threadID = threadIdx%x - 1')
      code('i1 = ishft(blockDim%x,-1)')
      code('CALL syncthreads()')
      code('sharedDouble8(threadID*dim:threadID*dim+dim-1) = inputValue(1:dim)')

      DOWHILE('i1 > 0')
      code('CALL syncthreads()')
      IF('threadID < i1')
      DO('i2','0','dim')
      code('sharedDouble8(threadID*dim + i2) = sharedDouble8(threadID*dim + i2) + sharedDouble8((threadID + i1)*dim + i2)')
      ENDDO()
      ENDIF()
      code('i1 = ishft(i1,-1)')
      ENDDO()

      code('CALL syncthreads()')

      IF('threadID .EQ. 0')
      code('reductionResult(1:dim) = reductionResult(1:dim) + sharedDouble8(0:dim-1)')
      ENDIF()

      code('CALL syncthreads()')
      code('END SUBROUTINE')
      code('')



##########################################################################
#  Inline user kernel function
##########################################################################
    if hydra:
      code('')
      comm(name + ' user functions (CPU and GPU)')
      code('')
      text = text.replace('module','!module')
      text = text.replace('contains','!contains')
      text = text.replace('end !module','!end module')
      text = text.replace('recursive subroutine','attributes(host) subroutine')
      text = text.replace('subroutine '+name, 'subroutine '+name+'_cpu')
      file_text += text
      code('')
      code('')
      i = text.find('const2.inc')
      if i > -1:
        fi2 = open("hydra_constants_list.txt","r")
        for line in fi2:
          fstr = '\\b'+line[:-1]+'\\b'
          rstr = line[:-1]+'_OP2CONSTANT'
          text = re.sub(fstr,rstr,text)
      text = text.replace('#include "const2.inc"','!#include "const2.inc"')
      text = text.replace('attributes(host) subroutine','attributes(device) subroutine')
      text = text.replace('subroutine '+name+'_cpu', 'subroutine '+name+'_gpu')
      text = text.replace('use BCS_KERNELS', '!use BCS_KERNELS')
      text = text.replace('use REALGAS_KERNELS', '!use REALGAS_KERNELS')
      text = text.replace('use UPDATE_KERNELS', '!use UPDATE_KERNELS')
      if ('BCFLUXK' in name) or ('INVISCBNDS' in name):
        code('#include "../../bcs_kernels_gpufun.inc"')
        kern_names = ['QRG_SET','OUTFLOW_FS','FREESTREAM','INFLOW','MP_INFLOW_CHAR','MP_OUTFLOW_CHAR','OUTFLOW','INFLOW_WHIRL','UNIQUE_INC','FILM_INJ','WFLUX','FFLUX']
        for i in range(0,12):
          text = text.replace('call '+kern_names[i]+'(', 'call '+kern_names[i]+'_gpu(')
          text = text.replace('call '+kern_names[i].lower()+'(', 'call '+kern_names[i].lower()+'_gpu(')
          text = text.replace('CALL '+kern_names[i]+'(', 'CALL '+kern_names[i]+'_gpu(')
      if 'call LOW' in text:
        kern_names = ['LOW','LOWH','LOWK']
        for i in range(0,3):
          text = text.replace('call '+kern_names[i]+'(', 'call '+kern_names[i]+'_gpu(')
          text = text.replace('CALL '+kern_names[i]+'(', 'CALL '+kern_names[i]+'_gpu(')
        code('#include "../../flux_low_gpufun.inc"')
      if ('INVJACS' in name):
        text = text.replace('call MATINV5(', 'call MATINV5_gpu(')
        code('#include "../../update_kernels_gpufun.inc"')
      file_text += text

    else:
      depth -= 2
      code('attributes (device) &')
      code('#include "'+name+'.inc"')
      depth += 2
      code('')

    code('')

##########################################################################
#  Generate CUDA kernel function
##########################################################################
    comm('CUDA kernel function')
    code('attributes (global) SUBROUTINE op_cuda_'+name+'( &'); depth = depth + 2
    if nopts >0:
      code('&  optflags,        &')
    if is_soa > -1:
      code('&  soa_stride,      &')
    for g_m in range(0,ninds):
      code('& opDat'+str(invinds[g_m]+1)+'Device'+name+', &')
      code('& opMap'+str(invinds[g_m]+1)+'Device'+name+', &')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code('& opDat'+str(g_m+1)+'Device'+name+', &')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL:
        if accs[g_m] == OP_INC or accs[g_m] == OP_MIN or accs[g_m] == OP_MAX:
          code('& reductionArrayDevice'+str(g_m+1)+',   &')
        elif accs[g_m] == OP_READ and dims[g_m].isdigit() and int(dims[g_m])==1:
          code('& opGblDat'+str(g_m+1)+'Device'+name+',   &')

    if ninds > 0: #indirect loop
      code('& pblkMap, &')
      code('& poffset, &')
      code('& pnelems, &')
      code('& pnthrcol, &')
      code('& pthrcol, &')
      code('& setSize, &')
      code('& blockOffset)')
    else: #direct loop
      code('& setSize)')

    code('')
    code('IMPLICIT NONE')
    code('')

##########################################################################
#  Declare local variables
##########################################################################
    comm('local variables')
    if nopts>0:
      code('INTEGER(kind=4), VALUE :: optflags')
    for g_m in range(0,ninds):
      if indaccs[g_m]==OP_READ:
        code(typs[invinds[g_m]]+', DEVICE, INTENT(IN) :: opDat'+str(invinds[g_m]+1)+'Device'+name+'(*)')
      else:
        code(typs[invinds[g_m]]+', DEVICE :: opDat'+str(invinds[g_m]+1)+'Device'+name+'(*)')
      code('INTEGER(kind=4), DEVICE :: opMap'+str(invinds[g_m]+1)+'Device'+name+'(*)')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        if accs[g_m] == OP_READ:
          code(typs[g_m]+', DEVICE, INTENT(IN) :: opDat'+str(g_m+1)+'Device'+name+'(*)')
        else:
          code(typs[g_m]+', DEVICE :: opDat'+str(g_m+1)+'Device'+name+'(*)')
    code('')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL:
        if accs[g_m] == OP_INC or accs[g_m] == OP_MIN or accs[g_m] == OP_MAX:
          #if it's a global reduction, then we pass in a reductionArrayDevice
          code(typs[g_m]+', DIMENSION(:), DEVICE :: reductionArrayDevice'+str(g_m+1))
          #and additionally we need registers to store contributions, depending on dim:
          if dims[g_m].isdigit() and int(dims[g_m]) == 1:
            code(typs[g_m]+' :: opGblDat'+str(g_m+1)+'Device'+name)
          else:
            code(typs[g_m]+', DIMENSION(0:'+dims[g_m]+'-1) :: opGblDat'+str(g_m+1)+'Device'+name)
        else:
          #if it's not  a global reduction, and multidimensional then we pass in a device array
          if dims[g_m].isdigit() and int(dims[g_m]) == 1:
            if accs[g_m] == OP_READ: #if OP_READ and dim 1, we can pass in by value
              code(typs[g_m]+', VALUE :: opGblDat'+str(g_m+1)+'Device'+name)

    if is_soa > -1:
      code('INTEGER(kind=4), VALUE :: soa_stride')
      for g_m in range(0,nargs):
        if soaflags[g_m] == 1 and (maps[g_m] <> OP_MAP or accs[g_m] <> OP_INC) and optflags[g_m]==0:
          code(typs[g_m]+', DIMENSION(0:'+dims[g_m]+'-1) :: opDat'+str(g_m+1)+'SoALocal')

    if ninds > 0: #indirect loop
      code('INTEGER(kind=4), DIMENSION(0:*), DEVICE :: pblkMap')
      code('INTEGER(kind=4), DIMENSION(0:*), DEVICE :: poffset')
      code('INTEGER(kind=4), DIMENSION(0:*), DEVICE :: pnelems')
      code('INTEGER(kind=4), DIMENSION(0:*), DEVICE :: pnthrcol')
      code('INTEGER(kind=4), DIMENSION(0:*), DEVICE :: pthrcol')
      code('INTEGER(kind=4), VALUE :: blockOffset')
      code('INTEGER(kind=4), VALUE :: setSize')
      code('')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and accs[g_m] == OP_INC:
          if dims[g_m].isdigit() and int(dims[g_m]) == 1:
            code(typs[g_m]+' :: opDat'+str(g_m+1)+'Local')
          else:
            code(typs[g_m]+', DIMENSION(0:'+dims[g_m]+'-1) :: opDat'+str(g_m+1)+'Local')

      code('')

      code('INTEGER(kind=4), SHARED :: numOfColours')
      code('INTEGER(kind=4), SHARED :: numberOfActiveThreadsCeiling')
      code('INTEGER(kind=4), SHARED :: blockID')
      code('INTEGER(kind=4), SHARED :: threadBlockOffset')
      code('INTEGER(kind=4), SHARED :: numberOfActiveThreads')
      code('INTEGER(kind=4) :: colour1')
      code('INTEGER(kind=4) :: colour2')
      code('INTEGER(kind=4) :: n1')
      code('INTEGER(kind=4) :: i1')
      code('INTEGER(kind=4) :: i2')

    else: #direct loop
      code('INTEGER(kind=4), VALUE :: setSize')
      code('INTEGER(kind=4) :: i1')
      if is_soa > -1:
        code('INTEGER(kind=4) :: i2')

    if nopts > 0:
      code('')
      comm('optional variables')
  # for indirect OP_READ, we would pass in a pointer to shared, offset by map, but if opt, then map may not exist, thus we need a separate pointer
      for g_m in range(0,nargs):
        if (accs[g_m] == OP_READ or accs[g_m] == OP_RW or accs[g_m] == OP_WRITE) and maps[g_m] == OP_MAP and optflags[g_m]==1:
          if dims[g_m].isdigit() and int(dims[g_m])==1:
            code(typs[g_m]+' :: opDat'+str(g_m+1)+'Opt')
          else:
            code(typs[g_m]+', DIMENSION(0:'+dims[g_m]+'-1) :: opDat'+str(g_m+1)+'Opt')

    code('')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and accs[g_m] == OP_INC:
        code('opGblDat'+str(g_m+1)+'Device'+name+' = 0')

    code('')
    if ninds > 0:
      IF('threadIdx%x - 1 .EQ. 0')
      code('blockID = pblkMap(blockIdx%x - 1 + blockOffset)')
      code('numberOfActiveThreads = pnelems(blockID)')
      code('numberOfActiveThreadsCeiling = blockDim%x * (1 + (numberOfActiveThreads - 1) / blockDim%x)')
      code('numOfColours = pnthrcol(blockID)')
      code('threadBlockOffset = poffset(blockID)')
      code('')
      ENDIF()

      code('')
      code('CALL syncthreads()')
      code('')
      code('i1 = threadIdx%x - 1')
      code('')


      DOWHILE('i1 < numberOfActiveThreadsCeiling')
      if ind_inc or ind_rw:
        code('colour2 = -1')
      #-----Begin Indirect RW handling-----
      if ind_rw:
        DO('colour1','0','numOfColours')
        IF('i1 < numberOfActiveThreads')
        code('colour2 = pthrcol(i1 + threadBlockOffset)')
        IF('colour2 .EQ. colour1')
      #-----End Indirect RW handling-----
      else:
        IF('i1 < numberOfActiveThreads')

      for g_m in range(0,nargs):
        if accs[g_m] == OP_INC and maps[g_m] == OP_MAP:
          if dims[g_m].isdigit() and int(dims[g_m]) == 1:
            code('opDat'+str(g_m+1)+'Local = 0')
          else:
            DO('i2','0',dims[g_m])
            code('opDat'+str(g_m+1)+'Local(i2) = 0')
            ENDDO()

      for g_m in range(0,nargs):
        if (accs[g_m] == OP_READ or accs[g_m] == OP_RW or accs[g_m] == OP_WRITE) and maps[g_m] == OP_MAP and optflags[g_m]==1:
          IF('BTEST(optflags,'+str(optidxs[g_m])+')')
          if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
            DO('i2','0', dims[g_m])
            code('opDat'+str(g_m+1)+'Opt(i2) = opDat'+str(invinds[inds[g_m]-1]+1)+'Device'+name+ ' &')
            code('  & (1 + i2 + opMap'+str(invinds[inds[g_m]-1]+1)+'Device'+name+'(1 + i1 + threadBlockOffset + setSize * '+str(int(idxs[g_m])-1)+') * ('+dims[g_m]+'))')
            ENDDO()
          else:
            code('opDat'+str(g_m+1)+'Opt = opDat'+str(invinds[inds[g_m]-1]+1)+'Device'+name+ \
            '(1 + opMap'+str(invinds[inds[g_m]-1]+1)+'Device'+name+'(1 + i1 + threadBlockOffset + setSize * '+str(int(idxs[g_m])-1)+'))')
          ENDIF()

      for g_m in range(0,nargs):
        if soaflags[g_m] == 1 and (maps[g_m] <> OP_MAP or accs[g_m] <> OP_INC) and optflags[g_m]==0:
          DO('i2','0', dims[g_m])
          if maps[g_m] == OP_MAP:
            code('opDat'+str(g_m+1)+'SoALocal(i2) = opDat'+str(invinds[inds[g_m]-1]+1)+'Device'+name+ ' &')
            code('  & (1 + i2 * soa_stride + opMap'+str(invinds[inds[g_m]-1]+1)+'Device'+name+'(1 + i1 + threadBlockOffset + setSize * '+str(int(idxs[g_m])-1)+'))')
          else:
            code('opDat'+str(g_m+1)+'SoALocal(i2) = opDat'+str(g_m+1)+'Device'+name+ '(1 + i2 * soa_stride + (i1 + threadBlockOffset))')
          ENDDO()

      code('')
      comm('kernel call')

    else:
      DO_STEP('i1','threadIdx%x - 1 + (blockIdx%x - 1) * blockDim%x','setSize','blockDim%x * gridDim%x')
      for g_m in range(0,nargs):
        if soaflags[g_m] == 1 and optflags[g_m]==0:
          DO('i2','0', dims[g_m])
          code('opDat'+str(g_m+1)+'SoALocal(i2) = opDat'+str(g_m+1)+'Device'+name+ '(1 + i2 * soa_stride + i1)')
          ENDDO()
      code('')
      comm('kernel call')
      code('')

##########################################################################
#  CUDA kernel call
##########################################################################
    if ninds > 0: #indirect kernel call
      line = '  CALL '+name+'_gpu( &'
      indent = '\n'+' '*depth
      for g_m in range(0,nargs):
        if soaflags[g_m] == 1 and (maps[g_m] <> OP_MAP or accs[g_m] <> OP_INC) and optflags[g_m]==0:
          line = line +indent + '& opDat'+str(g_m+1)+'SoALocal'
        elif maps[g_m] == OP_ID:
          if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
            line = line + indent + '& opDat'+str(g_m+1)+'Device'+name+ \
            '((i1 + threadBlockOffset) * ('+dims[g_m]+') +1' + \
            ':(i1 + threadBlockOffset) * ('+dims[g_m]+') + ('+dims[g_m]+'))'
          else:
            line = line + indent + '& opDat'+str(g_m+1)+'Device'+name+ \
            '((i1 + threadBlockOffset) * ('+dims[g_m]+') +1)'
        elif maps[g_m] == OP_MAP and (accs[g_m] == OP_READ or accs[g_m] == OP_RW or accs[g_m] == OP_WRITE) and optflags[g_m]==0:
          if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
            line = line +indent + '& opDat'+str(invinds[inds[g_m]-1]+1)+'Device'+name+ \
            '(1 + opMap'+str(invinds[inds[g_m]-1]+1)+'Device'+name+'(1 + i1 + threadBlockOffset + setSize * '+str(int(idxs[g_m])-1)+') * ('+dims[g_m]+'):'+ \
            '     opMap'+str(invinds[inds[g_m]-1]+1)+'Device'+name+'(1 + i1 + threadBlockOffset + setSize * '+str(int(idxs[g_m])-1)+') * ('+dims[g_m]+') + '+dims[g_m]+')'
          else:
            line = line +indent + '& opDat'+str(invinds[inds[g_m]-1]+1)+'Device'+name+ \
            '(1 + opMap'+str(invinds[inds[g_m]-1]+1)+'Device'+name+'(1 + i1 + threadBlockOffset + setSize * '+str(int(idxs[g_m])-1)+'))'
        elif maps[g_m] == OP_MAP and (accs[g_m] == OP_READ or accs[g_m] == OP_RW or accs[g_m] == OP_WRITE) and optflags[g_m]==1:
#          if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
#            line = line +indent + '& opDat'+str(g_m+1)+'Opt'
#          else:
          line = line +indent + '& opDat'+str(g_m+1)+'Opt'
        elif maps[g_m] == OP_MAP and accs[g_m] == OP_INC:
          if dims[g_m].isdigit() and int(dims[g_m])==1:
            line = line +indent + '& opDat'+str(g_m+1)+'Local'
          else:
            line = line +indent + '& opDat'+str(g_m+1)+'Local'
        elif maps[g_m] == OP_GBL:
          if accs[g_m] == OP_WRITE and dims[g_m].isdigit() and int(dims[g_m]) == 1:
            line = line + indent +'& opGblDat'+str(g_m+1)+'Device'+name+'(1)'
          else:
            line = line + indent +'& opGblDat'+str(g_m+1)+'Device'+name
        if g_m < nargs-1:
          line = line +', &'
        else:
           line = line +' &'
      depth = depth - 2
      code(line + indent + '& )')
      depth = depth + 2
      code('')
      #write optional/SoA arguments back from registers
      for g_m in range(0,nargs):
        if (accs[g_m] == OP_RW or accs[g_m] == OP_WRITE) and maps[g_m] == OP_MAP and optflags[g_m]==1:
          IF('BTEST(optflags,'+str(optidxs[g_m])+')')
          if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
            DO('i2','0', dims[g_m])
            code('opDat'+str(invinds[inds[g_m]-1]+1)+'Device'+name+ ' &')
            code('  & (1 + i2 + opMap'+str(invinds[inds[g_m]-1]+1)+'Device'+name+'(1 + i1 + threadBlockOffset + setSize * '+str(int(idxs[g_m])-1)+') * ('+dims[g_m]+')) = opDat'+str(g_m+1)+'Opt(i2)')
            ENDDO()
          else:
            code('opDat'+str(invinds[inds[g_m]-1]+1)+'Device'+name+ \
            '(1 + opMap'+str(invinds[inds[g_m]-1]+1)+'Device'+name+'(1 + i1 + threadBlockOffset + setSize * '+str(int(idxs[g_m])-1)+')) = opDat'+str(g_m+1)+'Opt')
          ENDIF()
        if soaflags[g_m] == 1 and (maps[g_m] <> OP_MAP or accs[g_m] <> OP_INC) and accs[g_m] <> OP_READ:
          DO('i2','0', dims[g_m])
          if maps[g_m] == OP_MAP:
            code('opDat'+str(invinds[inds[g_m]-1]+1)+'Device'+name+ ' &')
            code('  & (1 + i2 * soa_stride + opMap'+str(invinds[inds[g_m]-1]+1)+'Device'+name+'(1 + i1 + threadBlockOffset + setSize * '+str(int(idxs[g_m])-1)+')) = opDat'+str(g_m+1)+'SoALocal(i2)')
          else:
            code('opDat'+str(g_m+1)+'Device'+name+ '(1 + i2 * soa_stride + (i1 + threadBlockOffset)) = opDat'+str(g_m+1)+'SoALocal(i2)')
          ENDDO()

      if ind_inc and not ind_rw:
        code('colour2 = pthrcol(i1 + threadBlockOffset)')
      if not ind_rw:
        ENDIF()

      if ind_inc or ind_rw:
        if ind_inc and not ind_rw:
          DO('colour1','0','numOfColours')
          IF('colour2 .EQ. colour1')
        for g_m in range(0,nargs):
          if optflags[g_m]==1 and maps[g_m]==OP_MAP and accs[g_m] == OP_INC:
            IF('BTEST(optflags,'+str(optidxs[g_m])+')')
          if accs[g_m] == OP_INC and maps[g_m] == OP_MAP:
            if dims[g_m].isdigit() and int(dims[g_m])==1:
              code('opDat'+str(invinds[inds[g_m]-1]+1)+'Device'+name+ \
              '(1 + opMap'+str(invinds[inds[g_m]-1]+1)+'Device'+name+'(1 + i1 + threadBlockOffset + setSize * '+str(int(idxs[g_m])-1)+') * ('+dims[g_m]+')) = &')
              code('& opDat'+str(invinds[inds[g_m]-1]+1)+'Device'+name+ \
              '(1 + opMap'+str(invinds[inds[g_m]-1]+1)+'Device'+name+'(1 + i1 + threadBlockOffset + setSize * '+str(int(idxs[g_m])-1)+') * ('+dims[g_m]+')) + opDat'+str(g_m+1)+'Local')
            else:
              if soaflags[g_m] == 1:
                DO('i2','0', dims[g_m])
                code('opDat'+str(invinds[inds[g_m]-1]+1)+'Device'+name+ \
                '(1 + i2*soa_stride + opMap'+str(invinds[inds[g_m]-1]+1)+'Device'+name+'(1 + i1 + threadBlockOffset + setSize * '+str(int(idxs[g_m])-1)+')) = &')
                code('& opDat'+str(invinds[inds[g_m]-1]+1)+'Device'+name+ \
                '(1 + i2*soa_stride + opMap'+str(invinds[inds[g_m]-1]+1)+'Device'+name+'(1 + i1 + threadBlockOffset + setSize * '+str(int(idxs[g_m])-1)+')) + opDat'+str(g_m+1)+'Local(i2)')
                ENDDO()
              else:
                DO('i2','0', dims[g_m])
                code('opDat'+str(invinds[inds[g_m]-1]+1)+'Device'+name+ \
                '(1 + i2 + opMap'+str(invinds[inds[g_m]-1]+1)+'Device'+name+'(1 + i1 + threadBlockOffset + setSize * '+str(int(idxs[g_m])-1)+') * ('+dims[g_m]+')) = &')
                code('& opDat'+str(invinds[inds[g_m]-1]+1)+'Device'+name+ \
                '(1 + i2 + opMap'+str(invinds[inds[g_m]-1]+1)+'Device'+name+'(1 + i1 + threadBlockOffset + setSize * '+str(int(idxs[g_m])-1)+') * ('+dims[g_m]+')) + opDat'+str(g_m+1)+'Local(i2)')
                ENDDO()
            if optflags[g_m]<>1:
              code('')
          if optflags[g_m]==1 and maps[g_m]==OP_MAP and (accs[g_m] == OP_INC):
            ENDIF()
            code('')
        ENDIF()
        if ind_rw:
          ENDIF()
        code('CALL syncthreads()')
        ENDDO()
      code('i1 = i1 + blockDim%x')
      ENDDO()
      code('')

    else: #direct kernel call
      line = '  CALL '+name+'_gpu( &'
      indent = '\n'+' '*depth
      for g_m in range(0,nargs):
        if soaflags[g_m] == 1 and (maps[g_m] <> OP_MAP or accs[g_m] <> OP_INC) and optflags[g_m]==0:
          line = line +indent + '& opDat'+str(g_m+1)+'SoALocal'
        elif maps[g_m] == OP_GBL:
          if accs[g_m] == OP_WRITE and dims[g_m].isdigit() and int(dims[g_m]) == 1:
            line = line + indent +'& opGblDat'+str(g_m+1)+'Device'+name+'(1)'
          else:
            line = line + indent +'& opGblDat'+str(g_m+1)+'Device'+name
        else:
          if dims[g_m].isdigit() and int(dims[g_m]) == 1:
            line = line + indent +'& opDat'+str(g_m+1)+'Device'+name+'(i1 + 1)'
          else:
            line = line + indent +'& opDat'+str(g_m+1)+'Device'+name+'(i1 * ('+dims[g_m]+') + 1: i1 * ('+dims[g_m]+') + '+dims[g_m]+')'

        if g_m < nargs-1:
          line = line + ', &'
        else:
          line = line + ' &'
      depth = depth - 2
      code(line + indent +  '& )')
      depth = depth + 2
      for g_m in range(0,nargs):
        if soaflags[g_m] == 1 and accs[g_m] <> OP_READ:
          DO('i2','0', dims[g_m])
          code('opDat'+str(g_m+1)+'Device'+name+ '(1 + i2 * soa_stride + i1) = opDat'+str(g_m+1)+'SoALocal(i2)')
          ENDDO()
      code('')
      ENDDO()

    #call cuda reduction for each OP_GBL
    code('')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and (accs[g_m] == OP_INC or accs[g_m] == OP_MIN or accs[g_m] == OP_MAX):
        if 'real' in typs[g_m].lower():
          if dims[g_m].isdigit() and int(dims[g_m])==1:
            code('CALL ReductionFloat8(reductionArrayDevice'+str(g_m+1)+'(blockIdx%x - 1 + 1:),opGblDat'+str(g_m+1)+'Device'+name+',0)')
          else:
            code('CALL ReductionFloat8Mdim(reductionArrayDevice'+str(g_m+1)+'((blockIdx%x - 1)*('+dims[g_m]+') + 1:),opGblDat'+str(g_m+1)+'Device'+name+',0,'+dims[g_m]+')')
        elif 'integer' in typs[g_m].lower():
          if dims[g_m].isdigit() and int(dims[g_m])==1:
            code('CALL ReductionInt4(reductionArrayDevice'+str(g_m+1)+'(blockIdx%x - 1 + 1:),opGblDat'+str(g_m+1)+'Device'+name+',0)')
          else:
            code('CALL ReductionInt4Mdim(reductionArrayDevice'+str(g_m+1)+'((blockIdx%x - 1)*('+dims[g_m]+') + 1:),opGblDat'+str(g_m+1)+'Device'+name+',0,'+dims[g_m]+')')
    code('')

    depth = depth - 2
    code('END SUBROUTINE')
    code('')

##########################################################################
#  Generate CPU hust stub
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
    IF('getHybridGPU()')
    code('CALL '+name+'_host_gpu( userSubroutine, set, &');
    for g_m in range(0,nargs):
      if g_m == nargs-1:
        code('& opArg'+str(g_m+1)+' )')
      else:
        code('& opArg'+str(g_m+1)+', &')
    ELSE()
    code('CALL '+name+'_host_cpu( userSubroutine, set, &');
    for g_m in range(0,nargs):
      if g_m == nargs-1:
        code('& opArg'+str(g_m+1)+' )')
      else:
        code('& opArg'+str(g_m+1)+', &')
    ENDIF()
    depth = depth - 2
    code('END SUBROUTINE')
    code('')
    code('')
    comm('Stub for GPU execution')
    code('')
    code('attributes (host) SUBROUTINE '+name+'_host_gpu( userSubroutine, set, &'); depth = depth + 2
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
    code('INTEGER(kind=4) :: n_upper')
    code('INTEGER(kind=4) :: returnSetKernelTiming')
    code('')
    code('')

    for g_m in range(0,ninds):
      code(typs[invinds[g_m]]+', DIMENSION(:), DEVICE, ALLOCATABLE :: opDat'+str(invinds[g_m]+1)+'Device'+name)
      code('INTEGER(kind=4), DIMENSION(:), DEVICE, ALLOCATABLE :: opMap'+str(invinds[g_m]+1)+'Device'+name)
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code(typs[g_m]+', DIMENSION(:), DEVICE, ALLOCATABLE :: opDat'+str(g_m+1)+'Device'+name)
    code('')

    for g_m in range(0,ninds):
      code('INTEGER(kind=4) :: opDat'+str(invinds[g_m]+1)+'Cardinality')
      code('INTEGER(kind=4) :: opMap'+str(invinds[g_m]+1)+'Cardinality')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code('INTEGER(kind=4) :: opDat'+str(g_m+1)+'Cardinality')
      elif maps[g_m] == OP_GBL:
        code('INTEGER(kind=4) :: opDat'+str(g_m+1)+'Cardinality')
    code('')

    if ninds > 0: #indirect loop
      code('TYPE ( op_plan ) , POINTER :: actualPlan_'+name+'')
      code('')
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
      for g_m in range(0,ninds):
          code('INTEGER(kind=4) :: mappingArray'+str(invinds[g_m]+1)+'Size')
      code('')

      code('INTEGER(kind=4) :: numberOfIndirectOpDats')
      code('INTEGER(kind=4) :: blockOffset')
      code('INTEGER(kind=4) :: pblkMapSize')
      code('INTEGER(kind=4) :: poffsetSize')
      code('INTEGER(kind=4) :: pnelemsSize')
      code('INTEGER(kind=4) :: pnthrcolSize')
      code('INTEGER(kind=4) :: pthrcolSize')
      code('INTEGER(kind=4), POINTER, DIMENSION(:) :: ncolblk')
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
      code('')

    else: #direct loop
      code('INTEGER(kind=4) :: blocksPerGrid')
      code('INTEGER(kind=4) :: threadsPerBlock')
      code('INTEGER(kind=4) :: dynamicSharedMemorySize')
      code('INTEGER(kind=4) :: threadSynchRet')
      code('INTEGER(kind=4), SAVE :: calledTimes')
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
        if accs[g_m] == OP_WRITE or (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
          code(typs[g_m]+', DIMENSION(:), POINTER :: opDat'+str(g_m+1)+'Host')
        else:
          code(typs[g_m]+', POINTER :: opDat'+str(g_m+1)+'Host')
        if (accs[g_m] == OP_INC or accs[g_m] == OP_MAX or accs[g_m] == OP_MIN):
          code(typs[g_m]+', DIMENSION(:), ALLOCATABLE :: reductionArrayHost'+str(g_m+1))
          code('INTEGER(kind=4) :: reductionCardinality'+str(g_m+1))

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

    code('n_upper = op_mpi_halo_exchanges_cuda(set%setCPtr,numberOfOpDats,opArgArray)')
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
      code('partitionSize = getPartitionSize(userSubroutine//C_NULL_CHAR,set%setPtr%size)')
      code('')
      code('planRet_'+name+' = FortranPlanCaller( &')
      code('& userSubroutine//C_NULL_CHAR, &')
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
      code('threadsPerBlock = getBlockSize(userSubroutine//C_NULL_CHAR,set%setPtr%size)')
      code('dynamicSharedMemorySize = reductionSize(opArgArray,numberOfOpDats) * threadsPerBlock')
      code('')


    for g_m in range(0,ninds):
      code('opDat'+str(invinds[g_m]+1)+'Cardinality = opArg'+str(invinds[g_m]+1)+'%dim * getSetSizeFromOpArg(opArg'+str(invinds[g_m]+1)+')')
      code('opMap'+str(invinds[g_m]+1)+'Cardinality = set%setPtr%size * getMapDimFromOpArg(opArg'+str(invinds[g_m]+1)+')')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code('opDat'+str(g_m+1)+'Cardinality = opArg'+str(g_m+1)+'%dim * getSetSizeFromOpArg(opArg'+str(g_m+1)+')')
      elif maps[g_m] == OP_GBL:
        code('opDat'+str(g_m+1)+'Cardinality = opArg'+str(g_m+1)+'%dim')
    code('')

    code('')
    for g_m in range(0,ninds):
      code('CALL c_f_pointer(opArg'+str(invinds[g_m]+1)+'%data_d,opDat'+str(invinds[g_m]+1)+'Device'+name+',(/opDat'+str(invinds[g_m]+1)+'Cardinality/))')
      code('CALL c_f_pointer(opArg'+str(invinds[g_m]+1)+'%map_data_d,opMap'+str(invinds[g_m]+1)+'Device'+name+',(/opMap'+str(invinds[g_m]+1)+'Cardinality/))')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code('CALL c_f_pointer(opArg'+str(g_m+1)+'%data_d,opDat'+str(g_m+1)+'Device'+name+',(/opDat'+str(g_m+1)+'Cardinality/))')
      elif maps[g_m] == OP_GBL:
        if accs[g_m] == OP_WRITE or (not dims[g_m].isdigit()) or int(dims[g_m])>1:
          code('CALL c_f_pointer(opArg'+str(g_m+1)+'%data,opDat'+str(g_m+1)+'Host,(/opDat'+str(g_m+1)+'Cardinality/))')
        else:
          code('CALL c_f_pointer(opArg'+str(g_m+1)+'%data,opDat'+str(g_m+1)+'Host)')
    code('')

    if ninds > 0:
      code('CALL c_f_pointer(planRet_'+name+',actualPlan_'+name+')')
      code('CALL c_f_pointer(actualPlan_'+name+'%ncolblk,ncolblk,(/set%setPtr%size/))')
      code('pblkMapSize = actualPlan_'+name+'%nblocks')
      code('CALL c_f_pointer(actualPlan_'+name+'%blkmap_d,pblkMap,(/pblkMapSize/))')
      code('poffsetSize = actualPlan_'+name+'%nblocks')
      code('CALL c_f_pointer(actualPlan_'+name+'%offset_d,poffset,(/poffsetSize/))')
      code('pnelemsSize = actualPlan_'+name+'%nblocks')
      code('CALL c_f_pointer(actualPlan_'+name+'%nelems_d,pnelems,(/pnelemsSize/))')
      code('pnthrcolSize = actualPlan_'+name+'%nblocks')
      code('CALL c_f_pointer(actualPlan_'+name+'%nthrcol,pnthrcol,(/pnthrcolSize/))')
      code('pthrcolSize = set%setPtr%size')
      code('CALL c_f_pointer(actualPlan_'+name+'%thrcol,pthrcol,(/pthrcolSize/))')
      code('')

    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and ((accs[g_m]==OP_READ and ((not dims[g_m].isdigit()) or int(dims[g_m]) > 1)) or accs[g_m]==OP_WRITE):
        IF('.not. allocated(opGblDat'+str(g_m+1)+'Device'+name+')')
        code('allocate(opGblDat'+str(g_m+1)+'Device'+name+'(opArg'+str(g_m+1)+'%dim))')
        ENDIF()
        code('opGblDat'+str(g_m+1)+'Device'+name+'(1:opArg'+str(g_m+1)+'%dim) = opDat'+str(g_m+1)+'Host(1:opArg'+str(g_m+1)+'%dim)')

    #setup for reduction
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and (accs[g_m] == OP_INC or accs[g_m] == OP_MAX or accs[g_m] == OP_MIN):
        code('reductionCardinality'+str(g_m+1)+' = blocksPerGrid * 1')
        code('allocate( reductionArrayHost'+str(g_m+1)+'(reductionCardinality'+str(g_m+1)+'* ('+dims[g_m]+')) )')
        IF ('.not. allocated(reductionArrayDevice'+str(g_m+1)+name+')')
        code('allocate( reductionArrayDevice'+str(g_m+1)+name+'(reductionCardinality'+str(g_m+1)+'* ('+dims[g_m]+')) )')
        ENDIF()
        code('')
        DO('i10','0','reductionCardinality'+str(g_m+1)+'')
        if dims[g_m].isdigit() and int(dims[g_m]) == 1:
          code('reductionArrayHost'+str(g_m+1)+'(i10+1) = 0.0')
        else:
          code('reductionArrayHost'+str(g_m+1)+'(i10 * ('+dims[g_m]+') + 1 : i10 * ('+dims[g_m]+') + ('+dims[g_m]+')) = 0.0')
        ENDDO()
        code('')
        code('reductionArrayDevice'+str(g_m+1)+name+' = reductionArrayHost'+str(g_m+1)+'')

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
      code('threadsPerBlock = getBlockSize(userSubroutine//C_NULL_CHAR,set%setPtr%size)')

      DO('i2','0','actualPlan_'+name+'%ncolors')
      IF('i2 .EQ. actualPlan_'+name+'%ncolors_core')
      code('CALL op_mpi_wait_all_cuda(numberOfOpDats,opArgArray)')
      ENDIF()
      code('')
      code('blocksPerGrid = ncolblk(i2 + 1)')
      code('dynamicSharedMemorySize = reductionSize(opArgArray,numberOfOpDats) * threadsPerBlock')
      code('')
      code('CALL op_cuda_'+name+' <<<blocksPerGrid,threadsPerBlock,dynamicSharedMemorySize>>> (&')
      if nopts>0:
        code('& optflags, &')
      if is_soa > -1:
        code('& getSetSizeFromOpArg(opArg'+str(is_soa+1)+'), &')
      for g_m in range(0,ninds):
        code('& opDat'+str(invinds[g_m]+1)+'Device'+name+', &')
        code('& opMap'+str(invinds[g_m]+1)+'Device'+name+', &')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          code('& opDat'+str(g_m+1)+'Device'+name+', &')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL and (accs[g_m] == OP_INC or accs[g_m] == OP_MIN or accs[g_m] == OP_MAX):
          code('reductionArrayDevice'+str(g_m+1)+name+', &')
        if maps[g_m] == OP_GBL and accs[g_m] == OP_READ and dims[g_m].isdigit() and int(dims[g_m])==1:
          code('& opDat'+str(g_m+1)+'Host, &')
      code('& pblkMap, &')
      code('& poffset,pnelems,pnthrcol,pthrcol,set%setPtr%size+set%setPtr%exec_size, blockOffset)')
      code('')
      code('blockOffset = blockOffset + blocksPerGrid')
      ENDDO()
      code('')
    else: #direct loop host stub call
      code('CALL op_cuda_'+name+' <<<blocksPerGrid,threadsPerBlock,dynamicSharedMemorySize>>>( &')
      if nopts>0:
        code('& optflags, &')
      if is_soa > -1:
        code('& getSetSizeFromOpArg(opArg'+str(is_soa+1)+'), &')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          code('& opDat'+str(g_m+1)+'Device'+name+', &')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL and (accs[g_m] == OP_INC or accs[g_m] == OP_MIN or accs[g_m] == OP_MAX):
          code('reductionArrayDevice'+str(g_m+1)+name+', &')
        if maps[g_m] == OP_GBL and accs[g_m] == OP_READ and dims[g_m].isdigit() and int(dims[g_m])==1:
          code('& opDat'+str(g_m+1)+'Host, &')
      code('set%setPtr%size)')

    code('')
    IF('(n_upper .EQ. 0) .OR. (n_upper .EQ. set%setPtr%core_size)')
    code('CALL op_mpi_wait_all_cuda(numberOfOpDats,opArgArray)')
    ENDIF()
    code('')

    code('')
    code('istat = cudaEventRecord(endTimeKernel,0)')
    code('istat = cudaEventSynchronize(endTimeKernel)')
    code('istat = cudaEventElapsedTime(accumulatorKernelTime,startTimeKernel,endTimeKernel)')
    code('loopTimeKernel'+name+' = loopTimeKernel'+name+' + accumulatorKernelTime')
    code('')

    code('')
    code('CALL op_mpi_set_dirtybit_cuda(numberOfOpDats,opArgArray)')
    code('')

    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and accs[g_m] == OP_WRITE:
        code('opDat'+str(g_m+1)+'Host(1:opArg'+str(g_m+1)+'%dim) = opGblDat'+str(g_m+1)+'Device'+name+'(1:opArg'+str(g_m+1)+'%dim)')

    if reduct:
      code('istat = cudaEventRecord(startTimeHost,0)') #timer for reduction
      #reductions
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL and accs[g_m] == OP_INC:
          code('reductionArrayHost'+str(g_m+1)+' = reductionArrayDevice'+str(g_m+1)+name+'')
          code('')
          DO('i10','0','reductionCardinality'+str(g_m+1)+'')
          if dims[g_m].isdigit() and int(dims[g_m]) == 1:
            code('opDat'+str(g_m+1)+'Host = opDat'+str(g_m+1)+'Host + reductionArrayHost'+str(g_m+1)+'(i10+1)')
          else:
            code('opDat'+str(g_m+1)+'Host(1:'+dims[g_m]+') = opDat'+str(g_m+1)+'Host(1:'+dims[g_m]+') + reductionArrayHost'+str(g_m+1)+'(i10 * ('+dims[g_m]+') + 1 : i10 * ('+dims[g_m]+') + ('+dims[g_m]+'))')
          ENDDO()
          code('')
          code('deallocate( reductionArrayHost'+str(g_m+1)+' )')
#          code('deallocate( reductionArrayDevice'+str(g_m+1)+' )')
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

    code('calledTimes = calledTimes + 1')
    depth = depth - 2
    code('END SUBROUTINE')
    code('')
    code('')
    comm('Stub for CPU execution')
    code('')
##########################################################################
#  Generate OpenMP host stub
##########################################################################
    code('SUBROUTINE '+name+'_host_cpu( userSubroutine, set, &'); depth = depth + 2
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

    code('INTEGER(kind=4) :: threadID')
    code('INTEGER(kind=4) :: numberOfThreads')
    code('INTEGER(kind=4), DIMENSION(1:8) :: timeArrayStart')
    code('INTEGER(kind=4), DIMENSION(1:8) :: timeArrayEnd')
    code('REAL(kind=8) :: startTimeHost')
    code('REAL(kind=8) :: endTimeHost')
    code('REAL(kind=8) :: startTimeKernel')
    code('REAL(kind=8) :: endTimeKernel')
    code('REAL(kind=8) :: accumulatorHostTime')
    code('REAL(kind=8) :: accumulatorKernelTime')
    code('INTEGER(kind=4) :: returnSetKernelTiming')

    if ninds > 0: #if indirect loop
      code('LOGICAL :: firstTime_'+name+' = .TRUE.')
      code('type ( c_ptr )  :: planRet_'+name)
      code('type ( op_plan ) , POINTER :: actualPlan_'+name)
      code('INTEGER(kind=4), POINTER, DIMENSION(:) :: ncolblk_'+name)
      code('INTEGER(kind=4), POINTER, DIMENSION(:) :: blkmap_'+name)
      code('INTEGER(kind=4), POINTER, DIMENSION(:) :: nelems_'+name)
      code('INTEGER(kind=4), POINTER, DIMENSION(:) :: offset_'+name)
      code('INTEGER(kind=4), DIMENSION(1:'+str(nargs)+') :: indirectionDescriptorArray')
      code('INTEGER(kind=4) :: numberOfIndirectOpDats')
      code('INTEGER(kind=4) :: blockOffset')
      code('INTEGER(kind=4) :: nblocks')
      code('INTEGER(kind=4) :: partitionSize')
      code('INTEGER(kind=4) :: blockID')
      code('INTEGER(kind=4) :: nelem')
      code('INTEGER(kind=4) :: offset_b')
    else:
      code('INTEGER(kind=4) :: sliceStart')
      code('INTEGER(kind=4) :: sliceEnd')

    code('')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and accs[g_m] == OP_INC:
        code(typs[g_m]+', DIMENSION(:), ALLOCATABLE :: reductionArrayHost'+str(g_m+1))

    code('')
    code('INTEGER(kind=4) :: i1,i2,n')

    code('')
    code('numberOfOpDats = '+str(nargs))
    code('')

    for g_m in range(0,nargs):
      code('opArgArray('+str(g_m+1)+') = opArg'+str(g_m+1))
    code('')


    #mpi halo exchange call
    code('n_upper = op_mpi_halo_exchanges(set%setCPtr,numberOfOpDats,opArgArray)')

    code('numberCalled'+name+' = numberCalled'+name+'+ 1')
    code('')
    code('call date_and_time(values=timeArrayStart)')
    code('startTimeHost = 1.00000 * timeArrayStart(8) + &')
    code('& 1000.00 * timeArrayStart(7) + &')
    code('& 60000 * timeArrayStart(6) + &')
    code('& 3600000 * timeArrayStart(5)')
    code('')

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
      code('& indirectionDescriptorArray)')
      code('')
      code('CALL c_f_pointer(planRet_'+name+',actualPlan_'+name+')')
      code('CALL c_f_pointer(actualPlan_'+name+'%ncolblk,ncolblk_'+name+',(/actualPlan_'+name+'%ncolors_core/))')
      code('CALL c_f_pointer(actualPlan_'+name+'%blkmap,blkmap_'+name+',(/actualPlan_'+name+'%nblocks/))')
      code('CALL c_f_pointer(actualPlan_'+name+'%offset,offset_'+name+',(/actualPlan_'+name+'%nblocks/))')
      code('CALL c_f_pointer(actualPlan_'+name+'%nelems,nelems_'+name+',(/actualPlan_'+name+'%nblocks/))')

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

    #reductions
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and accs[g_m] == OP_INC:
        code('allocate( reductionArrayHost'+str(g_m+1)+'(numberOfThreads * (('+dims[g_m]+'-1)/64+1)*64) )')
        DO('i1','1','numberOfThreads+1')
        DO('i2','1',dims[g_m]+'+1')
        code('reductionArrayHost'+str(g_m+1)+'((i1 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + i2) = 0')
        ENDDO()
        ENDDO()

    code('')
    code('')
    code('call date_and_time(values=timeArrayEnd)')
    code('endTimeHost = 1.00000 * timeArrayEnd(8) + &')
    code('& 1000 * timeArrayEnd(7)  + &')
    code('& 60000 * timeArrayEnd(6) + &')
    code('& 3600000 * timeArrayEnd(5)')
    code('')
    code('accumulatorHostTime = endTimeHost - startTimeHost')
    code('loopTimeHost'+name+' = loopTimeHost'+name+' + accumulatorHostTime')
    code('')
    code('call date_and_time(values=timeArrayStart)')
    code('startTimeKernel = 1.00000 * timeArrayStart(8) + &')
    code('& 1000 * timeArrayStart(7) + &')
    code('& 60000 * timeArrayStart(6) + &')
    code('& 3600000 * timeArrayStart(5)')
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
      line = ''
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and optflags[g_m]==1:
          line = line + ', opDat'+str(g_m+1)+'OptPtr'
      code('!$OMP PARALLEL DO private (threadID, blockID, nelem, offset_b'+line+')')
      DO('i2','0','nblocks')
      code('threadID = omp_get_thread_num()')
      code('blockID = blkmap_'+name+'(i2+blockOffset+1)')
      code('nelem = nelems_'+name+'(blockID+1)')
      code('offset_b = offset_'+name+'(blockID+1)')
      DO('n','0','nelem')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and optflags[g_m]==1:
          IF('opArg'+str(g_m+1)+'%opt == 1')
          if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
            code('opDat'+str(g_m+1)+'OptPtr => opDat'+str(invinds[inds[g_m]-1]+1)+'Local(1 + opDat'+str(invinds[inds[g_m]-1]+1)+'Map(1 + (n+offset_b) * opDat'+str(invinds[inds[g_m]-1]+1)+'MapDim + '+str(int(idxs[g_m])-1)+') * ('+dims[g_m]+'):)')
          ELSE()
          code('opDat'+str(g_m+1)+'OptPtr => opDat'+str(invinds[inds[g_m]-1]+1)+'Local(1:)')
          ENDIF()
      comm('kernel call')
      line = 'CALL '+name+'_cpu( &'
      indent = '\n'+' '*depth
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
            line = line + indent + '& opDat'+str(g_m+1)+'Local(1 + (n+offset_b) * ('+dims[g_m]+') : (n+offset_b) * ('+dims[g_m]+') + '+dims[g_m]+')'
          else:
            line = line + indent + '& opDat'+str(g_m+1)+'Local(1 + (n+offset_b))'
        if maps[g_m] == OP_MAP and optflags[g_m]==0:
          if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
            line = line +indent + '& opDat'+str(invinds[inds[g_m]-1]+1)+'Local(1 + opDat'+str(invinds[inds[g_m]-1]+1)+'Map(1 + (n+offset_b) * opDat'+str(invinds[inds[g_m]-1]+1)+'MapDim + '+str(int(idxs[g_m])-1)+') * ('+dims[g_m]+') : opDat'+str(invinds[inds[g_m]-1]+1)+'Map(1 + (n+offset_b) * opDat'+str(invinds[inds[g_m]-1]+1)+'MapDim + '+str(int(idxs[g_m])-1)+') * ('+dims[g_m]+') + '+dims[g_m]+')'
          else:
            line = line +indent + '& opDat'+str(invinds[inds[g_m]-1]+1)+'Local(1 + opDat'+str(invinds[inds[g_m]-1]+1)+'Map(1 + (n+offset_b) * opDat'+str(invinds[inds[g_m]-1]+1)+'MapDim + '+str(int(idxs[g_m])-1)+'))'
        elif maps[g_m] == OP_MAP and optflags[g_m]==1:
          if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
            line = line +indent + '& opDat'+str(g_m+1)+'OptPtr(1:'+dims[g_m]+')'
          else:
            line = line +indent + '& opDat'+str(g_m+1)+'OptPtr(1)'
        if maps[g_m] == OP_GBL:
          if accs[g_m] == OP_INC:
            code('& reductionArrayHost'+str(g_m+1)+'(threadID * (('+dims[g_m]+'-1)/64+1)*64 + 1), &')
          else:
            if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
              line = line + indent +'& opDat'+str(g_m+1)+'Local(1:'+dims[g_m]+')'
            else:
              line = line + indent +'& opDat'+str(g_m+1)+'Local(1)'
        if g_m < nargs-1:
          line = line +', &'
        else:
           line = line +' &'
      depth = depth - 2
      code(line + indent + '& )')
      depth = depth + 2
      ENDDO()
      ENDDO()
      code('!$OMP END PARALLEL DO')
      code('blockOffset = blockOffset + nblocks')
      ENDDO()
    else:
      code('!$OMP PARALLEL DO private (sliceStart,sliceEnd,i1,threadID)')
      DO('i1','0','numberOfThreads')
      code('sliceStart = opSetCore%size * i1 / numberOfThreads')
      code('sliceEnd = opSetCore%size * (i1 + 1) / numberOfThreads')
      code('threadID = omp_get_thread_num()')
      comm('kernel call')
      DO('n','sliceStart', 'sliceEnd')
      line = 'CALL '+name+'_cpu( &'
      indent = '\n'+' '*depth
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
            line = line + indent + '& opDat'+str(g_m+1)+'Local(1 + n * ('+dims[g_m]+') : n * ('+dims[g_m]+') + '+dims[g_m]+')'
          else:
            line = line + indent + '& opDat'+str(g_m+1)+'Local(1 + n)'
        if maps[g_m] == OP_GBL:
          if accs[g_m] == OP_INC:
            line = line + indent + '& reductionArrayHost'+str(g_m+1)+'(threadID * (('+dims[g_m]+'-1)/64+1)*64 + 1)'
          else:
            if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
              line = line + indent +'& opDat'+str(g_m+1)+'Local(1:'+dims[g_m]+')'
            else:
              line = line + indent +'& opDat'+str(g_m+1)+'Local(1)'
        if g_m < nargs-1:
          line = line +', &'
        else:
           line = line +' &'
      depth = depth - 2
      code(line + indent + '& )')
      depth = depth + 2
      ENDDO()
      ENDDO()
      code('!$OMP END PARALLEL DO')


    IF('(n_upper .EQ. 0) .OR. (n_upper .EQ. opSetCore%core_size)')
    code('CALL op_mpi_wait_all(numberOfOpDats,opArgArray)')
    ENDIF()
    code('')


    code('')
    code('call date_and_time(values=timeArrayEnd)')
    code('endTimeKernel = 1.00000 * timeArrayEnd(8) + &')
    code('& 1000 * timeArrayEnd(7) + &')
    code('& 60000 * timeArrayEnd(6) + &')
    code('& 3600000 * timeArrayEnd(5)')
    code('')
    code('accumulatorKernelTime = endTimeKernel - startTimeKernel')
    code('loopTimeKernel'+name+' = loopTimeKernel'+name+' + accumulatorKernelTime')
    code('')
    code('call date_and_time(values=timeArrayStart)')
    code('startTimeHost = 1.00000 * timeArrayStart(8) + &')
    code('& 1000.00 * timeArrayStart(7) + &')
    code('& 60000 * timeArrayStart(6) + &')
    code('& 3600000 * timeArrayStart(5)')

    code('')
    code('CALL op_mpi_set_dirtybit(numberOfOpDats,opArgArray)')
    code('')

    #reductions
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and (accs[g_m] == OP_INC or accs[g_m] == OP_MIN or accs[g_m] == OP_MAX):
        DO('i1','1','numberOfThreads+1')
        if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
          DO('i2','1',dims[g_m]+'+1')
          code('opDat'+str(g_m+1)+'Local(i2) = opDat'+str(g_m+1)+'Local(i2) + reductionArrayHost'+str(g_m+1)+'((i1 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + i2)')
          ENDDO()
        else:
          code('opDat'+str(g_m+1)+'Local = opDat'+str(g_m+1)+'Local + reductionArrayHost'+str(g_m+1)+'((i1 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + 1)')
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

    code('call date_and_time(values=timeArrayEnd)')
    code('endTimeHost = 1.00000 * timeArrayEnd(8) + &')
    code('1000 * timeArrayEnd(7) + &')
    code('60000 * timeArrayEnd(6) + &')
    code('3600000 * timeArrayEnd(5)')
    code('')
    code('accumulatorHostTime = endTimeHost - startTimeHost')
    code('loopTimeHost'+name+' = loopTimeHost'+name+' + accumulatorHostTime')
    code('')
    code('returnSetKernelTiming = setKernelTime('+str(nk)+' , userSubroutine//C_NULL_CHAR, &')

    if ninds > 0:
      code('& accumulatorKernelTime / 1000.00,actualPlan_'+name+'%transfer,actualPlan_'+name+'%transfer2)')
    else:
      code('& accumulatorKernelTime / 1000.00,0.00000,0.00000)')

    depth = depth - 2
    code('END SUBROUTINE')

    code('END MODULE')
##########################################################################
#  output individual kernel file
##########################################################################
    if hydra:
      name = 'kernels/'+kernels[nk]['master_file']+'/'+name
      fid = open(name+'_kernel.CUF','w')
    else:
      fid = open(name+'_kernel.CUF','w')
    date = datetime.datetime.now()
    fid.write('!\n! auto-generated by op2.py on '+date.strftime("%Y-%m-%d %H:%M")+'\n!\n\n')
    fid.write(file_text)
    fid.close()

##########################################################################
#  Assemble Hydra master file
##########################################################################
def op2_gen_cuda_hydra():
  global dims, idxs, typs, indtyps, inddims
  global file_format, cont, comment
  global FORTRAN, CPP, g_m, file_text, depth, header_text, body_text

  file_text = ''
  code('MODULE HYDRA_CUDA_MODULE')
  code('USE OP2_FORTRAN_DECLARATIONS')
  code('USE OP2_FORTRAN_RT_SUPPORT')
  code('USE ISO_C_BINDING')
  code('USE CUDAFOR')
  code('USE CUDACONFIGURATIONPARAMS')
  code('')
  code('')
  comm('Constant declarations')
  code('#include "hydra_constants.inc"')
  code('')
  comm('Loop-specific global variables')
  file_text += header_text

  code('')
  code('CONTAINS')
  code('')
  code('#include "hydra_constants_set.inc"')
  code('#include "flux_low_gpufun.inc"')
  code('#include "bcs_kernels_gpufun.inc"')
  code('#include "update_kernels_gpufun.inc"')

  file_text += body_text
  code('END MODULE')
  fid = open('hydra_kernels.CUF','w')
  date = datetime.datetime.now()
  fid.write('!\n! auto-generated by op2.py on '+date.strftime("%Y-%m-%d %H:%M")+'\n!\n\n')
  fid.write(file_text)
  fid.close()
