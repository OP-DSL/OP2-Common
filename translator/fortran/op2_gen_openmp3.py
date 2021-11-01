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


def op2_gen_openmp3(master, date, consts, kernels, hydra,bookleaf):

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
    dims  = list(kernels[nk]['dims'])
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
      if maps[i] == OP_MAP and accs[i] == OP_INC:
        j = i
    ind_inc = j >= 0

    j = -1
    for i in range(0,nargs):
      if maps[i] == OP_GBL and accs[i] != OP_READ:
        j = i
    reduct = j >= 0


    FORTRAN = 1;
    CPP     = 0;
    g_m = 0;
    file_text = ''
    depth = 0

    needDimList = []
    for g_m in range(0,nargs):
      if (not dims[g_m].isdigit()):# and not (dims[g_m] in ['NPDE','DNTQMU','DNFCROW','1*1']):
        needDimList = needDimList + [g_m]

    for idx in needDimList:
      dims[idx] = 'opDat'+str(idx+1)+'Dim'


##########################################################################
#  Generate Header
##########################################################################
    if hydra:
      code('MODULE '+kernels[nk]['master_file']+'_'+kernels[nk]['mod_file'][9:]+'_module_MODULE')
    else:
      code('MODULE '+name.upper()+'_MODULE')
    code('USE OP2_FORTRAN_DECLARATIONS')
    code('USE OP2_FORTRAN_RT_SUPPORT')
    code('USE ISO_C_BINDING')
    if hydra == 0 and bookleaf==0:
      code('USE OP2_CONSTANTS')
    if bookleaf:
      code('USE kinds_mod,    ONLY: ink,rlk')
      code('USE parameters_mod,ONLY: LI')

    code('')
    if bookleaf==0:
      code('#ifdef _OPENMP'); depth = depth + 2
    code('USE OMP_LIB'); depth = depth - 2
    if bookleaf==0:
      code('#endif')


    code('')

##########################################################################
#  Inline user kernel function
##########################################################################
    code('')
    code('CONTAINS')
    code('')
    if hydra:
      file_text += '!DEC$ ATTRIBUTES FORCEINLINE :: ' + name + '\n'
      modfile = kernels[nk]['mod_file'][9:]+'_module'
      filename = 'kernels/'+kernels[nk]['master_file']+'_'+name+'.inc'
      if not os.path.isfile(filename):
        files = [f for f in glob.glob('kernels/*'+name+'.inc')]
        if len(files)>0:
          filename = files[0]
        else:
          print('kernel for '+name+' not found')
      fid = open(filename, 'r')
      text = fid.read()
      fid.close()
      text = text.replace('recursive subroutine','subroutine')
      text = text.replace(' module',' !module')
      text = text.replace('contains','!contains')
      text = text.replace('end !module','!end module')

      #
      # substitute npdes with DNPDE
      #
#      using_npdes = 0
#      for g_m in range(0,nargs):
#        if var[g_m] == 'npdes':
#          using_npdes = 1
#      if using_npdes:
#        i = re.search('\\bnpdes\\b',text)
#        j = i.start()
#        i = re.search('\\bnpdes\\b',text[j:])
#        j = j + i.start()+5
#        i = re.search('\\bnpdes\\b',text[j:])
#        j = j + i.start()+5
#        text = text[1:j] + re.sub('\\bnpdes\\b','NPDE',text[j:])

      file_text += text
      file_text += '\n#undef MIN\n'
      file_text += '\n#undef MAX\n'
      #code(kernels[nk]['mod_file'])
    elif bookleaf:
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
      if maps[g_m] != OP_MAP:
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
      code(typs[invinds[g_m]]+' opDat'+str(invinds[g_m]+1)+'Local('+str(dims[invinds[g_m]])+',*)')
    for g_m in range(0,nargs):
      if maps[g_m] != OP_MAP:
        if g_m in needDimList:
          code('INTEGER(kind=4) opDat'+str(g_m+1)+'Dim')
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

##########################################################################
#  Generate OpenMP host stub
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
    code('REAL(kind=8) :: startTime')
    code('REAL(kind=8) :: endTime')
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
      code('REAL(kind=4) :: dataTransfer')

    code('')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and accs[g_m] != OP_READ:
        code(typs[g_m]+', DIMENSION(:), ALLOCATABLE :: reductionArrayHost'+str(g_m+1))

    code('')
    code('INTEGER(kind=4) :: i1,i2,n')

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
    #code('n_upper = op_mpi_halo_exchanges(set%setCPtr,numberOfOpDats,opArgArray)')
    code('n_upper = op_mpi_halo_exchanges_grouped(set%setCPtr,numberOfOpDats,opArgArray,1)')
    code('')

    if ninds > 0:
      if bookleaf==0:
        code_pre('#ifdef OP_PART_SIZE_1')
        code_pre('  partitionSize = OP_PART_SIZE_1')
        code_pre('#else')
      code_pre('  partitionSize = 0')
      if bookleaf==0:
        code_pre('#endif')

    code('')
    if bookleaf==0:
      code_pre('#ifdef _OPENMP')
    code_pre('  numberOfThreads = omp_get_max_threads()')
    if bookleaf==0:
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
      code('& indirectionDescriptorArray,2)')
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

    for idx in needDimList:
      dims[idx] = 'opArg'+str(idx+1)+'%dim'

    #reductions
    for g_m in range(0,nargs):
      if optflags[g_m] == 1:
        if maps[g_m] == OP_GBL and accs[g_m] != OP_READ:
          code('allocate( reductionArrayHost'+str(g_m+1)+'(numberOfThreads * (('+dims[g_m]+'-1)/64+1)*64) )')
          IF('opArg'+str(g_m+1)+'%opt == 1')
          DO('i1','1','numberOfThreads+1')
          DO('i2','1',dims[g_m]+'+1')
          if accs[g_m] == OP_INC:
            code('reductionArrayHost'+str(g_m+1)+'((i1 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + i2) = 0')
          else:
            code('reductionArrayHost'+str(g_m+1)+'((i1 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + i2) = opDat'+str(g_m+1)+'Local(i2)')
          ENDDO()
          ENDDO()
          ENDIF()
      else:
        if maps[g_m] == OP_GBL and accs[g_m] != OP_READ:
          code('allocate( reductionArrayHost'+str(g_m+1)+'(numberOfThreads * (('+dims[g_m]+'-1)/64+1)*64) )')
          DO('i1','1','numberOfThreads+1')
          DO('i2','1',dims[g_m]+'+1')
          if accs[g_m] == OP_INC:
            code('reductionArrayHost'+str(g_m+1)+'((i1 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + i2) = 0')
          else:
            code('reductionArrayHost'+str(g_m+1)+'((i1 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + i2) = opDat'+str(g_m+1)+'Local(i2)')
          ENDDO()
          ENDDO()

    code('')

    if ninds > 0: #indirect loop host stub call
      code('blockOffset = 0')
      code('')
      DO('i1','0','actualPlan_'+name+'%ncolors')

      IF('i1 .EQ. actualPlan_'+name+'%ncolors_core')
      #code('CALL op_mpi_wait_all(numberOfOpDats,opArgArray)')
      code('CALL op_mpi_wait_all_grouped(numberOfOpDats,opArgArray,1)')
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

      code('CALL op_wrap_'+name+'( &')
      for g_m in range(0,ninds):
        if invinds[g_m] in needDimList:
          code('& opArg'+str(invinds[g_m]+1)+'%dim, &')
        code('& opDat'+str(invinds[g_m]+1)+'Local, &')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          if g_m in needDimList:
            code('& opArg'+str(g_m+1)+'%dim, &')
          code('& opDat'+str(g_m+1)+'Local, &')
        elif maps[g_m] == OP_GBL:
          if g_m in needDimList:
            code('& opArg'+str(g_m+1)+'%dim, &')
          if accs[g_m] != OP_READ:
            code('& reductionArrayHost'+str(g_m+1)+'(threadID * (('+dims[g_m]+'-1)/64+1)*64 + 1), &')
          else:
            code('& opDat'+str(g_m+1)+'Local, &')
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
            k = k + [mapnames[g_m]]
            code('& opDat'+str(invinds[inds[g_m]-1]+1)+'Map, &')
            code('& opDat'+str(invinds[inds[g_m]-1]+1)+'MapDim, &')
      code('& offset_b, offset_b+nelem)')
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
      code('CALL op_wrap_'+name+'( &')
      for g_m in range(0,nargs):
        if g_m in needDimList:
          code('& opArg'+str(g_m+1)+'%dim, &')
        if maps[g_m] == OP_ID:
          code('& opDat'+str(g_m+1)+'Local, &')
        elif maps[g_m] == OP_GBL:
          if accs[g_m] != OP_READ:
            code('& reductionArrayHost'+str(g_m+1)+'(threadID * (('+dims[g_m]+'-1)/64+1)*64 + 1), &')
          else:
            code('& opDat'+str(g_m+1)+'Local, &')
      code('& sliceStart, sliceEnd)')
      ENDDO()
      code('!$OMP END PARALLEL DO')


    IF('(n_upper .EQ. 0) .OR. (n_upper .EQ. opSetCore%core_size)')
    #code('CALL op_mpi_wait_all(numberOfOpDats,opArgArray)')
    code('CALL op_mpi_wait_all_grouped(numberOfOpDats,opArgArray,1)')
    ENDIF()
    code('')
    code('CALL op_mpi_set_dirtybit(numberOfOpDats,opArgArray)')
    code('')

    #reductions
    for g_m in range(0,nargs):
      if optflags[g_m] == 1:
        if maps[g_m] == OP_GBL and (accs[g_m] == OP_INC or accs[g_m] == OP_MIN or accs[g_m] == OP_MAX):
          DO('i1','1','numberOfThreads+1')
          if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
            DO('i2','1',dims[g_m]+'+1')
            if accs[g_m] == OP_INC:
              IF('opArg'+str(g_m+1)+'%opt == 1')
              code('opDat'+str(g_m+1)+'Local(i2) = opDat'+str(g_m+1)+'Local(i2) + reductionArrayHost'+str(g_m+1)+'((i1 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + i2)')
              ENDIF()
            if accs[g_m] == OP_MIN:
              IF('opArg'+str(g_m+1)+'%opt == 1')
              code('opDat'+str(g_m+1)+'Local(i2) = MIN(opDat'+str(g_m+1)+'Local(i2) , reductionArrayHost'+str(g_m+1)+'((i1 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + i2))')
              ENDIF()
            if accs[g_m] == OP_MAX:
              IF('opArg'+str(g_m+1)+'%opt == 1')
              code('opDat'+str(g_m+1)+'Local(i2) = MAX(opDat'+str(g_m+1)+'Local(i2) , reductionArrayHost'+str(g_m+1)+'((i1 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + i2))')
              ENDIF()
            ENDDO()
          else:
            if accs[g_m] == OP_INC:
              IF('opArg'+str(g_m+1)+'%opt == 1')
              code('opDat'+str(g_m+1)+'Local = opDat'+str(g_m+1)+'Local + reductionArrayHost'+str(g_m+1)+'((i1 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + 1)')
              ENDIF()
            if accs[g_m] == OP_MIN:
              IF('opArg'+str(g_m+1)+'%opt == 1')
              code('opDat'+str(g_m+1)+'Local = MIN(opDat'+str(g_m+1)+'Local, reductionArrayHost'+str(g_m+1)+'((i1 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + 1))')
              ENDIF()
            if accs[g_m] == OP_MAX:
              IF('opArg'+str(g_m+1)+'%opt == 1')
              code('opDat'+str(g_m+1)+'Local = MAX(opDat'+str(g_m+1)+'Local, reductionArrayHost'+str(g_m+1)+'((i1 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + 1))')
              ENDIF()
          ENDDO()
          code('')
          code('deallocate( reductionArrayHost'+str(g_m+1)+' )')
          code('')
      else:
        if maps[g_m] == OP_GBL and (accs[g_m] == OP_INC or accs[g_m] == OP_MIN or accs[g_m] == OP_MAX):
          DO('i1','1','numberOfThreads+1')
          if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
            DO('i2','1',dims[g_m]+'+1')
            if accs[g_m] == OP_INC:
              code('opDat'+str(g_m+1)+'Local(i2) = opDat'+str(g_m+1)+'Local(i2) + reductionArrayHost'+str(g_m+1)+'((i1 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + i2)')
            if accs[g_m] == OP_MIN:
              code('opDat'+str(g_m+1)+'Local(i2) = MIN(opDat'+str(g_m+1)+'Local(i2) , reductionArrayHost'+str(g_m+1)+'((i1 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + i2))')
            if accs[g_m] == OP_MAX:
              code('opDat'+str(g_m+1)+'Local(i2) = MAX(opDat'+str(g_m+1)+'Local(i2) , reductionArrayHost'+str(g_m+1)+'((i1 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + i2))')
            ENDDO()
          else:
            if accs[g_m] == OP_INC:
              code('opDat'+str(g_m+1)+'Local = opDat'+str(g_m+1)+'Local + reductionArrayHost'+str(g_m+1)+'((i1 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + 1)')
            if accs[g_m] == OP_MIN:
              code('opDat'+str(g_m+1)+'Local = MIN(opDat'+str(g_m+1)+'Local, reductionArrayHost'+str(g_m+1)+'((i1 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + 1))')
            if accs[g_m] == OP_MAX:
              code('opDat'+str(g_m+1)+'Local = MAX(opDat'+str(g_m+1)+'Local, reductionArrayHost'+str(g_m+1)+'((i1 - 1) * (('+dims[g_m]+'-1)/64+1)*64 + 1))')
          ENDDO()
          code('')
          code('deallocate( reductionArrayHost'+str(g_m+1)+' )')
          code('')

      if maps[g_m] == OP_GBL and (accs[g_m] == OP_INC or accs[g_m] == OP_MIN or accs[g_m] == OP_MAX or accs[g_m] == OP_WRITE):
        if optflags[g_m] == 1:
          if typs[g_m] == 'real(8)' or typs[g_m] == 'REAL(kind=8)' or typs[g_m] == 'real*8' or typs[g_m] == 'r8':
            IF('opArg'+str(g_m+1)+'%opt == 1')
            code('CALL op_mpi_reduce_double(opArg'+str(g_m+1)+',opArg'+str(g_m+1)+'%data)')
            ENDIF()
          elif typs[g_m] == 'real(4)' or typs[g_m] == 'REAL(kind=4)' or typs[g_m] == 'real*4' or typs[g_m] == 'r4':
            IF('opArg'+str(g_m+1)+'%opt == 1')
            code('CALL op_mpi_reduce_float(opArg'+str(g_m+1)+',opArg'+str(g_m+1)+'%data)')
            ENDIF()
          elif typs[g_m] == 'integer(4)' or typs[g_m] == 'INTEGER(kind=4)' or typs[g_m] == 'integer*4' or typs[g_m] == 'i4':
            IF('opArg'+str(g_m+1)+'%opt == 1')
            code('CALL op_mpi_reduce_int(opArg'+str(g_m+1)+',opArg'+str(g_m+1)+'%data)')
            ENDIF()
          elif typs[g_m] == 'logical' or typs[g_m] == 'logical*1':
            IF('opArg'+str(g_m+1)+'%opt == 1')
            code('CALL op_mpi_reduce_bool(opArg'+str(g_m+1)+',opArg'+str(g_m+1)+'%data)')
            ENDIF()
          code('')
        else:
          if typs[g_m] == 'real(8)' or typs[g_m] == 'REAL(kind=8)' or typs[g_m] == 'real*8' or typs[g_m] == 'r8':
            code('CALL op_mpi_reduce_double(opArg'+str(g_m+1)+',opArg'+str(g_m+1)+'%data)')
          elif typs[g_m] == 'real(4)' or typs[g_m] == 'REAL(kind=4)' or typs[g_m] == 'real*4' or typs[g_m] == 'r4':
            code('CALL op_mpi_reduce_float(opArg'+str(g_m+1)+',opArg'+str(g_m+1)+'%data)')
          elif typs[g_m] == 'integer(4)' or typs[g_m] == 'INTEGER(kind=4)' or typs[g_m] == 'integer*4' or typs[g_m] == 'i4':
            code('CALL op_mpi_reduce_int(opArg'+str(g_m+1)+',opArg'+str(g_m+1)+'%data)')
          elif typs[g_m] == 'logical' or typs[g_m] == 'logical*1':
            code('CALL op_mpi_reduce_bool(opArg'+str(g_m+1)+',opArg'+str(g_m+1)+'%data)')
          code('')

    code('call op_timers_core(endTime)')
    code('')
    if ninds == 0:
      code('dataTransfer = 0.0')
      for g_m in range(0,nargs):
        if accs[g_m] == OP_READ or accs[g_m] == OP_WRITE:
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
      code('& endTime-startTime, dataTransfer, 0.00000_4, 1)')

    depth = depth - 2
    code('END SUBROUTINE')
    code('END MODULE')
    code('')

##########################################################################
#  output individual kernel file
##########################################################################
    if hydra:
      name = 'kernels/'+kernels[nk]['master_file']+'_'+name
      fid = open(name+'_ompkernel.F90','w')
    elif bookleaf:
      fid = open(prefixes[prefix_i]+name+'_kernel.f90','w')
    else:
      fid = open(name+'_kernel.F90','w')
    date = datetime.datetime.now()
    fid.write('!\n! auto-generated by op2.py\n!\n\n')
    fid.write(file_text.strip())
    fid.close()
