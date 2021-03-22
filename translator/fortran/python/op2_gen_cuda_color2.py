##########################################################################
#
# CUDA code generator
#
# This routine is called by op2_fortran which parses the input files
#
# It produces a file xxx_kernel.CUF for each kernel,
# plus a master kernel file
#
##########################################################################

import re
import datetime
import os
import sys
import util
import pprint

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

arg_parse=util.arg_parse
replace_consts=util.replace_consts
replace_npdes=util.replace_npdes
get_stride_string=util.get_stride_string
replace_soa = util.replace_soa
find_function_calls=util.find_function_calls

def op2_gen_cuda_color2(master, date, consts, kernels, hydra, bookleaf):

#  global util.funlist, util.const_list
  global dims, idxs, typs, indtyps, inddims
  global file_format, cont, comment
  global FORTRAN, CPP, g_m, file_text, depth, header_text, body_text

  util.funlist = []
  util.const_list = []

  header_text = ''
  body_text = ''

  OP_ID   = 1;  OP_GBL   = 2;  OP_MAP = 3;

  OP_READ = 1;  OP_WRITE = 2;  OP_RW  = 3;
  OP_INC  = 4;  OP_MAX   = 5;  OP_MIN = 6;

  accsstring = ['OP_READ','OP_WRITE','OP_RW','OP_INC','OP_MAX','OP_MIN' ]

  any_soa = 0
  for nk in range (0,len(kernels)):
    any_soa = any_soa or sum(kernels[nk]['soaflags'])
  hybrid = 0
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
    set_name = kernels[nk]['set']
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
    reduct_mdim = 0
    reduct_1dim = 0
    for i in range(0,nargs):
      if maps[i] == OP_GBL and (accs[i] == OP_INC or accs[i] == OP_MAX or accs[i] == OP_MIN):
        j = i
        if (not dims[i].isdigit()) or int(dims[i])>1:
          reduct_mdim = 1
        else:
          reduct_1dim = 1
      if maps[i] == OP_GBL and accs[i] == OP_WRITE:
        j = i
    reduct = reduct_1dim or reduct_mdim

    #npdes->DNPDE
    for g_m in range(0,nargs):
      if (not dims[g_m].isdigit()) and maps[g_m] == OP_MAP and accs[g_m] == OP_INC:
        if 'npdes' in dims[g_m]:
          dims[g_m] = dims[g_m].replace('npdes','DNPDE')
    for g_m in range(0,ninds):
      if (not inddims[g_m].isdigit()) and indaccs[g_m] == OP_INC:
        if 'npdes' in dims[g_m]:
          inddims[g_m] = inddims[g_m].replace('npdes','DNPDE')

#atomics option
    ind_rw=0
    ind_inc=0
    atomics=0
    for i in range(0,nargs):
      if maps[i] == OP_MAP and accs[i] == OP_RW:
        ind_rw=1
      if maps[i] == OP_MAP and accs[i] == OP_INC:
        ind_inc=1
    if not ind_rw and ind_inc:
      atomics = 1
#    atomics=0


#    for g_m in range(0,nargs):
#      if dims[g_m] == 'NPDE':
#        dims[g_m] = '6'

#    if ('GRADL_EDGECON' in name):
#    for g_m in range(0,nargs):
#      if 'NPDE' in dims[g_m]:
#        dims[g_m] = dims[g_m].replace('NPDE','6')
#        try:
#          newdim = str(eval(dims[g_m]))
#          dims[g_m]  = newdim
#        except NameError as inst:
#          dims[g_m]
#          #do nothing

    atomic_reduction = 0
    if 'MP_FOURIER_CDCH' in name or 'intclineprim' in name or 'accumintegrals' in name:
      atomic_reduction = 1
    unknown_reduction_size = 0
    unknown_size_red_atomic = [0]*nargs
    needDimList = []
    for g_m in range(0,nargs):
      if (not dims[g_m].isdigit()):
        found=0
        if found==0:
          needDimList = needDimList + [g_m]
          if maps[g_m] == OP_GBL and (accs[g_m] == OP_INC or accs[g_m] == OP_MAX or accs[g_m] == OP_MIN):
            if atomic_reduction == 1:
              unknown_reduction_size = 2
              unknown_size_red_atomic[g_m] = 1
            else:
              unknown_reduction_size = 1
              soaflags[g_m] = 1

    for idx in needDimList:
      dims[idx] = 'opDat'+str(idx+1)+'Dim'
    for i in range(0,nargs):
      if maps[i]==OP_MAP:
        dims[i] = dims[invinds[inds[i]-1]]

    stage_flags = [0]*nargs
    FORTRAN = 1;
    CPP     = 0;
    g_m = 0;
    file_text = ''
    depth = 0
    permute = 0

##########################################################################
#  Generate Header
##########################################################################

    if hydra:
      code('MODULE '+kernels[nk]['master_file']+'_'+kernels[nk]['mod_file'][9:]+'_module_MODULE')
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
    else:
      code('MODULE '+name.upper()+'_MODULE')
      code('USE OP2_CONSTANTS')
    if bookleaf:
      code('USE kinds_mod,    ONLY: ink,rlk')
      code('USE parameters_mod,ONLY: LI')
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

    #strides for SoA
    if any_soa:
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
            k = k + [mapnames[g_m]]
            code('INTEGER(kind=4), CONSTANT :: opDat'+str(invinds[inds[g_m]-1]+1)+'_stride_OP2CONSTANT')
            code('INTEGER(kind=4) :: opDat'+str(invinds[inds[g_m]-1]+1)+'_stride_OP2HOST')
      dir_soa = -1
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID and ((not dims[g_m].isdigit()) or int(dims[g_m]) > 1):
          code('INTEGER(kind=4), CONSTANT :: direct_stride_OP2CONSTANT')
          code('INTEGER(kind=4) :: direct_stride_OP2HOST')
          dir_soa = g_m
          break

    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL:
        if (accs[g_m]== OP_INC or accs[g_m]== OP_MIN or accs[g_m]== OP_MAX):
          code(typs[g_m]+', DIMENSION(:), DEVICE, ALLOCATABLE :: reductionArrayDevice'+str(g_m+1)+name)
        if ((accs[g_m]==OP_READ and ((not dims[g_m].isdigit()) or int(dims[g_m]) > 1)) or accs[g_m]==OP_WRITE):
          code(typs[g_m]+', DIMENSION(:), DEVICE, ALLOCATABLE :: opGblDat'+str(g_m+1)+'Device'+name)


    code('')

    if ninds > 0:
      code('TYPE ( c_ptr )  :: planRet_'+name)
    code('')
    if any_soa > -1:
      code('#define OP2_SOA(var,dim,stride) var((dim-1)*stride+1)')
    code('')
    code('CONTAINS')
    code('')

##########################################################################
#  Reduction kernel function - if an OP_GBL exists
##########################################################################
    if reduct_1dim or (unknown_reduction_size==1):
      comm('Reduction cuda kernel'); depth = depth +2;
      code('attributes (device) SUBROUTINE ReductionFloat8(sharedDouble8, reductionResult,inputValue,reductionOperation)')
      code('REAL(kind=8), DIMENSION(:), DEVICE :: reductionResult')
      code('REAL(kind=8) :: inputValue')
      code('INTEGER(kind=4), VALUE :: reductionOperation')
      code('REAL(kind=8), DIMENSION(0:*) :: sharedDouble8')
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

      code('attributes (device) SUBROUTINE ReductionInt4(sharedInt4, reductionResult,inputValue,reductionOperation)')
      code('INTEGER(kind=4), DIMENSION(:), DEVICE :: reductionResult')
      code('INTEGER(kind=4) :: inputValue')
      code('INTEGER(kind=4), VALUE :: reductionOperation')
      code('INTEGER(kind=4), DIMENSION(0:*) :: sharedInt4')
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
      code('attributes (device) SUBROUTINE ReductionFloat8Mdim(sharedDouble8, reductionResult,inputValue,reductionOperation,dim)')
      code('REAL(kind=8), DIMENSION(:), DEVICE :: reductionResult')
      code('REAL(kind=8), DIMENSION(:) :: inputValue')
      code('INTEGER(kind=4), VALUE :: reductionOperation')
      code('INTEGER(kind=4), VALUE :: dim')
      code('REAL(kind=8), DIMENSION(0:*) :: sharedDouble8')
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
      code('SELECT CASE(reductionOperation)')
      code('CASE (0)')
      DO('i2','0','dim')
      code('sharedDouble8(threadID*dim + i2) = sharedDouble8(threadID*dim + i2) + sharedDouble8((threadID + i1)*dim + i2)')
      ENDDO()
      code('CASE (1)')
      DO('i2','0','dim')
#      IF('sharedDouble8(threadID*dim + i2).GT.sharedDouble8((threadID + i1)*dim + i2)')
      code('sharedDouble8(threadID*dim + i2) = MIN(sharedDouble8(threadID*dim + i2), sharedDouble8((threadID + i1)*dim + i2))')
      #ENDIF()
      ENDDO()
      code('CASE (2)')
      DO('i2','0','dim')
      code('sharedDouble8(threadID*dim + i2) = MAX(sharedDouble8(threadID*dim + i2), sharedDouble8((threadID + i1)*dim + i2))')
      ENDDO()
      code('END SELECT')
      ENDIF()
      code('i1 = ishft(i1,-1)')
      ENDDO()

      code('CALL syncthreads()')

      IF('threadID .EQ. 0')
      code('SELECT CASE(reductionOperation)')
      code('CASE (0)')
      code('reductionResult(1:dim) = reductionResult(1:dim) + sharedDouble8(0:dim-1)')
      code('CASE (1)')
      DO('i2','0','dim')
      code('reductionResult(1+i2) = MIN(reductionResult(1+i2) , sharedDouble8(i2))')
      ENDDO()
      code('CASE (2)')
      DO('i2','0','dim')
      code('reductionResult(1+i2) = MAX(reductionResult(1+i2) , sharedDouble8(i2))')
      ENDDO()
      code('END SELECT')
      ENDIF()

      code('CALL syncthreads()')
      code('END SUBROUTINE')
      code('')



##########################################################################
#  Inline user kernel function
##########################################################################
    using_consts = 0
    if hydra:
      code('')
      comm(name + ' user functions (CPU and GPU)')
      code('')
      #text = text.replace('module','!module')
      text = text.replace('contains','!contains')
      text = text.replace('end !module','!end module')
      text = text.replace('recursive subroutine','subroutine')
      if hybrid == 1:
        text = text.replace('subroutine '+name, 'attributes(host) subroutine '+name)
        file_text += text
      code('')
      code('')
      #remove all comments
      util.const_list = []
      text = re.sub('!.*\n','\n',text)
      text = replace_consts(text)
      using_consts = text.find('use HYDRA_CONST_MODULE')>=0
      text = text.replace('subroutine '+name, 'attributes(device) subroutine '+name+'_gpu',1)

      #find subroutine calls
      util.funlist = [name.lower()]
      util.funlist2 = []
      plus_kernels, text = find_function_calls(text,'attributes(device) ',name+'_gpu')
      funcs = util.replace_soa_subroutines(util.funlist2,0,soaflags,maps,accs,mapnames,1,hydra,bookleaf,unknown_size_red_atomic,[],atomics)
      text = ''
      for func in funcs:
          text = text + '\n' + func['function_text']
      for fun in util.funlist:
        regex = re.compile('\\b'+fun+'\\b',re.I)
        text = regex.sub(fun+'_gpu',text)
        text = text.replace('cop2rep','!cop2rep')

      #strip "use" statements
      i = re.search('\\buse\\b',text.lower())
      i_offset = 0
      while not (i is None):
        i_offset = i_offset+i.start()
        if not ('HYDRA_CONST_MODULE' in text[i_offset:i_offset+23]):
          text = text[0:i_offset]+'!'+text[i_offset:]
        i_offset = i_offset+4
        i = re.search('\\buse\\b',text[i_offset:].lower())


      file_text += text
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
      file_text += 'attributes (host) subroutine ' + name + '' + text[i+ 11 + len(name):j]+'\n\n'
      kern_text = 'attributes (device) subroutine ' + name + '_gpu' + text[i+ 11 + len(name):j]+'_gpu\n\n'
      for const in range(0,len(consts)):
        i = re.search('\\b'+consts[const]['name']+'\\b',kern_text)
        if i != None:
#          print 'Found ' + consts[const]['name']
          j = i.start()
          kern_text = kern_text[0:j+1] + re.sub('\\b'+consts[const]['name']+'\\b',consts[const]['name']+'_OP2',kern_text[j+1:])

      text = replace_soa(kern_text,nargs,soaflags,name,maps,accs,set_name,mapnames,1,hydra,bookleaf)
      if any_soa:
        text = re.sub('\\bDIMENSION\([A-Za-z0-9_]*\)','DIMENSION(*)',text)
      file_text += text

    else:
      depth -= 2
      code('attributes (host) &')
      code('#include "'+name+'.inc"')
      code('attributes (device) &')
      fid = open(name+'.inc2', 'r')
      text = fid.read()
#      text = replace_soa(text,nargs,soaflags,name,maps,accs,set_name,mapnames,1,hydra,bookleaf,[],atomics)
      #find subroutine calls
      util.funlist = [name.lower()]
      util.funlist2 = []
      plus_kernels, text = find_function_calls(text,'attributes(device) ',name+'_gpu')
      funcs = util.replace_soa_subroutines(util.funlist2,0,soaflags,maps,accs,mapnames,1,hydra,bookleaf,unknown_size_red_atomic,[],atomics)
      text = ''
      for func in funcs:
          text = text + '\n' + func['function_text']
      for fun in util.funlist:
        regex = re.compile('\\b'+fun+'\\b',re.I)
        text = regex.sub(fun+'_gpu',text)
      code(text)
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
    for g_m in range(0,ninds):
      if invinds[g_m] in needDimList:
        code('& opDat'+str(invinds[g_m]+1)+'Dim, &')
      code('& opDat'+str(invinds[g_m]+1)+'Device'+name+', &')
    if nmaps > 0:
      k = []
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
          k = k + [mapnames[g_m]]
          code('& opDat'+str(invinds[inds[g_m]-1]+1)+'Map, &')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        if g_m in needDimList:
          code('& opDat'+str(g_m+1)+'Dim, &')
        code('& opDat'+str(g_m+1)+'Device'+name+', &')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL:
        if g_m in needDimList:
          code('& opDat'+str(g_m+1)+'Dim, &')
        if (accs[g_m] == OP_INC or accs[g_m] == OP_MIN or accs[g_m] == OP_MAX) and unknown_size_red_atomic[g_m] != 1:
          if g_m in needDimList:
            code('& scratchDevice'+str(g_m+1)+',   &')
        elif accs[g_m] == OP_READ and dims[g_m].isdigit() and int(dims[g_m])==1:
          code('& opGblDat'+str(g_m+1)+'Device'+name+',   &')

    if ninds > 0: #indirect loop
      code('& start, end, &')
      if not atomics:
        code('& pcol_reord, &')
      code('& setSize)')
    else: #direct loop
      code('& setSize)')

    code('')
    if hydra and using_consts:
        code('use HYDRA_CONST_MODULE')
    code('IMPLICIT NONE')
    code('')

##########################################################################
#  Declare local variables
##########################################################################
    comm('local variables')
    if nopts>0:
      code('INTEGER(kind=4), VALUE :: optflags')
    for g_m in range(0,ninds):
      if invinds[g_m] in needDimList:
        code('INTEGER(kind=4), VALUE :: opDat'+str(invinds[g_m]+1)+'Dim')
      if indaccs[g_m]==OP_READ:
        code(typs[invinds[g_m]]+', DEVICE :: opDat'+str(invinds[g_m]+1)+'Device'+name+'(*)')
      else:
        code(typs[invinds[g_m]]+', DEVICE :: opDat'+str(invinds[g_m]+1)+'Device'+name+'(*)')
    if nmaps > 0:
      k = []
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
          k = k + [mapnames[g_m]]
          code('INTEGER(kind=4), DEVICE, INTENT(IN) :: opDat'+str(invinds[inds[g_m]-1]+1)+'Map(*)')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        if g_m in needDimList:
          code('INTEGER(kind=4), VALUE :: opDat'+str(g_m+1)+'Dim')
        if accs[g_m] == OP_READ:
          code(typs[g_m]+', DEVICE, INTENT(IN) :: opDat'+str(g_m+1)+'Device'+name+'(*)')
        else:
          code(typs[g_m]+', DEVICE :: opDat'+str(g_m+1)+'Device'+name+'(*)')
    code('')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL:
        if g_m in needDimList:
          code('INTEGER(kind=4), VALUE :: opDat'+str(g_m+1)+'Dim')
        if accs[g_m] == OP_INC or accs[g_m] == OP_MIN or accs[g_m] == OP_MAX:
          #and additionally we need registers to store contributions, depending on dim:
          if dims[g_m].isdigit() and int(dims[g_m]) == 1:
            code(typs[g_m]+' :: opGblDat'+str(g_m+1)+'Device'+name)
          elif unknown_size_red_atomic[g_m]==0:
            if g_m in needDimList:
              code(typs[g_m]+', DEVICE :: scratchDevice'+str(g_m+1)+'(*)')
            else:
              code(typs[g_m]+', DIMENSION(0:'+dims[g_m]+'-1) :: opGblDat'+str(g_m+1)+'Device'+name)
        else:
          #if it's not  a global reduction, and multidimensional then we pass in a device array
          if dims[g_m].isdigit() and int(dims[g_m]) == 1:
            if accs[g_m] == OP_READ: #if OP_READ and dim 1, we can pass in by value
              code(typs[g_m]+', VALUE :: opGblDat'+str(g_m+1)+'Device'+name)

    if nmaps > 0:
      k = []
      line = 'INTEGER(kind=4) '
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (not mapinds[g_m] in k):
          k = k + [mapinds[g_m]]
          line += 'map'+str(mapinds[g_m]+1)+'idx, '
      code(line[:-2])
    code('')

    if ninds > 0: #indirect loop
      code('INTEGER(kind=4), VALUE :: start, end')
      if not atomics:
        code('INTEGER(kind=4), DIMENSION(0:*), DEVICE :: pcol_reord')
      code('INTEGER(kind=4), VALUE :: setSize')
      code('')

      code('INTEGER(kind=4) :: i3')

    else: #direct loop
      code('INTEGER(kind=4), VALUE :: setSize')

    code('INTEGER(kind=4) :: i1')
    code('INTEGER(kind=4) :: i2')
    if unknown_reduction_size == 1:
      code('INTEGER(kind=4) :: thrIdx')

    if reduct:
        add_real = 0
        add_int = 0
        for g_m in range(0,nargs):
          if maps[g_m] == OP_GBL and accs[g_m] != OP_READ and unknown_size_red_atomic[g_m]==0:
            if 'real' in typs[g_m].lower():
              add_real = 1
            elif 'integer' in typs[g_m].lower():
              add_int = 1
        if add_real:
          code('REAL(kind=8), DIMENSION(0:*), SHARED :: redFloat8')
        if add_int:
          code('INTEGER(kind=4), DIMENSION(0:*), SHARED :: redInt4')


    for g_m in range(0,nargs):
      if stage_flags[g_m] == 1:
        if g_m in needDimList:
              print('Error, cannot statically determine dim of argument '+str(g_m+1)+' in kernel '+name)
              sys.exit(-1)
        code(typs[g_m]+', DIMENSION('+dims[g_m]+') :: opDat'+str(g_m+1)+'Staged')

    code('')
    if unknown_reduction_size == 1:
      code('thrIdx = threadIdx%x - 1 + (blockIdx%x - 1) * blockDim%x')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and unknown_size_red_atomic[g_m]==0:
        if accs[g_m] == OP_INC:
          if g_m in needDimList:
            DO('i1','0',dims[g_m]) 
            code('scratchDevice'+str(g_m+1)+'(thrIdx+1+i1*(blockDim%x*gridDim%x)) = 0')
            ENDDO()
          else:
            code('opGblDat'+str(g_m+1)+'Device'+name+' = 0')
        elif accs[g_m] == OP_MIN or accs[g_m] == OP_MAX:
          if dims[g_m].isdigit() and int(dims[g_m])==1:
            code('opGblDat'+str(g_m+1)+'Device'+name+' = reductionArrayDevice'+str(g_m+1)+name+'(blockIdx%x - 1 + 1)')
          else:
            if g_m in needDimList:
              DO('i1','0',dims[g_m]) 
              code('scratchDevice'+str(g_m+1)+'(thrIdx+1+i1*(blockDim%x*gridDim%x)) = reductionArrayDevice'+str(g_m+1)+name+'((blockIdx%x - 1)*('+dims[g_m]+') + 1+i1)')
              ENDDO()
            else:
              code('opGblDat'+str(g_m+1)+'Device'+name+' = reductionArrayDevice'+str(g_m+1)+name+'((blockIdx%x - 1)*('+dims[g_m]+') + 1:(blockIdx%x - 1)*('+dims[g_m]+') + ('+dims[g_m]+'))')


    code('')
    if ninds > 0:
      code('')
      code('i1 = threadIdx%x - 1 + (blockIdx%x - 1) * blockDim%x')
      IF('i1+start<end')

      if not atomics:
        code('i3 = pcol_reord(i1+start)')
      else:
        code('i3 = i1+start')
      k = []
      for g_m in range(0,nargs):
        #workaround for optional arguments: its map may not exist either
        if maps[g_m] == OP_MAP and ((not (optflags[g_m]*nargs+mapinds[g_m]) in k) and (not mapinds[g_m] in k)):
          k = k + [(optflags[g_m]*nargs+mapinds[g_m])]
          if optflags[g_m]==1:
            IF('BTEST(optflags,'+str(optidxs[g_m])+')')
            code('map'+str(mapinds[g_m]+1)+'idx = opDat'+str(invmapinds[inds[g_m]-1]+1)+'Map(1 + i3 + setSize * '+str(int(idxs[g_m])-1)+')')
            ENDIF()
          else:
            code('map'+str(mapinds[g_m]+1)+'idx = opDat'+str(invmapinds[inds[g_m]-1]+1)+'Map(1 + i3 + setSize * '+str(int(idxs[g_m])-1)+')')

      for g_m in range(0,nargs):
        if stage_flags[g_m] == 1:
          if optflags[g_m]==1:
            IF('BTEST(optflags,'+str(optidxs[g_m])+')')
          if maps[g_m] == OP_MAP:
            DO('i2','0', dims[g_m])
            code('opDat'+str(g_m+1)+'Staged(i2+1) = opDat'+str(invinds[inds[g_m]-1]+1)+'Device'+name+ ' &')
            code('  & (1 + i2 * '+get_stride_string(g_m,maps,mapnames,set_name)+' + map'+str(mapinds[g_m]+1)+'idx)')
            ENDDO()
          else:
            DO('i2','0', dims[g_m])
            code('opDat'+str(g_m+1)+'Staged(i2+1) = opDat'+str(g_m+1)+'Device'+name+ ' &')
            code('  & (1 + i2 * direct_stride_OP2CONSTANT + i3)')
            ENDDO()
          if optflags[g_m]==1:
            ENDIF()


      code('')
      comm('kernel call')

    else:
      DO_STEP('i1','threadIdx%x - 1 + (blockIdx%x - 1) * blockDim%x','setSize','blockDim%x * gridDim%x')
      code('')
      comm('kernel call')
      code('')

      for g_m in range(0,nargs):
        if stage_flags[g_m] == 1:
          if optflags[g_m]==1:
            IF('BTEST(optflags,'+str(optidxs[g_m])+')')
          DO('i2','0', dims[g_m])
          code('opDat'+str(g_m+1)+'Staged(i2+1) = opDat'+str(g_m+1)+'Device'+name+ ' &')
          code('  & (1 + i2 * direct_stride_OP2CONSTANT + i1)')
          ENDDO()
          if optflags[g_m]==1:
            ENDIF()

##########################################################################
# user kernel call
##########################################################################
    if ninds > 0: #indirect kernel call
      line = '  CALL '+name+'_gpu( &'
      indent = '\n'+' '*depth
      for g_m in range(0,nargs):
        if stage_flags[g_m] == 1:
          line = line + indent + '& opDat'+str(g_m+1)+'Staged'
        elif soaflags[g_m] == 1 and maps[g_m] != OP_GBL:# and optflags[g_m]==0:
#          line = line +indent + '& opDat'+str(g_m+1)+'SoALocal'
          if maps[g_m] == OP_MAP:
            line = line +indent + '& opDat'+str(invinds[inds[g_m]-1]+1)+'Device'+name+ '(1 + map'+str(mapinds[g_m]+1)+'idx)'
          else:
            line = line +indent + '& opDat'+str(g_m+1)+'Device'+name+ '(1 + i3)'
        elif maps[g_m] == OP_ID:
          if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
            line = line + indent + '& opDat'+str(g_m+1)+'Device'+name+ \
            '(i3 * ('+dims[g_m]+') +1' + \
            ':i3 * ('+dims[g_m]+') + ('+dims[g_m]+'))'
          else:
            line = line + indent + '& opDat'+str(g_m+1)+'Device'+name+ \
            '(i3 * ('+dims[g_m]+') +1)'
        elif maps[g_m] == OP_MAP :# and optflags[g_m]==0:
          if (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
            line = line +indent + '& opDat'+str(invinds[inds[g_m]-1]+1)+'Device'+name+ \
            '(1 + map'+str(mapinds[g_m]+1)+'idx * ('+dims[g_m]+'):'+ \
            '     map'+str(mapinds[g_m]+1)+'idx * ('+dims[g_m]+') + '+dims[g_m]+')'
          else:
            line = line +indent + '& opDat'+str(invinds[inds[g_m]-1]+1)+'Device'+name+ \
            '(1 + map'+str(mapinds[g_m]+1)+'idx)'
        elif maps[g_m] == OP_GBL:
          if accs[g_m] == OP_WRITE and dims[g_m].isdigit() and int(dims[g_m]) == 1:
            line = line + indent +'& opGblDat'+str(g_m+1)+'Device'+name+'(1)'
          else:
            if (accs[g_m] == OP_MIN or accs[g_m] == OP_MAX or accs[g_m] == OP_INC) and g_m in needDimList:
              if unknown_size_red_atomic[g_m] == 0:
                line = line + indent +'& scratchDevice'+str(g_m+1)+'(thrIdx+1:)'
              else:
                line = line + indent +'& reductionArrayDevice'+str(g_m+1)+name
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

      for g_m in range(0,nargs):
        if stage_flags[g_m] == 1 and accs[g_m] != OP_READ:
          if optflags[g_m]==1:
            IF('BTEST(optflags,'+str(optidxs[g_m])+')')
          if maps[g_m] == OP_MAP:
            DO('i2','0', dims[g_m])
            code('opDat'+str(invinds[inds[g_m]-1]+1)+'Device'+name+ '(1 + i2 * '+get_stride_string(g_m,maps,mapnames,set_name)+' + map'+str(mapinds[g_m]+1)+'idx) = &')
            code('  & opDat'+str(g_m+1)+'Staged(i2+1)')
            ENDDO()
          else:
            DO('i2','0', dims[g_m])
            code('opDat'+str(g_m+1)+'Device'+name+ '(1 + i2 * direct_stride_OP2CONSTANT + i3) = &')
            code('  & opDat'+str(g_m+1)+'Staged(i2+1)')
            ENDDO()
          if optflags[g_m]==1:
            ENDIF()
      ENDIF() #if i2+start < end
    else: #direct kernel call
      line = '  CALL '+name+'_gpu( &'
      indent = '\n'+' '*depth
      for g_m in range(0,nargs):
        if stage_flags[g_m] == 1:
          line = line + indent + '& opDat'+str(g_m+1)+'Staged'
        elif soaflags[g_m] == 1 and maps[g_m] != OP_GBL and (maps[g_m] != OP_MAP or accs[g_m] != OP_INC):# and optflags[g_m]==0:
#          line = line +indent + '& opDat'+str(g_m+1)+'SoALocal'
           line = line +indent + '& opDat'+str(g_m+1)+'Device'+name+ '(1 + i1)'
        elif maps[g_m] == OP_GBL:
          if accs[g_m] == OP_WRITE and dims[g_m].isdigit() and int(dims[g_m]) == 1:
            line = line + indent +'& opGblDat'+str(g_m+1)+'Device'+name+'(1)'
          else:
            if (accs[g_m] == OP_MIN or accs[g_m] == OP_MAX or accs[g_m] == OP_INC) and g_m in needDimList:
              if unknown_size_red_atomic[g_m] == 0:
                line = line + indent +'& scratchDevice'+str(g_m+1)+'(thrIdx+1:)'
              else:
                line = line + indent +'& reductionArrayDevice'+str(g_m+1)+name
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
        if stage_flags[g_m] == 1 and accs[g_m] != OP_READ:
          if optflags[g_m]==1:
            IF('BTEST(optflags,'+str(optidxs[g_m])+')')
          DO('i2','0', dims[g_m])
          code('opDat'+str(g_m+1)+'Device'+name+ '(1 + i2 * direct_stride_OP2CONSTANT + i1) = &')
          code('  & opDat'+str(g_m+1)+'Staged(i2+1) ')
          ENDDO()
          if optflags[g_m]==1:
            ENDIF()
      ENDDO()

    #call cuda reduction for each OP_GBL
    code('')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and (accs[g_m] == OP_INC or accs[g_m] == OP_MIN or accs[g_m] == OP_MAX) and unknown_size_red_atomic[g_m] == 0:
        if accs[g_m] == OP_INC:
          op = '0'
        elif accs[g_m] == OP_MIN:
          op = '1'
        elif accs[g_m] == OP_MAX:
          op = '2'
        if 'real' in typs[g_m].lower():
          if dims[g_m].isdigit() and int(dims[g_m])==1:
            code('CALL ReductionFloat8(redFloat8, reductionArrayDevice'+str(g_m+1)+name+'(blockIdx%x - 1 + 1:),opGblDat'+str(g_m+1)+'Device'+name+','+op+')')
          else:
            if g_m in needDimList:
              code('do i1=0,'+dims[g_m]+'-1,1')
              code('  CALL ReductionFloat8(redFloat8, reductionArrayDevice'+str(g_m+1)+name+'((blockIdx%x - 1)*('+dims[g_m]+') + 1+i1:),scratchDevice'+str(g_m+1)+'(thrIdx+1+i1*(blockDim%x*gridDim%x)),'+op+')')
            else:
              code('do i1=0,'+dims[g_m]+'-1,8')
              code('i2 = MIN(i1+8,'+dims[g_m]+')')
              code('  CALL ReductionFloat8Mdim(redFloat8, reductionArrayDevice'+str(g_m+1)+name+'((blockIdx%x - 1)*('+dims[g_m]+') + 1+i1:),opGblDat'+str(g_m+1)+'Device'+name+'(i1:),'+op+',i2-i1)')
            code('end do')
        elif 'integer' in typs[g_m].lower():
          if dims[g_m].isdigit() and int(dims[g_m])==1:
            code('CALL ReductionInt4(redInt4, reductionArrayDevice'+str(g_m+1)+name+'(blockIdx%x - 1 + 1:),opGblDat'+str(g_m+1)+'Device'+name+','+op+')')
          else:
            if g_m in needDimList:
              code('do i1=0,'+dims[g_m]+'-1,1')
              code('  CALL ReductionInt4(redInt4, reductionArrayDevice'+str(g_m+1)+name+'((blockIdx%x - 1)*('+dims[g_m]+') + 1+i1:),scratchDevice'+str(g_m+1)+'(thrIdx+1+i1*(blockDim%x*gridDim%x)),'+op+')')
            else:
              code('do i1=0,'+dims[g_m]+'-1,8')
              code('i2 = MIN(i1+8,'+dims[g_m]+')')
              code('  CALL ReductionInt4Mdim(redInt4, reductionArrayDevice'+str(g_m+1)+name+'((blockIdx%x - 1)*('+dims[g_m]+') + 1+i1:),opGblDat'+str(g_m+1)+'Device'+name+'(i1:),'+op+',i2-i1)')
            code('end do')
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
    code('character(kind=c_char,len=*), INTENT(IN) :: userSubroutine')
    code('TYPE ( op_set ) , INTENT(IN) :: set')
    code('')

    for g_m in range(0,nargs):
      code('TYPE ( op_arg ) , INTENT(IN) :: opArg'+str(g_m+1))
    code('')
    IF('getHybridGPU().EQ.1')
    code('CALL '+name+'_host_gpu( userSubroutine, set, &');
    for g_m in range(0,nargs):
      if g_m == nargs-1:
        code('& opArg'+str(g_m+1)+' )')
      else:
        code('& opArg'+str(g_m+1)+', &')
    if hybrid == 1:
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
    if util.const_list:
      code('use HYDRA_CONST_MODULE')
    code('IMPLICIT NONE')
    code('character(kind=c_char,len=*), INTENT(IN) :: userSubroutine')
    code('TYPE ( op_set ) , INTENT(IN) :: set')
    code('')

    for g_m in range(0,nargs):
      code('TYPE ( op_arg ) , INTENT(IN) :: opArg'+str(g_m+1))
    code('')
    code('TYPE ( op_arg ) , DIMENSION('+str(nargs)+') :: opArgArray')
    code('INTEGER(kind=4) :: numberOfOpDats')
    code('INTEGER(kind=4) :: n_upper')
    code('INTEGER(kind=4), DIMENSION(1:8) :: timeArrayStart')
    code('INTEGER(kind=4), DIMENSION(1:8) :: timeArrayEnd')
    code('REAL(kind=8) :: startTime')
    code('REAL(kind=8) :: endTime')
    code('INTEGER(kind=4) :: returnSetKernelTiming')
    code('')
    code('')

    for g_m in range(0,ninds):
      code(typs[invinds[g_m]]+', DIMENSION(:), DEVICE, POINTER :: opDat'+str(invinds[g_m]+1)+'Device'+name)
      code('INTEGER(kind=4), DIMENSION(:), DEVICE, POINTER :: opMap'+str(invinds[g_m]+1)+'Device'+name)
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code(typs[g_m]+', DIMENSION(:), DEVICE, POINTER :: opDat'+str(g_m+1)+'Device'+name)
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
      code('INTEGER(kind=4), DIMENSION(1:'+str(nargs)+') :: accessDescriptorArray')
      code('INTEGER(kind=4), DIMENSION(1:'+str(nargs)+') :: indirectionDescriptorArray')
      code('')
      for g_m in range(0,ninds):
          code('INTEGER(kind=4) :: mappingArray'+str(invinds[g_m]+1)+'Size')
      code('')

      code('INTEGER(kind=4) :: numberOfIndirectOpDats')
      code('INTEGER(kind=4) :: blockOffset')
      if not atomics:
        code('INTEGER(kind=4), DIMENSION(:), DEVICE, POINTER :: pcol_reord')
        code('INTEGER(kind=4), DIMENSION(:), POINTER :: color2_offsets')
      else:
        code('INTEGER(kind=4) :: itstart, itend')
      code('INTEGER(kind=4) :: partitionSize')
      code('INTEGER(kind=4) :: blockSize')
      code('INTEGER(kind=4) :: i1')
      code('INTEGER(kind=4) :: i2')
      code('INTEGER(kind=4) :: i10')
      code('')

    else: #direct loop
      code('INTEGER(kind=4) :: blocksPerGrid')
      code('INTEGER(kind=4) :: threadsPerBlock')
      code('INTEGER(kind=4) :: dynamicSharedMemorySize')
      code('INTEGER(kind=4) :: threadSynchRet')
      code('INTEGER(kind=4) :: i1')
      code('INTEGER(kind=4) :: i2')
      code('INTEGER(kind=4) :: i10')
      code('INTEGER(kind=4) :: i20')
      code('REAL(kind=4) :: dataTransfer')
      code('')
    code('INTEGER(kind=4), SAVE :: calledTimes=0')
    code('INTEGER(kind=4) :: istat')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL:
        if accs[g_m] == OP_WRITE or (not dims[g_m].isdigit()) or int(dims[g_m]) > 1:
          code(typs[g_m]+', DIMENSION(:), POINTER :: opDat'+str(g_m+1)+'Host')
        else:
          code(typs[g_m]+', POINTER :: opDat'+str(g_m+1)+'Host')
          if accs[g_m] == OP_READ and dims[g_m].isdigit() and int(dims[g_m])==1:
            code(typs[g_m]+' :: opDat'+str(g_m+1)+'Host_tmp') #XLF workaround
        if (accs[g_m] == OP_INC or accs[g_m] == OP_MAX or accs[g_m] == OP_MIN):
          code(typs[g_m]+', DIMENSION(:), ALLOCATABLE :: reductionArrayHost'+str(g_m+1))
          if g_m in needDimList and unknown_size_red_atomic[g_m] == 0:
            code(typs[g_m]+', DIMENSION(:), DEVICE, POINTER :: scratchDevice'+str(g_m+1))
            code('INTEGER(kind=4) :: scratchDevice'+str(g_m+1)+'Size')
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
      print('ERROR: too many optional arguments to store flags in an integer')

    code('')
    code('numberOfOpDats = '+str(nargs))
    code('')

    for g_m in range(0,nargs):
      code('opArgArray('+str(g_m+1)+') = opArg'+str(g_m+1))
    code('')


    code('returnSetKernelTiming = setKernelTime('+str(nk)+' , userSubroutine//C_NULL_CHAR, &')
    code('& 0.0_8, 0.00000_4,0.00000_4, 0)')
    #code('print *,"'+name+'"')

    #managing constants
    if any_soa:
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
            k = k + [mapnames[g_m]]
            IF('(calledTimes.EQ.0).OR.(opDat'+str(invinds[inds[g_m]-1]+1)+'_stride_OP2HOST.NE.getSetSizeFromOpArg(opArg'+str(g_m+1)+'))')
            code('opDat'+str(invinds[inds[g_m]-1]+1)+'_stride_OP2HOST = getSetSizeFromOpArg(opArg'+str(g_m+1)+')')
            code('opDat'+str(invinds[inds[g_m]-1]+1)+'_stride_OP2CONSTANT = opDat'+str(invinds[inds[g_m]-1]+1)+'_stride_OP2HOST')
            ENDIF()
      if dir_soa!=-1:
          IF('(calledTimes.EQ.0).OR.(direct_stride_OP2HOST.NE.getSetSizeFromOpArg(opArg'+str(dir_soa+1)+'))')
          code('direct_stride_OP2HOST = getSetSizeFromOpArg(opArg'+str(dir_soa+1)+')')
          code('direct_stride_OP2CONSTANT = direct_stride_OP2HOST')
          ENDIF()

    #TODO: this is terrible
    # for const in util.const_list:
    #   code(const+'_OP2CONSTANT = '+const)

    code('call op_timers_core(startTime)')
    code('')
    #code('n_upper = op_mpi_halo_exchanges_cuda(set%setCPtr,numberOfOpDats,opArgArray)')
    code('n_upper = op_mpi_halo_exchanges_grouped(set%setCPtr,numberOfOpDats,opArgArray,2)')
    code('threadsPerBlock = getBlockSize(userSubroutine//C_NULL_CHAR,set%setPtr%size)')
    code('')

    if ninds > 0:
      for g_m in range(0,nargs):
        code('indirectionDescriptorArray('+str(g_m+1)+') = '+str(inds[g_m]-1))
      code('')
      code('numberOfIndirectOpDats = '+str(ninds))
      code('')
      code('partitionSize = getPartitionSize(userSubroutine//C_NULL_CHAR,set%setPtr%size)')
      #code('partitionSize = OP_PART_SIZE_ENV')
      code('')
      if not atomics:
        code('planRet_'+name+' = FortranPlanCaller( &')
        code('& userSubroutine//C_NULL_CHAR, &')
        code('& set%setCPtr, &')
        code('& partitionSize, &')
        code('& numberOfOpDats, &')
        code('& opArgArray, &')
        code('& numberOfIndirectOpDats, &')
        code('& indirectionDescriptorArray,4)')
        code('')
    else:
      code('')
      if unknown_reduction_size == 1:
        code('blocksPerGrid = 200')
      else:
        code('blocksPerGrid = 600')
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
          if accs[g_m] == OP_READ and dims[g_m].isdigit() and int(dims[g_m])==1:
            code('opDat'+str(g_m+1)+'Host_tmp = opDat'+str(g_m+1)+'Host') #XLF workaround
    code('')

    if ninds > 0 and not atomics:
      code('CALL c_f_pointer(planRet_'+name+',actualPlan_'+name+')')
      code('CALL c_f_pointer(actualPlan_'+name+'%color2_offsets,color2_offsets,(/actualPlan_'+name+'%ncolors+1/))')
      code('CALL c_f_pointer(actualPlan_'+name+'%col_reord,pcol_reord,(/set%setPtr%size+set%setPtr%exec_size/))')
      code('')

    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and ((accs[g_m]==OP_READ and ((not dims[g_m].isdigit()) or int(dims[g_m]) > 1)) or accs[g_m]==OP_WRITE):
        IF('.not. allocated(opGblDat'+str(g_m+1)+'Device'+name+')')
        code('allocate(opGblDat'+str(g_m+1)+'Device'+name+'(opArg'+str(g_m+1)+'%dim))')
        ENDIF()
        code('opGblDat'+str(g_m+1)+'Device'+name+'(1:opArg'+str(g_m+1)+'%dim) = opDat'+str(g_m+1)+'Host(1:opArg'+str(g_m+1)+'%dim)')
    if ninds>0 and reduct:
      code('blocksPerGrid=0')
      if not atomics:
        DO('i2','0','actualPlan_'+name+'%ncolors')
        code('blocksPerGrid = blocksPerGrid+(color2_offsets(i2+2)-color2_offsets(i2+1)-1)/threadsPerBlock+1')
        ENDDO()
      else:
        code('blocksPerGrid = (set%setPtr%size+set%setPtr%exec_size-1)/threadsPerBlock+1')

    for idx in needDimList:
      dims[idx] = 'opArg'+str(idx+1)+'%dim'
    
    #setup for reduction
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and (accs[g_m] == OP_INC or accs[g_m] == OP_MAX or accs[g_m] == OP_MIN):
        if unknown_size_red_atomic[g_m] == 0:
          code('reductionCardinality'+str(g_m+1)+' = blocksPerGrid * 1')
        else:
          code('reductionCardinality'+str(g_m+1)+' = 1')
        code('allocate( reductionArrayHost'+str(g_m+1)+'(reductionCardinality'+str(g_m+1)+'* ('+dims[g_m]+')) )')
        IF ('.not. allocated(reductionArrayDevice'+str(g_m+1)+name+')')
        code('allocate( reductionArrayDevice'+str(g_m+1)+name+'(reductionCardinality'+str(g_m+1)+'* ('+dims[g_m]+')) )')
        ENDIF()
        code('')
        DO('i10','0','reductionCardinality'+str(g_m+1)+'')
        if accs[g_m] == OP_INC:
          if dims[g_m].isdigit() and int(dims[g_m]) == 1:
            code('reductionArrayHost'+str(g_m+1)+'(i10+1) = 0.0')
          else:
            code('reductionArrayHost'+str(g_m+1)+'(i10 * ('+dims[g_m]+') + 1 : i10 * ('+dims[g_m]+') + ('+dims[g_m]+')) = 0.0')
        else:
          if dims[g_m].isdigit() and int(dims[g_m]) == 1:
            code('reductionArrayHost'+str(g_m+1)+'(i10+1) = opDat'+str(g_m+1)+'Host')
          else:
            code('reductionArrayHost'+str(g_m+1)+'(i10 * ('+dims[g_m]+') + 1 : i10 * ('+dims[g_m]+') + ('+dims[g_m]+')) = opDat'+str(g_m+1)+'Host')
        ENDDO()
        code('')
        code('reductionArrayDevice'+str(g_m+1)+name+' = reductionArrayHost'+str(g_m+1)+'')

    code('')
    if unknown_reduction_size==1:
      if ninds>0:
        code('blocksPerGrid = 0')
        if not atomics:
          DO('i2','0','actualPlan_'+name+'%ncolors')
          code('blocksPerGrid = MAX(blocksPerGrid,(color2_offsets(i2+2)-color2_offsets(i2+1)-1)/threadsPerBlock+1)')
          ENDDO()
        else:
          code('blocksPerGrid = MAX((set%setPtr%core_size-1)/threadsPerBlock+1,(set%setPtr%size+set%setPtr%exec_size-set%setPtr%core_size-1)/threadsPerBlock+1)')
      code('call prepareScratch(opArgArray,numberOfOpDats,blocksPerGrid*threadsPerBlock)')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL and (accs[g_m] == OP_INC or accs[g_m] == OP_MAX or accs[g_m] == OP_MIN) and (g_m in needDimList):
          #IF('opArg'+str(g_m+1)+'%dim.gt.20')
          #code('print *,"'+name+'",'+str(g_m+1)+',opArg'+str(g_m+1)+'%dim')
          #ENDIF()
          code('scratchDevice'+str(g_m+1)+'Size = opArg'+str(g_m+1)+'%dim*blocksPerGrid*threadsPerBlock')
          code('call c_f_pointer(opArgArray('+str(g_m+1)+')%data_d,scratchDevice'+str(g_m+1)+',(/scratchDevice'+str(g_m+1)+'Size/))')
      

    #indirect loop host stub call
    if ninds > 0:
      if not atomics:
        DO('i2','0','actualPlan_'+name+'%ncolors')
        IF('i2 .EQ. actualPlan_'+name+'%ncolors_core')
        #code('CALL op_mpi_wait_all_cuda(numberOfOpDats,opArgArray)')
        code('CALL op_mpi_wait_all_grouped(numberOfOpDats,opArgArray,2)')
        ENDIF()
        code('')
        code('blocksPerGrid = (color2_offsets(i2+2)-color2_offsets(i2+1)-1)/threadsPerBlock+1')
      else:
        DO('i2','0','2')
        IF('i2 .EQ. 1')
        code('itstart = set%setPtr%core_size')
        code('itend = n_upper')
        #code('CALL op_mpi_wait_all_cuda(numberOfOpDats,opArgArray)')
        code('CALL op_mpi_wait_all_grouped(numberOfOpDats,opArgArray,2)')
        ELSE()
        code('itstart = 0')
        code('itend = set%setPtr%core_size')
        ENDIF()
        code('')
        code('blocksPerGrid = (itend-itstart-1)/threadsPerBlock+1')

      code('dynamicSharedMemorySize = reductionSize(opArgArray,numberOfOpDats) * threadsPerBlock')
      code('')
      IF('blocksPerGrid.gt.0')
      #code('print *,"'+name+'", blocksPerGrid')
      code('CALL op_cuda_'+name+' <<<blocksPerGrid,threadsPerBlock,dynamicSharedMemorySize>>> (&')
      if nopts>0:
        code('& optflags, &')
      for g_m in range(0,ninds):
        if invinds[g_m] in needDimList:
            code('& opArg'+str(invinds[g_m]+1)+'%dim, &')
        code('& opDat'+str(invinds[g_m]+1)+'Device'+name+', &')
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
            k = k + [mapnames[g_m]]
            code('& opMap'+str(invinds[inds[g_m]-1]+1)+'Device'+name+', &')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          if g_m in needDimList:
            code('& opArg'+str(g_m+1)+'%dim, &')
          code('& opDat'+str(g_m+1)+'Device'+name+', &')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL:
          if g_m in needDimList:
            code('& opArg'+str(g_m+1)+'%dim, &')
          if (accs[g_m] == OP_INC or accs[g_m] == OP_MIN or accs[g_m] == OP_MAX):
            if g_m in needDimList and unknown_size_red_atomic[g_m] == 0:
              code('& scratchDevice'+str(g_m+1)+', &')
          if accs[g_m] == OP_READ and dims[g_m].isdigit() and int(dims[g_m])==1:
            code('& opDat'+str(g_m+1)+'Host_tmp, &') #XLF workaround
      if not atomics:
        code('& color2_offsets(i2+1), color2_offsets(i2+2), &')
        code('& pcol_reord,set%setPtr%size+set%setPtr%exec_size)')
      else:
        code('& itstart, itend, &')
        code('& set%setPtr%size+set%setPtr%exec_size)')
      ENDIF()
      ENDDO()
      code('')
    else: #direct loop host stub call
      code('CALL op_cuda_'+name+' <<<blocksPerGrid,threadsPerBlock,dynamicSharedMemorySize>>>( &')
      if nopts>0:
        code('& optflags, &')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          if g_m in needDimList:
            code('& opArg'+str(g_m+1)+'%dim, &')
          code('& opDat'+str(g_m+1)+'Device'+name+', &')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL:
          if g_m in needDimList:
            code('& opArg'+str(g_m+1)+'%dim, &')
          if (accs[g_m] == OP_INC or accs[g_m] == OP_MIN or accs[g_m] == OP_MAX):
            if g_m in needDimList and unknown_size_red_atomic[g_m] == 0:
              code('& scratchDevice'+str(g_m+1)+', &')
          if accs[g_m] == OP_READ and dims[g_m].isdigit() and int(dims[g_m])==1:
            code('& opDat'+str(g_m+1)+'Host_tmp, &') #XLF workaround
      code('set%setPtr%size)')
    code('')
    if not atomics:
      IF('(n_upper .EQ. 0) .OR. (n_upper .EQ. set%setPtr%core_size)')
      #code('CALL op_mpi_wait_all_cuda(numberOfOpDats,opArgArray)')
      code('CALL op_mpi_wait_all_grouped(numberOfOpDats,opArgArray,2)')
      ENDIF()
      code('')

    code('')
    code('CALL op_mpi_set_dirtybit_cuda(numberOfOpDats,opArgArray)')
    code('')

    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and accs[g_m] == OP_WRITE:
        code('opDat'+str(g_m+1)+'Host(1:opArg'+str(g_m+1)+'%dim) = opGblDat'+str(g_m+1)+'Device'+name+'(1:opArg'+str(g_m+1)+'%dim)')

    if reduct:
      #reductions
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL and (accs[g_m] == OP_INC or accs[g_m] == OP_MIN or accs[g_m] == OP_MAX):
          if optflags[g_m] == 1:
            IF('opArg'+str(g_m+1)+'%opt == 1')
          code('reductionArrayHost'+str(g_m+1)+' = reductionArrayDevice'+str(g_m+1)+name+'')
          code('')
          DO('i10','0','reductionCardinality'+str(g_m+1)+'')
          if accs[g_m] == OP_INC:
            if dims[g_m].isdigit() and int(dims[g_m]) == 1:
              code('opDat'+str(g_m+1)+'Host = opDat'+str(g_m+1)+'Host + reductionArrayHost'+str(g_m+1)+'(i10+1)')
            else:
              code('opDat'+str(g_m+1)+'Host(1:'+dims[g_m]+') = opDat'+str(g_m+1)+'Host(1:'+dims[g_m]+') + reductionArrayHost'+str(g_m+1)+'(i10 * ('+dims[g_m]+') + 1 : i10 * ('+dims[g_m]+') + ('+dims[g_m]+'))')
          elif accs[g_m] == OP_MIN:
            if dims[g_m].isdigit() and int(dims[g_m]) == 1:
              code('opDat'+str(g_m+1)+'Host = MIN(opDat'+str(g_m+1)+'Host , reductionArrayHost'+str(g_m+1)+'(i10+1))')
            else:
              code('opDat'+str(g_m+1)+'Host(1:'+dims[g_m]+') = MIN(opDat'+str(g_m+1)+'Host(1:'+dims[g_m]+') , reductionArrayHost'+str(g_m+1)+'(i10 * ('+dims[g_m]+') + 1 : i10 * ('+dims[g_m]+') + ('+dims[g_m]+')))')
          elif accs[g_m] == OP_MAX:
            if dims[g_m].isdigit() and int(dims[g_m]) == 1:
              code('opDat'+str(g_m+1)+'Host = MAX(opDat'+str(g_m+1)+'Host , reductionArrayHost'+str(g_m+1)+'(i10+1))')
            else:
              code('opDat'+str(g_m+1)+'Host(1:'+dims[g_m]+') = MAX(opDat'+str(g_m+1)+'Host(1:'+dims[g_m]+') , reductionArrayHost'+str(g_m+1)+'(i10 * ('+dims[g_m]+') + 1 : i10 * ('+dims[g_m]+') + ('+dims[g_m]+')))')
          ENDDO()
          code('')
          if optflags[g_m] == 1:
            ENDIF()
          code('deallocate( reductionArrayHost'+str(g_m+1)+' )')
#          code('deallocate( reductionArrayDevice'+str(g_m+1)+' )')
        if maps[g_m] == OP_GBL and (accs[g_m] == OP_INC or accs[g_m] == OP_MIN or accs[g_m] == OP_MAX or accs[g_m] == OP_WRITE):
          if optflags[g_m] == 1:
            IF('opArg'+str(g_m+1)+'%opt == 1')
          if typs[g_m] == 'real(8)' or typs[g_m] == 'REAL(kind=8)' or typs[g_m] == 'real*8' or typs[g_m] == 'r8':
            code('CALL op_mpi_reduce_double(opArg'+str(g_m+1)+',opArg'+str(g_m+1)+'%data)')
          elif typs[g_m] == 'real(4)' or typs[g_m] == 'REAL(kind=4)' or typs[g_m] == 'real*4' or typs[g_m] == 'r4':
            code('CALL op_mpi_reduce_float(opArg'+str(g_m+1)+',opArg'+str(g_m+1)+'%data)')
          elif typs[g_m] == 'integer(4)' or typs[g_m] == 'INTEGER(kind=4)' or typs[g_m] == 'integer*4' or typs[g_m] == 'i4':
            code('CALL op_mpi_reduce_int(opArg'+str(g_m+1)+',opArg'+str(g_m+1)+'%data)')
          elif typs[g_m] == 'logical' or typs[g_m] == 'logical*1':
            code('CALL op_mpi_reduce_bool(opArg'+str(g_m+1)+',opArg'+str(g_m+1)+'%data)')
          else:
            print('Error, reduction type '+typs[g_m]+' unrecognised')
          code('')
          if optflags[g_m] == 1:
            ENDIF()

    code('istat = cudaDeviceSynchronize()')
    IF('istat.ne.0')
    code('print *,cudaGetErrorString(istat)')
    ENDIF()
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
      if not atomics:
        code('& endTime-startTime, actualPlan_'+name+'%transfer,actualPlan_'+name+'%transfer2, 1)')
      else:
        code('& endTime-startTime, 0.00000_4, 0.00000_4, 1)')
    else:
      code('& endTime-startTime, dataTransfer, 0.00000_4, 1)')

    code('calledTimes = calledTimes + 1')
    depth = depth - 2
    code('END SUBROUTINE')
    code('')
    if hybrid == 1:
      code('')
      comm('Stub for CPU execution')
      code('')
##########################################################################
#  Generate OpenMP host stub
##########################################################################
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

############################################################################
###  Generate OpenMP host stub
############################################################################
      code('SUBROUTINE '+name+'_host_cpu( userSubroutine, set, &'); depth = depth + 2
      for g_m in range(0,nargs):
        if g_m == nargs-1:
          code('& opArg'+str(g_m+1)+' )')
        else:
          code('& opArg'+str(g_m+1)+', &')

      code('END SUBROUTINE')
    code('END MODULE')
##########################################################################
#  output individual kernel file
##########################################################################
    if hydra:
      name = 'kernels/'+kernels[nk]['master_file']+'_'+name
      fid = open(name+'_gpukernel.CUF','w')
    elif bookleaf:
      fid = open(prefixes[prefix_i]+name+'_gpukernel.CUF','w')
    else:
      fid = open(name+'_kernel.CUF','w')
    date = datetime.datetime.now()
    fid.write('!\n! auto-generated by op2.py\n!\n\n')
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
  fid.write('!\n! auto-generated by op2.py\n!\n\n')
  fid.write(file_text)
  fid.close()
