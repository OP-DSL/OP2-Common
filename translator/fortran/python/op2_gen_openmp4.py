##########################################################################
#
# OpenACC code generator
#
# This routine is called by op2_fortran which parses the input files
#
# It produces a file xxx_acckernel.F90 for each kernel,
# plus a master kernel file
#
##########################################################################

import re
import datetime
import os
import util 

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


arg_parse=util.arg_parse
replace_consts=util.replace_consts
replace_npdes=util.replace_npdes
get_stride_string=util.get_stride_string
replace_soa = util.replace_soa
find_function_calls=util.find_function_calls


def op2_gen_openmp4(master, date, consts, kernels, hydra,bookleaf):

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

    stage_soa = nopts
    stage_flags=[0]*nargs;
    host_exec = 0

    for g_m in range(0,nargs):
      if 'NPDE' in dims[g_m]:
        dims[g_m] = dims[g_m].replace('NPDE','6')
        try:
          newdim = str(eval(dims[g_m]))
          dims[g_m]  = newdim
        except NameError as inst:
          dims[g_m]
      if maps[g_m] == OP_GBL and accs[g_m] <> OP_READ and accs[g_m] <> OP_WRITE and not dims[g_m].isdigit():
        print 'WARNING: unknown dimension reduction argument '+str(g_m)+' in '+name+': host sequential execution'
        host_exec = 1
        for i in range(0,nargs):
          soaflags[i] = 0
#    for g_m in range(0,nargs):
#      if dims[g_m] == 'NPDE':
#        dims[g_m] = '6'

    if 'UPDATE_EXPK' in name:
      host_exec=1

    if host_exec:
      for i in range(0,nargs):
        soaflags[i] = 0


    is_soa = -1
    for i in range(0,nargs):
      if soaflags[i] == 1:
        is_soa = i
        break

    unknown_reduction_size = 0
    needDimList = []
    for g_m in range(0,nargs):
      if (not dims[g_m].isdigit()):
        found=0
        for string in ['NPDE','DNTQMU','DNFCROW','1*1']:
          if string in dims[g_m]:
            found=1
        if found==0:
          needDimList = needDimList + [g_m]
          if maps[g_m] == OP_GBL and (accs[g_m] == OP_INC or accs[g_m] == OP_MAX or accs[g_m] == OP_MIN):
            unknown_reduction_size = 1
            soaflags[g_m] = 1
            is_soa = 1

    for idx in needDimList:
      dims[idx] = 'opDat'+str(idx+1)+'Dim'


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
    else:
      code('MODULE '+name.upper()+'_MODULE')
      code('USE OP2_CONSTANTS')

    code('USE OP2_FORTRAN_DECLARATIONS')
    code('USE OP2_FORTRAN_RT_SUPPORT')
    code('USE ISO_C_BINDING')
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
#  Variable declarations
##########################################################################
    code('')
    comm(name+'variable declarations')
    code('')

    #strides for SoA
    if any_soa and not host_exec:
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
            k = k + [mapnames[g_m]]
            code('INTEGER(kind=4) :: opDat'+str(invinds[inds[g_m]-1]+1)+'_stride_OP2CONSTANT')
            code('!$omp declare target(opDat'+str(invinds[inds[g_m]-1]+1)+'_stride_OP2CONSTANT)')
      dir_soa = -1
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID and ((not dims[g_m].isdigit()) or int(dims[g_m]) > 1):
          code('INTEGER(kind=4) :: direct_stride_OP2CONSTANT')
          code('!$omp declare target(direct_stride_OP2CONSTANT)')
          dir_soa = g_m
          break

    code('')

    if is_soa > -1:
      code('#define OP2_SOA(var,dim,stride) var((dim-1)*stride+1)')

##########################################################################
#  Inline user kernel function
##########################################################################
    code('')
    code('CONTAINS')
    code('')
    if hydra:
      file_text += '!DEC$ ATTRIBUTES FORCEINLINE :: ' + name + '\n'
      code('')
      comm(name + ' user functions (CPU and GPU)')
      code('')
      text = text.replace('module','!module')
      text = text.replace('contains','!contains')
      text = text.replace('end !module','!end module')
      text = text.replace('recursive subroutine','subroutine')
      code('')
      #remove all comments
      util.const_list = []
      text = re.sub('!.*\n','\n',text)
#      if not host_exec:
#        text = replace_consts(text)

      text = text.replace('subroutine '+name, 'subroutine '+name+'_gpu')

      using_npdes = 0
      for g_m in range(0,nargs):
        if var[g_m] == 'npdes':
          using_npdes = 1
      if using_npdes==1:
        text = replace_npdes(text)

      if not host_exec:
        #find subroutine calls
        util.funlist = [name.lower()]
        plus_kernels = find_function_calls(text,'')

        if plus_kernels == '':
          text = replace_soa(text,nargs,soaflags,name,maps,accs,set_name,mapnames,1,hydra,bookleaf)
          
        text = text + '\n' + plus_kernels
        for fun in util.funlist:
          regex = re.compile('\\b'+fun+'\\b',re.I)
          text = regex.sub(fun+'_gpu',text)

        if plus_kernels <> '':
          print name
          for i in range(0,nargs):
            if soaflags[i]==1 and not (maps[i] ==OP_GBL):
              stage_flags[i] = 1;
              stage_soa = 1

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
      fid = open(name+'.inc', 'r')
      text = fid.read()
      text = replace_soa(text,nargs,soaflags,name,maps,accs,set_name,mapnames,1,hydra,bookleaf)
      text = text.replace(name, name+'_gpu',1)
      code(text)


    code('')

##########################################################################
#  Generate wrapper to iterate over set
##########################################################################

    code('SUBROUTINE op_wrap_'+name+'( &')
    depth = depth + 2
    if nopts >0:
      code('&  optflags,        &')
    for g_m in range(0,ninds):
      if invinds[g_m] in needDimList:
        code('& opDat'+str(invinds[g_m]+1)+'Dim, &')
      code('& opDat'+str(invinds[g_m]+1)+'Local, &')
      code('& opDat'+str(invinds[g_m]+1)+'Size, &')
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
    if ninds > 0:
      code('& col_reord, set_size, &')
    code('& bottom,top,set_size_full)')

    code('implicit none')
    if ninds>0:
      code('INTEGER(kind=4) set_size')
      code('INTEGER(kind=4) col_reord(set_size)')
    code('INTEGER(kind=4) set_size_full')
    if nopts>0:
      code('INTEGER(kind=4), VALUE :: optflags')
    for g_m in range(0,ninds):
      code('INTEGER(kind=4) opDat'+str(invinds[g_m]+1)+'Size')
      if invinds[g_m] in needDimList:
        code('INTEGER(kind=4) opDat'+str(invinds[g_m]+1)+'Dim')
      if soaflags[invinds[g_m]]:
        code(typs[invinds[g_m]]+' opDat'+str(invinds[g_m]+1)+'Local('+str(dims[invinds[g_m]])+'*opDat'+str(invinds[g_m]+1)+'Size)')
      else:
        code(typs[invinds[g_m]]+' opDat'+str(invinds[g_m]+1)+'Local('+str(dims[invinds[g_m]])+',opDat'+str(invinds[g_m]+1)+'Size)')
    for g_m in range(0,nargs):
      if maps[g_m] <> OP_MAP:
        if g_m in needDimList:
          code('INTEGER(kind=4) opDat'+str(g_m+1)+'Dim')
      if maps[g_m] == OP_ID:
        if soaflags[g_m]:
          code(typs[g_m]+' opDat'+str(g_m+1)+'Local('+str(dims[g_m])+'*set_size_full)')
        else:
          code(typs[g_m]+' opDat'+str(g_m+1)+'Local('+str(dims[g_m])+',set_size_full)')
      elif maps[g_m] == OP_GBL:
        if accs[g_m]<>OP_READ and accs[g_m]<>OP_WRITE and dims[g_m].isdigit() and int(dims[g_m])==1:
          code(typs[g_m]+' opDat'+str(g_m+1)+'Local')
        else:
          code(typs[g_m]+' opDat'+str(g_m+1)+'Local('+str(dims[g_m])+')')
    if nmaps > 0:
      k = []
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
          k = k + [mapnames[g_m]]
          code('INTEGER(kind=4) opDat'+str(invinds[inds[g_m]-1]+1)+'MapDim')
          code('INTEGER(kind=4) opDat'+str(invinds[inds[g_m]-1]+1)+'Map(opDat'+str(invinds[inds[g_m]-1]+1)+'MapDim*set_size)')


    if not host_exec:
      #when functions call functions, we can no longer reliably do SoA, therefore we need to stage everything in registers
      for g_m in range(0,nargs):
        if stage_flags[g_m] == 1:
          if g_m in needDimList:
                print 'Error, cannot statically determine dim of argument '+str(g_m+1)+' in kernel '+name
                sys.exit(-1)
          code(typs[g_m]+', DIMENSION('+dims[g_m]+') :: opDat'+str(g_m+1)+'Staged')

    code('')

    if not host_exec:
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL and accs[g_m] <> OP_READ and accs[g_m] <> OP_WRITE and (not dims[g_m].isdigit() or int(dims[g_m])>1):
          for d in range(0,int(dims[g_m])):
            code(typs[g_m]+' opDat'+str(g_m+1)+'Local_'+str(d+1))
          code(typs[g_m]+' opDat'+str(g_m+1)+'LocalArr('+dims[g_m]+')')
          

    code('INTEGER(kind=4) bottom,top,i1,i2')
    if nmaps > 0:
      k = []
      line = 'INTEGER(kind=4) '
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (not mapinds[g_m] in k):
          k = k + [mapinds[g_m]]
          line += 'map'+str(mapinds[g_m]+1)+'idx, '
      code(line[:-2])
    if stage_soa>0:
      code('INTEGER(kind=4) i3')

    code('')
    if not host_exec:
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL and accs[g_m] <> OP_READ and accs[g_m] <> OP_WRITE and (not dims[g_m].isdigit() or int(dims[g_m])>1):
          for d in range(0,int(dims[g_m])):
            code('opDat'+str(g_m+1)+'Local_'+str(d+1)+' = opDat'+str(g_m+1)+'Local('+str(d+1)+')')

    code('')

    line = '!$omp target teams distribute parallel do &\n'
    for g_m in range(0,ninds):
      line = line + '!$omp& map(to:opDat'+str(invinds[g_m]+1)+'Local) &\n'
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        line = line + '!$omp& map(to:opDat'+str(g_m+1)+'Local) &\n'
    if nmaps > 0:
      k = []
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
          k = k + [mapnames[g_m]]
          line = line + '!$omp& map(to:opDat'+str(invinds[inds[g_m]-1]+1)+'Map) &\n'
    if ninds > 0:
      line = line + '!$omp& map(to:col_reord) private(i1) &\n'
      for g_m in range(0,nargs):
        if maps[g_m] == OP_MAP and (not mapinds[g_m] in k):
          k = k + [mapinds[g_m]]
          line = line + '!$omp& private(map'+str(mapinds[g_m]+1)+'idx) &\n'


    if not host_exec:
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL and accs[g_m] <> OP_READ and accs[g_m] <> OP_WRITE:
          if int(dims[g_m])==1:
            if accs[g_m] == OP_INC:
              line = line + '!$omp& reduction(+:'+'opDat'+str(g_m+1)+'Local) map(tofrom:opDat'+str(g_m+1)+'Local) &\n'
            if accs[g_m] == OP_MIN:
              line = line + '!$omp& reduction(min:'+'opDat'+str(g_m+1)+'Local) map(tofrom:opDat'+str(g_m+1)+'Local) &\n'
            if accs[g_m] == OP_MAX:
              line = line + '!$omp& reduction(max:'+'opDat'+str(g_m+1)+'Local) map(tofrom:opDat'+str(g_m+1)+'Local) &\n'
          else:
            for d in range(0,int(dims[g_m])):
              if accs[g_m] == OP_INC:
                line = line + '!$omp& reduction(+:'+'opDat'+str(g_m+1)+'Local_'+str(d+1)+') map(tofrom:opDat'+str(g_m+1)+'Local_'+str(d+1)+') &\n'
              if accs[g_m] == OP_MIN:
                line = line + '!$omp& reduction(min:'+'opDat'+str(g_m+1)+'Local_'+str(d+1)+') map(tofrom:opDat'+str(g_m+1)+'Local_'+str(d+1)+') &\n'
              if accs[g_m] == OP_MAX:
                line = line + '!$omp& reduction(max:'+'opDat'+str(g_m+1)+'Local_'+str(d+1)+') map(tofrom:opDat'+str(g_m+1)+'Local_'+str(d+1)+') &\n'
      if stage_soa>0:
        line = line + '!$omp& private(i3) &\n'
      for g_m in range(0,nargs):
        if stage_flags[g_m] == 1:
          line = line + '!$omp& private(opDat'+str(g_m+1)+'Staged) &\n'

    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and accs[g_m] <> OP_READ and accs[g_m] <> OP_WRITE and (not dims[g_m].isdigit() or int(dims[g_m])>1):
        line = line + '!$omp& private(opDat'+str(g_m+1)+'LocalArr) &\n'
    line = line[:-2]
    if not host_exec:
      code(line)

    if ninds > 0 and not host_exec:
      DO('i2','bottom','top')
      code('i1 = col_reord(i2+1)')
    else:
      DO('i1','bottom','top')
    if not host_exec:
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL and accs[g_m] <> OP_READ and accs[g_m] <> OP_WRITE and (not dims[g_m].isdigit() or int(dims[g_m])>1):
          if accs[g_m] == OP_INC:
            code('opDat'+str(g_m+1)+'LocalArr = 0')
          if accs[g_m] == OP_MIN:
            code('opDat'+str(g_m+1)+'LocalArr = HUGE(opDat'+str(g_m+1)+'Local_1)')
          if accs[g_m] == OP_MAX:
            code('opDat'+str(g_m+1)+'LocalArr = -HUGE(opDat'+str(g_m+1)+'Local_1)')

    k = []
    for g_m in range(0,nargs):
      if maps[g_m] == OP_MAP and ((not (optflags[g_m]*nargs+mapinds[g_m]) in k) and (not mapinds[g_m] in k)):
        k = k + [(optflags[g_m]*nargs+mapinds[g_m])]
        if optflags[g_m]==1:
          IF('BTEST(optflags,'+str(optidxs[g_m])+')')
        if host_exec:
          code('map'+str(mapinds[g_m]+1)+'idx = opDat'+str(invmapinds[inds[g_m]-1]+1)+'Map(1 + i1 * opDat'+str(invmapinds[inds[g_m]-1]+1)+'MapDim + '+str(int(idxs[g_m])-1)+')+1')
        else:
          code('map'+str(mapinds[g_m]+1)+'idx = opDat'+str(invmapinds[inds[g_m]-1]+1)+'Map(1 + i1 + set_size * '+str(int(idxs[g_m])-1)+')+1')
        if optflags[g_m]==1:
          ENDIF()

    for g_m in range(0,nargs):
        if stage_flags[g_m] == 1:
          if optflags[g_m]==1:
            IF('BTEST(optflags,'+str(optidxs[g_m])+')')
          if maps[g_m] == OP_MAP:
            DO('i3','0', dims[g_m])
            code('opDat'+str(g_m+1)+'Staged(i3+1) = opDat'+str(invinds[inds[g_m]-1]+1)+'Local &')
            code('  & (i3 * '+get_stride_string(g_m,maps,mapnames,set_name,hydra,bookleaf)+' + map'+str(mapinds[g_m]+1)+'idx)')
            ENDDO()
          else:
            DO('i3','0', dims[g_m])
            code('opDat'+str(g_m+1)+'Staged(i3+1) = opDat'+str(g_m+1)+'Local &')
            code('  & (1 + i3 * direct_stride_OP2CONSTANT + i1)')
            ENDDO()
          if optflags[g_m]==1:
            ENDIF()

    comm('kernel call')
    line = 'CALL '+name+'_gpu( &'
    indent = '\n'+' '*depth
    for g_m in range(0,nargs):
      if stage_flags[g_m] == 1:
          line = line + indent + '& opDat'+str(g_m+1)+'Staged'
      elif maps[g_m] == OP_ID:
        if soaflags[g_m]:
          line = line + indent + '& opDat'+str(g_m+1)+'Local(i1+1)'
        else:
          line = line + indent + '& opDat'+str(g_m+1)+'Local(1,i1+1)'
      elif maps[g_m] == OP_MAP:
        if soaflags[g_m]:
          line = line +indent + '& opDat'+str(invinds[inds[g_m]-1]+1)+'Local(map'+str(mapinds[g_m]+1)+'idx)'
        else:
          line = line +indent + '& opDat'+str(invinds[inds[g_m]-1]+1)+'Local(1,map'+str(mapinds[g_m]+1)+'idx)'
      elif maps[g_m] == OP_GBL:
        if accs[g_m]<>OP_READ and accs[g_m] <> OP_WRITE:
          if dims[g_m].isdigit() and int(dims[g_m])==1:
            line = line + indent +'& opDat'+str(g_m+1)+'Local'
          else:
            if host_exec:
              line = line + indent +'& opDat'+str(g_m+1)+'Local(1)'
            else:
              line = line + indent +'& opDat'+str(g_m+1)+'LocalArr'
        else:
          line = line + indent +'& opDat'+str(g_m+1)+'Local(1)'
      if g_m < nargs-1:
        line = line +', &'
      else:
         line = line +' &'
    code(line + indent + '& )')
    if not host_exec:
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL and accs[g_m] <> OP_READ and accs[g_m] <> OP_WRITE and (not dims[g_m].isdigit() or int(dims[g_m])>1):
          for d in range(0,int(dims[g_m])):
            if accs[g_m] == OP_INC:
              code('opDat'+str(g_m+1)+'Local_'+str(d+1)+' = opDat'+str(g_m+1)+'Local_'+str(d+1)+' + opDat'+str(g_m+1)+'LocalArr('+str(d+1)+')')
            if accs[g_m] == OP_MIN:
              code('opDat'+str(g_m+1)+'Local_'+str(d+1)+' = MIN(opDat'+str(g_m+1)+'Local_'+str(d+1)+', opDat'+str(g_m+1)+'LocalArr('+str(d+1)+'))')
            if accs[g_m] == OP_MAX:
              code('opDat'+str(g_m+1)+'Local_'+str(d+1)+' = MAX(opDat'+str(g_m+1)+'Local_'+str(d+1)+', opDat'+str(g_m+1)+'LocalArr('+str(d+1)+'))')

    for g_m in range(0,nargs):
        if stage_flags[g_m] == 1 and accs[g_m] <> OP_READ:
          if optflags[g_m]==1:
            IF('BTEST(optflags,'+str(optidxs[g_m])+')')
          if maps[g_m] == OP_MAP:
            DO('i3','0', dims[g_m])
            code('opDat'+str(invinds[inds[g_m]-1]+1)+'Local(i3 * '+get_stride_string(g_m,maps,mapnames,set_name,hydra,bookleaf)+' + map'+str(mapinds[g_m]+1)+'idx) = &')
            code('  & opDat'+str(g_m+1)+'Staged(i3+1)')
            ENDDO()
          else:
            DO('i3','0', dims[g_m])
            code('opDat'+str(g_m+1)+'Local(1 + i3 * direct_stride_OP2CONSTANT + i1) = &')
            code('  & opDat'+str(g_m+1)+'Staged(i3+1)')
            ENDDO()
          if optflags[g_m]==1:
            ENDIF()
    depth = depth + 2

    depth = depth - 2
    ENDDO()
    if not host_exec:
      code('')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL and accs[g_m] <> OP_READ and accs[g_m] <> OP_WRITE and (not dims[g_m].isdigit() or int(dims[g_m])>1):
          for d in range(0,int(dims[g_m])):
            code('opDat'+str(g_m+1)+'Local('+str(d+1)+') = opDat'+str(g_m+1)+'Local_'+str(d+1))
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
    code('INTEGER(kind=4), SAVE :: calledTimes=0')

    if ninds > 0: #if indirect loop
      code('INTEGER(kind=4) :: exec_size')
      code('LOGICAL :: firstTime_'+name+' = .TRUE.')
      code('type ( c_ptr )  :: planRet_'+name)
      code('type ( op_plan ) , POINTER :: actualPlan_'+name)
      code('INTEGER(kind=4), POINTER, DIMENSION(:) :: col_reord_'+name) 
      code('INTEGER(kind=4), POINTER, DIMENSION(:) :: offset_'+name)
      code('INTEGER(kind=4), DIMENSION(1:'+str(nargs)+') :: indirectionDescriptorArray')
      code('INTEGER(kind=4) :: numberOfIndirectOpDats')
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
      if maps[g_m] == OP_GBL and accs[g_m] <> OP_READ and accs[g_m] <> OP_WRITE:
        code(typs[g_m]+', ALLOCATABLE, DIMENSION(:) :: opDat'+str(g_m+1)+'LocalReduction')

    code('')
    code('INTEGER(kind=4) :: i1,i2,n')
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
    code('& 0.d0, 0.00000_4,0.00000_4, 0)')

    #managing constants
    if any_soa and not host_exec:
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
            k = k + [mapnames[g_m]]
            IF('(calledTimes.EQ.0).OR.(opDat'+str(invinds[inds[g_m]-1]+1)+'_stride_OP2CONSTANT.NE.getSetSizeFromOpArg(opArg'+str(g_m+1)+'))')
            code('opDat'+str(invinds[inds[g_m]-1]+1)+'_stride_OP2CONSTANT = getSetSizeFromOpArg(opArg'+str(g_m+1)+')')
            code('!$omp target update to(opDat'+str(invinds[inds[g_m]-1]+1)+'_stride_OP2CONSTANT)') 
            ENDIF()
      if dir_soa<>-1:
          IF('(calledTimes.EQ.0).OR.(direct_stride_OP2CONSTANT.NE.getSetSizeFromOpArg(opArg'+str(dir_soa+1)+'))')
          code('direct_stride_OP2CONSTANT = getSetSizeFromOpArg(opArg'+str(dir_soa+1)+')')
          code('!$omp target update to(direct_stride_OP2CONSTANT)')
          ENDIF()


    code('call op_timers_core(startTime)')
    code('')
    #mpi halo exchange call
    if host_exec:
      code('n_upper = op_mpi_halo_exchanges(set%setCPtr,numberOfOpDats,opArgArray)')
    else:
      code('n_upper = op_mpi_halo_exchanges_cuda(set%setCPtr,numberOfOpDats,opArgArray)')
    code('')

    code('opSetCore => set%setPtr')
    code('')

    if ninds > 0:
      for g_m in range(0,nargs):
        code('indirectionDescriptorArray('+str(g_m+1)+') = '+str(inds[g_m]-1))
      code('')

      code('exec_size = opSetCore%size + opSetCore%exec_size')
      code('numberOfIndirectOpDats = '+str(ninds))
      code('')
      code('planRet_'+name+' = FortranPlanCaller( &')
      code('& userSubroutine//C_NULL_CHAR, &')
      code('& set%setCPtr, &')
      code('& partitionSize, &')
      code('& numberOfOpDats, &')
      code('& opArgArray, &')
      code('& numberOfIndirectOpDats, &')
      code('& indirectionDescriptorArray,4)')
      code('')
      code('CALL c_f_pointer(planRet_'+name+',actualPlan_'+name+')')
      code('CALL c_f_pointer(actualPlan_'+name+'%col_reord,col_reord_'+name+',(/exec_size/))')
      code('CALL c_f_pointer(actualPlan_'+name+'%color2_offsets,offset_'+name+',(/actualPlan_'+name+'%ncolors+1/))')

    for g_m in range(0,ninds):
      code('opDat'+str(invinds[g_m]+1)+'Cardinality = opArg'+str(invinds[g_m]+1)+'%dim * getSetSizeFromOpArg(opArg'+str(invinds[g_m]+1)+')')
      code('opDat'+str(invinds[g_m]+1)+'MapDim = getMapDimFromOpArg(opArg'+str(invinds[g_m]+1)+')')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code('opDat'+str(g_m+1)+'Cardinality = opArg'+str(g_m+1)+'%dim * getSetSizeFromOpArg(opArg'+str(g_m+1)+')')

    suffix = ''
    if not host_exec:
      suffix='_d'
    for g_m in range(0,ninds):
      code('CALL c_f_pointer(opArg'+str(invinds[g_m]+1)+'%data'+suffix+',opDat'+str(invinds[g_m]+1)+'Local,(/opDat'+str(invinds[g_m]+1)+'Cardinality/))')
      code('CALL c_f_pointer(opArg'+str(invinds[g_m]+1)+'%map_data'+suffix+',opDat'+str(invinds[g_m]+1)+'Map,(/exec_size*opDat'+str(invinds[g_m]+1)+'MapDim/))')
    for g_m in range(0,nargs):
      if maps[g_m] == OP_ID:
        code('CALL c_f_pointer(opArg'+str(g_m+1)+'%data'+suffix+',opDat'+str(g_m+1)+'Local,(/opDat'+str(g_m+1)+'Cardinality/))')
      elif maps[g_m] == OP_GBL:
        code('CALL c_f_pointer(opArg'+str(g_m+1)+'%data,opDat'+str(g_m+1)+'Local, (/opArg'+str(g_m+1)+'%dim/))')
        if accs[g_m] <> OP_READ and accs[g_m] <> OP_WRITE and not host_exec:
          code('allocate(opDat'+str(g_m+1)+'LocalReduction(opArg'+str(g_m+1)+'%dim)) ')
          code('opDat'+str(g_m+1)+'LocalReduction = opDat'+str(g_m+1)+'Local')
    code('')

    for idx in needDimList:
      dims[idx] = 'opArg'+str(idx+1)+'%dim'

    code('')

    if ninds > 0: #indirect loop host stub call
      code('')
      DO('i1','0','actualPlan_'+name+'%ncolors')

      IF('i1 .EQ. 1') #actualPlan_'+name+'%ncolors_core')
      code('CALL op_mpi_wait_all_cuda(numberOfOpDats,opArgArray)')
      ENDIF()
      code('')

      code('offset_b = offset_'+name+'(i1 + 1)')
      code('nelem = offset_'+name+'(i1 + 1 + 1)')
      
      code('CALL op_wrap_'+name+'( &')
      if nopts>0:
        code('& optflags, &')
      for g_m in range(0,ninds):
        if invinds[g_m] in needDimList:
          code('& opArg'+str(invinds[g_m]+1)+'%dim, &')
        code('& opDat'+str(invinds[g_m]+1)+'Local, &')
        code('& getSetSizeFromOpArg(opArg'+str(invinds[g_m]+1)+'), &')
      for g_m in range(0,nargs):
        if maps[g_m] == OP_ID:
          if g_m in needDimList:
            code('& opArg'+str(g_m+1)+'%dim, &')
          code('& opDat'+str(g_m+1)+'Local, &')
        elif maps[g_m] == OP_GBL:
          if g_m in needDimList:
            code('& opArg'+str(g_m+1)+'%dim, &')
          if accs[g_m] <> OP_READ and accs[g_m] <> OP_WRITE and not host_exec:
            code('& opDat'+str(g_m+1)+'LocalReduction(1), &')
          else:
            code('& opDat'+str(g_m+1)+'Local(1), &')
      if nmaps > 0:
        k = []
        for g_m in range(0,nargs):
          if maps[g_m] == OP_MAP and (not mapnames[g_m] in k):
            k = k + [mapnames[g_m]]
            code('& opDat'+str(invinds[inds[g_m]-1]+1)+'Map, &')
            code('& opDat'+str(invinds[inds[g_m]-1]+1)+'MapDim, &')
      code('& col_reord_'+name+', exec_size, offset_b, nelem, exec_size+opSetCore%nonexec_size )')

      if reduct and not host_exec:
        IF('i1 .EQ. actualPlan_'+name+'%ncolors_owned -1')
        for g_m in range(0,nargs):
          if maps[g_m] == OP_GBL and accs[g_m] <> OP_READ and accs[g_m] <> OP_WRITE:
            code('opDat'+str(g_m+1)+'Local = opDat'+str(g_m+1)+'LocalReduction')
        ENDIF()

      ENDDO()
    else:
      code('sliceStart = 0')
      code('sliceEnd = opSetCore%size')
      code('CALL op_wrap_'+name+'( &')
      if nopts>0:
        code('& optflags, &')
      for g_m in range(0,ninds):
        if invinds[g_m] in needDimList:
          code('& opArg'+str(invinds[g_m]+1)+'%dim, &')
        code('& opDat'+str(invinds[g_m]+1)+'Local, &')
        code('& getSetSizeFromOpArg(opArg'+str(invinds[g_m]+1)+'), &')
      for g_m in range(0,nargs):
        if g_m in needDimList:
          code('& opArg'+str(g_m+1)+'%dim, &')
        if maps[g_m] == OP_ID:
          code('& opDat'+str(g_m+1)+'Local, &')
        elif maps[g_m] == OP_GBL:
          if accs[g_m] <> OP_READ and accs[g_m] <> OP_WRITE and not host_exec:
            code('& opDat'+str(g_m+1)+'LocalReduction(1), &')
          else:
            code('& opDat'+str(g_m+1)+'Local(1), &')
      code('& sliceStart, sliceEnd, opSetCore%size+opSetCore%exec_size+opSetCore%nonexec_size)')

    IF('(n_upper .EQ. 0) .OR. (n_upper .EQ. opSetCore%core_size)')
    if host_exec:
      code('CALL op_mpi_wait_all(numberOfOpDats,opArgArray)')
    else:
      code('CALL op_mpi_wait_all_cuda(numberOfOpDats,opArgArray)')
    ENDIF()

    if ninds==0:
      for g_m in range(0,nargs):
        if maps[g_m] == OP_GBL and accs[g_m] <> OP_READ and accs[g_m] <> OP_WRITE and not host_exec:
          code('opDat'+str(g_m+1)+'Local = opDat'+str(g_m+1)+'LocalReduction')
    code('')
    if host_exec:
      code('CALL op_mpi_set_dirtybit(numberOfOpDats,opArgArray)')
    else:
      code('CALL op_mpi_set_dirtybit_cuda(numberOfOpDats,opArgArray)')
    code('')

    #reductions
    for g_m in range(0,nargs):
      if maps[g_m] == OP_GBL and accs[g_m] <> OP_READ and accs[g_m] <> OP_WRITE and not host_exec:
        code('deallocate( opDat'+str(g_m+1)+'LocalReduction )')
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
    code('calledTimes = calledTimes + 1')
    depth = depth - 2
    code('END SUBROUTINE')
    code('END MODULE')
    code('')

##########################################################################
#  output individual kernel file
##########################################################################
    if hydra:
      name = 'kernels/'+kernels[nk]['master_file']+'/'+name
      fid = open(name+'_omp4kernel.F95','w')
    elif bookleaf:
      fid = open(prefixes[prefix_i]+name+'_omp4kernel.f90','w')
    else:
      fid = open(name+'_omp4kernel.F90','w')
    date = datetime.datetime.now()
    fid.write('!\n! auto-generated by op2.py\n!\n\n')
    fid.write(file_text.strip())
    fid.close()
