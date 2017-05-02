#!/usr/bin/env python
#/opt/intel/intelpython27/bin/python

"""
 OP2 source code transformation tool

 This tool parses the user's original source code to produce
 target-specific code to execute the user's kernel functions.

 This prototype is written in Python and is directly based on the
 parsing and code generation of the matlab source code transformation code

 usage: ./op2_fortran.py 'file1','file2',...

 This code generator is for parsing applications written using the OP2 FORTRAN API

 This takes as input

 file1.F90, file2.F90, ...

 and produces as output modified versions .....

 file1_op.F90, file2_op.F90, ...

 then calls a number of target-specific code generators
 to produce individual kernel files of the form

 xxx_kernel.F90  -- for OpenMP x86 execution
 xxx_kernel.CUF   -- for CUDA execution (based on PGI CUDA FORTRAN)

"""

import sys
import re
import datetime

#import openmp code generation function
import op2_gen_openmp
from op2_gen_openmp import *
import op2_gen_openmp2
from op2_gen_openmp2 import *
import op2_gen_openmp3
from op2_gen_openmp3 import *

import op2_gen_openacc
from op2_gen_openacc import *

import op2_gen_openmp4
from op2_gen_openmp4 import *



#import mpiseq code generation function
import op2_gen_mpiseq
from op2_gen_mpiseq import *
import op2_gen_mpiseq2
from op2_gen_mpiseq2 import *
import op2_gen_mpiseq3
from op2_gen_mpiseq3 import *
import op2_gen_mpivec
from op2_gen_mpivec import *


#import cuda code generation function
import op2_gen_cuda
from op2_gen_cuda import *
import op2_gen_cuda_permute
from op2_gen_cuda_permute import *
import op2_gen_cudaINC
from op2_gen_cudaINC import *
import op2_gen_cuda_old
from op2_gen_cuda_old import *


#
# declare constants
#

ninit = 0; nexit = 0; npart = 0; nhdf5 = 0; nconsts  = 0; nkernels = 0;
consts = []
kernels = []

OP_ID   = 1;  OP_GBL   = 2;  OP_MAP = 3;

OP_READ = 1;  OP_WRITE = 2;  OP_RW  = 3;
OP_INC  = 4;  OP_MAX   = 5;  OP_MIN = 6;

OP_accs_labels = ['OP_READ','OP_WRITE','OP_RW','OP_INC','OP_MAX','OP_MIN' ]

global file_format, cont, comment

file_format = 0
cont = '& '
comment = '! '

hydra = 0
bookleaf=0

# from http://stackoverflow.com/a/241506/396967
##########################################################################
# Remove comments from text
##########################################################################

def comment_remover(text):

  def replacer(match):
      s = match.group(0)
      if s.startswith('/'):
          return ""
      else:
          return s
  pattern = re.compile(
      r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
      re.DOTALL | re.MULTILINE
  )
  return re.sub(pattern, replacer, text)


##########################################################################
# parsing for op_init/op_exit/op_partition/op_hdf5 calls
##########################################################################

def op_parse_calls(text):

  # remove comments just for this call
  text = comment_remover(text)

  inits = len(re.findall('op_init', text))
  exits = len(re.findall('op_exit', text))
  parts = len(re.findall('op_partition', text))
  hdf5s = len(re.findall('hdf5', text))

  return (inits, exits, parts, hdf5s)


##########################################################################
# parsing for op_decl_const calls
##########################################################################

def op_decl_const_parse(text):

    consts = []
    for m in re.finditer('call(.+)op_decl_const(.*)\((.*)\)', text):
        args = m.group(3).split(',')

        # check for syntax errors
        if len(args) != 3:
            print 'Error in op_decl_const : must have three arguments'
            return

        consts.append({
            'loc': m.start(),
            'dim': args[1].strip(),
            'type': args[1].strip(),
            'name': args[0].strip(),
            'name2': args[2].strip()
            })

    return consts

##########################################################################
# parsing for arguments in op_par_loop to find the correct closing brace
##########################################################################
def arg_parse(text,j):

    depth = 0
    loc2 = j;
    while 1:
      if text[loc2] == '(':
        depth = depth + 1

      elif text[loc2] == ')':
        depth = depth - 1
        if depth == 0:
          return loc2
      loc2 = loc2 + 1

def typechange(text):
  if '"INTEGER(kind=4)"' in text:
    return '"i4"'
  elif '"INTEGER(kind=4):soa"' in text:
    return '"i4:soa"'
  elif '"REAL(kind=8)"' in text:
    return '"r8"'
  elif '"REAL(kind=8):soa"' in text:
    return '"r8:soa"'
  elif '"REAL(kind=4)"' in text:
    return '"r4"'
  elif '"REAL(kind=4):soa"' in text:
    return '"r4:soa"'
  elif '"logical"' in text:
    return '"logical"'
  return text


def get_arg_dat(arg_string, j):
  loc = arg_parse(arg_string,j+1)
  dat_args_string = arg_string[arg_string.find('(',j)+1:loc]

  #remove comments
  dat_args_string = comment_remover(dat_args_string)
  dat_args_string = dat_args_string.replace('&','')
  #check for syntax errors
  if len(dat_args_string.split(',')) <> 6:
    print 'Error parsing op_arg_dat(%s): must have six arguments' \
          % dat_args_string
    return

  # split the dat_args_string into  6 and create a struct with the elements
  # and type as op_arg_dat
  temp_dat = {'type':'op_arg_dat',
  'dat':dat_args_string.split(',')[0].strip(),
  'idx':dat_args_string.split(',')[1].strip(),
  'map':dat_args_string.split(',')[2].strip(),
  'dim':dat_args_string.split(',')[3].strip(),
  'typ':dat_args_string.split(',')[4].strip(),
  'acc':dat_args_string.split(',')[5].strip(),
  'opt':''}

  if 'DNPDE' in temp_dat['dim']:
    temp_dat['dim'] = temp_dat['dim'].replace('DNPDE','6')
  if 'npdes' in temp_dat['dim']:
    temp_dat['dim'] = temp_dat['dim'].replace('npdes','NPDE')
  if 'nfcrow' in temp_dat['dim']:
    temp_dat['dim'] = temp_dat['dim'].replace('nfcrow','DNFCROW')
  if 'ntqmu' in temp_dat['dim']:
    temp_dat['dim'] = temp_dat['dim'].replace('ntqmu','DNTQMU')
  if temp_dat['dim']=='njaca':
    temp_dat['dim']='1'#'1*1'
  if 'mpdes' in temp_dat['dim']:
    temp_dat['dim'] = temp_dat['dim'].replace('mpdes','10')
  if 'maxgrp' in temp_dat['dim']:
    temp_dat['dim'] = temp_dat['dim'].replace('maxgrp','1000')
  if temp_dat['dim']=='njacs':
    temp_dat['dim']='1'#'1*1'
  if '"r8' in temp_dat['typ']:
    temp_dat['typ']= temp_dat['typ'].replace('"r8','"REAL(kind=8)')
  if '"i4' in temp_dat['typ']:
    temp_dat['typ']= temp_dat['typ'].replace('"i4','"INTEGER(kind=4)')
  if temp_dat['typ']=='"logical"':
    temp_dat['typ']='"logical*1"'
  return temp_dat

def get_opt_arg_dat(arg_string, j):
  loc = arg_parse(arg_string,j+1)
  dat_args_string = arg_string[arg_string.find('(',j)+1:loc]

  #remove comments
  dat_args_string = comment_remover(dat_args_string)
  dat_args_string = dat_args_string.replace('&','')

  #check for syntax errors
  if len(dat_args_string.split(',')) <> 7:
    print 'Error parsing op_arg_dat(%s): must have six arguments' \
          % dat_args_string
    return

  # split the dat_args_string into  6 and create a struct with the elements
  # and type as op_arg_dat
  temp_dat = {'type':'op_opt_arg_dat',
  'opt':dat_args_string.split(',')[0].strip(),
  'dat':dat_args_string.split(',')[1].strip(),
  'idx':dat_args_string.split(',')[2].strip(),
  'map':dat_args_string.split(',')[3].strip(),
  'dim':dat_args_string.split(',')[4].strip(),
  'typ':dat_args_string.split(',')[5].strip(),
  'acc':dat_args_string.split(',')[6].strip()}

  if 'DNPDE' in temp_dat['dim']:
    temp_dat['dim'] = temp_dat['dim'].replace('DNPDE','6')
  if 'npdes' in temp_dat['dim']:
    temp_dat['dim'] = temp_dat['dim'].replace('npdes','NPDE')
  if 'ntqmu' in temp_dat['dim']:
    temp_dat['dim'] = temp_dat['dim'].replace('ntqmu','DNTQMU')
  if 'nfcrow' in temp_dat['dim']:
    temp_dat['dim'] = temp_dat['dim'].replace('nfcrow','DNFCROW')
  if temp_dat['dim']=='njaca':
    temp_dat['dim']='1'#'1*1'
  if temp_dat['dim']=='njacs':
    temp_dat['dim']='1'#'1*1'
  if 'mpdes' in temp_dat['dim']:
    temp_dat['dim'] = temp_dat['dim'].replace('mpdes','10')
  if 'maxgrp' in temp_dat['dim']:
    temp_dat['dim'] = temp_dat['dim'].replace('maxgrp','1000')
  if '"r8' in temp_dat['typ']:
    temp_dat['typ']= temp_dat['typ'].replace('"r8','"REAL(kind=8)')
  if '"i4' in temp_dat['typ']:
    temp_dat['typ']= temp_dat['typ'].replace('"i4','"INTEGER(kind=4)')
  if temp_dat['typ']=='"logical"':
    temp_dat['typ']='"logical*1"'

  return temp_dat

def get_arg_gbl(arg_string, k):
  loc = arg_parse(arg_string,k+1)
  gbl_args_string = arg_string[arg_string.find('(',k)+1:loc]

  #remove comments
  gbl_args_string = comment_remover(gbl_args_string)
  gbl_args_string = gbl_args_string.replace('&','')

  #check for syntax errors
  if len(gbl_args_string.split(',')) != 4:
    print 'Error parsing op_arg_gbl(%s): must have four arguments' \
          % gbl_args_string
    return

  # split the gbl_args_string into  4 and create a struct with the elements
  # and type as op_arg_gbl
  temp_gbl = {'type':'op_arg_gbl',
  'data':gbl_args_string.split(',')[0].strip(),
  'dim':gbl_args_string.split(',')[1].strip(),
  'typ':gbl_args_string.split(',')[2].strip(),
  'acc':gbl_args_string.split(',')[3].strip(),
  'opt':''}

  if 'DNPDE' in temp_gbl['dim']:
    temp_gbl['dim'] = temp_gbl['dim'].replace('DNPDE','6')
  if 'nfcrow' in temp_gbl['dim']:
    temp_gbl['dim'] = temp_gbl['dim'].replace('nfcrow','DNFCROW')
  if 'npdes' in temp_gbl['dim']:
    temp_gbl['dim'] = temp_gbl['dim'].replace('npdes','NPDE')
  if 'ntqmu' in temp_gbl['dim']:
    temp_gbl['dim'] = temp_gbl['dim'].replace('ntqmu','DNTQMU')
  if 'maxzone' in temp_gbl['dim']:
    temp_gbl['dim'] = temp_gbl['dim'].replace('maxzone','DMAXZONE')
  if 'mpdes' in temp_gbl['dim']:
    temp_gbl['dim'] = temp_gbl['dim'].replace('mpdes','10')
  if 'maxgrp' in temp_gbl['dim']:
    temp_gbl['dim'] = temp_gbl['dim'].replace('maxgrp','1000')
  if temp_gbl['typ']=='"r8"':
    temp_gbl['typ']='"REAL(kind=8)"'
  if temp_gbl['typ']=='"i4"':
    temp_gbl['typ']='"INTEGER(kind=4)"'
  if temp_gbl['typ']=='"logical"':
    temp_gbl['typ']='"logical*1"'

  return temp_gbl

def append_init_soa(text):
  text = re.sub('\\bop_init(\\w*)\\b\\s*\((.*)\)','op_init\\1_soa(\\2,1)', text)
  text = re.sub('\\bop_mpi_init(\\w*)\\b\\s*\((.*)\)','op_mpi_init\\1_soa(\\2,1)', text)
  return text
##########################################################################
# parsing for op_par_loop calls
##########################################################################

def op_par_loop_parse(text):
    loop_args = []

    search = "op_par_loop"
    i = text.find(search)
    while i > -1:
      arg_string = text[text.find('(',i)+1:arg_parse(text,i+11)]

      #parse arguments in par loop
      temp_args = []
      num_args = 0

      #try:
      #parse each op_arg_dat
      search2 = "op_arg_dat"
      search3 = "op_arg_gbl"
      search4 = "op_opt_arg_dat"
      j = arg_string.find(search2)
      k = arg_string.find(search3)
      l = arg_string.find(search4)

      while j > -1 or k > -1 or l > -1:
        index = min(j if (j > -1) else sys.maxint,k if (k > -1) else sys.maxint,l if (l > -1) else sys.maxint)

        if index == j:
          temp_dat = get_arg_dat(arg_string,j)
          #append this struct to a temporary list/array
          temp_args.append(temp_dat)
          num_args = num_args + 1
          j= arg_string.find(search2, j+11)
        elif  index == k:
          temp_gbl = get_arg_gbl(arg_string,k)
          #append this struct to a temporary list/array
          temp_args.append(temp_gbl)
          num_args = num_args + 1
          k= arg_string.find(search3, k+11)
        elif  index == l :
          temp_dat = get_opt_arg_dat(arg_string,l)
          #append this struct to a temporary list/array
          temp_args.append(temp_dat)
          num_args = num_args + 1
          l = arg_string.find(search4, l+15)

      temp = {'loc':i,
              'name1':arg_string.split(',')[0].strip(),
              'set':arg_string.split(',')[1].strip(),
              'args':temp_args,
              'nargs':num_args}

      loop_args.append(temp)
      i=text.find(search, i+10)

    print '\n\n'
    return (loop_args)

###################END OF FUNCTIONS DECLARATIONS #########################



##########################################################################
#                      ** BEGIN MAIN APPLICATION **
##########################################################################

#####################loop over all input source files#####################
init_ctr = 1
auto_soa=os.getenv('OP_AUTO_SOA','0')
if len(sys.argv) > 1:
  if sys.argv[1] == 'hydra':
    hydra = 1
    init_ctr=2
  if sys.argv[1] == 'bookleaf':
    bookleaf = 1
    init_ctr=2

for a in range(init_ctr,len(sys.argv)):
  print 'processing file '+ str(a) + ' of ' + str(len(sys.argv)-init_ctr) + ' '+ \
  str(sys.argv[a])

  src_file = str(sys.argv[a])
  f = open(src_file,'r')
  text = f.read()
  if src_file.split('.')[1].upper() == 'F90' or src_file.split('.')[1].upper() == 'F95':
    file_format = 90
    cont = '& '
    cont_end = ' &'
    comment = '! '
  elif src_file.split('.')[1].upper() == 'F77' or src_file.split('.')[1].upper() == 'F':
    file_format = 77
    cont = '& '
    cont_end = ''
    comment = 'C '
  else:
    print "Error in parsing file: unsupported file format, only *.F90, *.F95 or *.F77 supported"
    exit()

############ check for op_init/op_exit/op_partition/op_hdf5 calls ########

  inits, exits, parts, hdf5s = op_parse_calls(text)

  if inits+exits+parts+hdf5s > 0:
    print ' '
  if inits > 0:
    print'contains op_init call'
    if auto_soa<>'0':
      text = append_init_soa(text)
  if exits > 0:
    print'contains op_exit call'
  if parts > 0:
    print'contains op_partition call'
  if hdf5s > 0:
    print'contains op_hdf5 calls'

  ninit = ninit + inits
  nexit = nexit + exits
  npart = npart + parts
  nhdf5 = nhdf5 + hdf5s

########################## parse and process constants ###################

  const_args = []
  if not hydra:
    const_args = op_decl_const_parse(text)

  #cleanup '&' symbols from name and convert dim to integer
  for i  in range(0,len(const_args)):
    if const_args[i]['name'][0] == '&':
      const_args[i]['name'] = const_args[i]['name'][1:]
      const_args[i]['dim'] = int(const_args[i]['dim'])

  #check for repeats
  nconsts = 0
  for i  in range(0,len(const_args)):
    repeat = 0
    name = const_args[i]['name']
    for c in range(0,nconsts):
      if  const_args[i]['name'] == consts[c]['name']:
        repeat = 1
        if const_args[i]['type'] <> consts[c]['type']:
          print 'type mismatch in repeated op_decl_const'
        if const_args[i]['dim'] <> consts[c]['dim']:
          print 'size mismatch in repeated op_decl_const'

    if repeat > 0:
      print 'repeated global constant ' + const_args[i]['name']
    else:
      print '\nglobal constant ('+ const_args[i]['name'].strip() + ') of size ' \
      + str(const_args[i]['dim'] + ' and type ' + const_args[i]['type'].strip())

  #store away in master list
    if repeat == 0:
      nconsts = nconsts + 1
      temp = {'dim': const_args[i]['dim'],
      'type': const_args[i]['type'].strip(),
      'name': const_args[i]['name'].strip()}
      consts.append(temp)

###################### parse and process op_par_loop calls ###############

  loop_args = op_par_loop_parse(text)

  for i in range (0, len(loop_args)):
    name = loop_args[i]['name1']
    set_name = loop_args[i]['set']
    nargs = loop_args[i]['nargs']
    print '\nprocessing kernel '+name+' with '+str(nargs)+' arguments',

#
# process arguments
#

#
# NOTE: Carlo's FORTRAN API has one fewer arguments than C++ API
#
    var = ['']*nargs
    idxs = [0]*nargs
    dims = ['']*nargs
    maps = [0]*nargs
    mapnames = ['']*nargs
    typs = ['']*nargs
    accs = [0]*nargs
    soaflags = [0]*nargs
    optflags= [0]*nargs

    for m in range (0,nargs):
      arg_type =  loop_args[i]['args'][m]['type']
      args =  loop_args[i]['args'][m]

      if arg_type.strip() == 'op_arg_dat' or arg_type.strip() == 'op_opt_arg_dat':
        var[m] = args['dat']
        idxs[m] =  args['idx']
        if str(args['map']).strip() == 'OP_ID':
          maps[m] = OP_ID
          if int(idxs[m]) <> -1:
             print 'invalid index for argument'+str(m)
        else:
          maps[m] = OP_MAP
          mapnames[m] = str(args['map']).strip()

        dims[m] = args['dim']
        soa_loc = args['typ'].find(':soa')
        if ((auto_soa=='1') and (((not dims[m].isdigit()) or int(dims[m])>1)) and (soa_loc < 0)):
          soa_loc = len(args['typ'])-1

        if soa_loc > 0:
          soaflags[m] = 1
          typs[m] = args['typ'][1:soa_loc]
        else:
            typs[m] = args['typ'][1:-1]

        l = -1
        for l in range(0,len(OP_accs_labels)):
          if args['acc'].strip() == OP_accs_labels[l].strip():
            break

        if l == -1:
          print 'unknown access type for argument '+str(m)
        else:
          accs[m] = l+1

        if arg_type.strip() == 'op_opt_arg_dat':
          optflags[m] = 1
          # if soaflags[m] == 1:
          #   print "ERROR: cannot have SoA and optional argument at the same time"
          #   sys.exit(-1)
        else:
          optflags[m] = 0

      if arg_type.strip() == 'op_arg_gbl':
        maps[m] = OP_GBL
        var[m] = args['data']
        dims[m] = args['dim']
        typs[m] = args['typ'][1:-1]
        optflags[m] = 0

        l = -1
        for l in range(0,len(OP_accs_labels)):
          if args['acc'].strip() == OP_accs_labels[l].strip():
            break

        if l == -1:
          print 'unknown access type for argument '+str(m)
        else:
          accs[m] = l+1

      if (maps[m]==OP_GBL) and (accs[m]==OP_WRITE or accs[m]==OP_RW):
         print 'invalid access type for argument '+str(m)

      if (maps[m]<>OP_GBL) and (accs[m]==OP_MIN or accs[m]==OP_MAX):
         print 'invalid access type for argument '+str(m)

    print ' '
#
# identify indirect datasets
#
    ninds = 0
    inds = [0]*nargs
    invinds = [0]*nargs
    invmapinds = [0]*nargs
    mapinds = [0]*nargs
    indtyps = ['']*nargs
    inddims = ['']*nargs
    indaccs = [0]*nargs

    j = [i for i, x in enumerate(maps) if x == OP_MAP]

    while len(j) > 0:

        indtyps[ninds] = typs[j[0]]
        inddims[ninds] = dims[j[0]]
        indaccs[ninds] = accs[j[0]]
        invinds[ninds] = j[0] #inverse mapping
        ninds = ninds + 1
        for i in range(0,len(j)):
          if var[j[0]] == var[j[i]] and typs[j[0]] == typs[j[i]] \
          and accs[j[0]] == accs[j[i]]: #same variable
            inds[j[i]] = ninds


        k = []
        for i in range(0,len(j)):
          if not (var[j[0]] == var[j[i]] and typs[j[0]] == typs[j[i]] \
          and accs[j[0]] == accs[j[i]]): #same variable
            k = k+[j[i]]
        j = k

    if ninds > 0:
      invmapinds = invinds[:]
      for i in range(0,ninds):
        for j in range(0,i):
          if (mapnames[invinds[i]] == mapnames[invinds[j]]):
            invmapinds[i] = invmapinds[j]
      for i in range(0,nargs):
        mapinds[i] = i
        for j in range(0,i):
          if (maps[i] == OP_MAP) and (mapnames[i] == mapnames[j]) and (idxs[i] == idxs[j]):
            mapinds[i] = mapinds[j]
#
# check for repeats
#
    repeat = False
    rep1 = False
    rep2 = False

    for nk in range (0,nkernels):
      rep1 = kernels[nk]['name'] == name and \
      kernels[nk]['nargs'] == nargs and \
      kernels[nk]['ninds'] == ninds
      if rep1:
         rep2 = True
         for arg in range(0,nargs):
            rep2 =  rep2 and kernels[nk]['dims'][arg] == dims[arg] and \
            kernels[nk]['maps'][arg] == maps[arg] and \
            kernels[nk]['typs'][arg] == typs[arg] and \
            kernels[nk]['accs'][arg] == accs[arg] and \
            kernels[nk]['idxs'][arg] == idxs[arg] and \
            kernels[nk]['soaflags'][arg] == soaflags[arg] and \
            kernels[nk]['optflags'][arg] == optflags[arg] and \
            kernels[nk]['inds'][arg] == inds[arg]

         for arg in range(0,ninds):
            rep2 =  rep2 and kernels[nk]['inddims'][arg] == inddims[arg] and \
            kernels[nk]['indaccs'][arg] == indaccs[arg] and \
            kernels[nk]['indtyps'][arg] == indtyps[arg] and \
            kernels[nk]['invinds'][arg] == invinds[arg]

         if rep2:
           print 'repeated kernel with compatible arguments: '+ kernels[nk]['name']
           repeat = True
         else:
           print 'repeated kernel with incompatible arguments: ERROR'
           break

#
# output various diagnostics
#
    if not repeat:
      print '  local constants:',
      for arg in range(0,nargs):
          if maps[arg] == OP_GBL and accs[arg] == OP_READ:
            print str(arg),
      print '\n  global reductions:',
      for arg in range(0,nargs):
          if maps[arg] == OP_GBL and accs[arg] <> OP_READ:
            print str(arg),
      print '\n  direct arguments:',
      for arg in range(0,nargs):
          if maps[arg] == OP_ID:
            print str(arg),
      print '\n  indirect arguments:',
      for arg in range(0,nargs):
          if maps[arg] == OP_MAP:
            print str(arg),
      if ninds > 0:
          print '\n  number of indirect datasets: '+str(ninds),

      print '\n'
#
# store away in master list
#
    if not repeat:
      nkernels = nkernels+1;
      temp = {'name': name,
              'set' : set_name,
              'nargs': nargs,
              'dims': dims,
              'maps': maps,
              'var': var,
              'typs': typs,
              'accs': accs,
              'idxs': idxs,
              'inds': inds,
              'soaflags': soaflags,
              'optflags': optflags,

              'ninds': ninds,
              'inddims': inddims,
              'indaccs': indaccs,
              'indtyps': indtyps,
              'invinds': invinds,
              'mapnames' : mapnames,
              'mapinds': mapinds,
              'invmapinds' : invmapinds }

      if hydra==1:
        temp['master_file'] = src_file.split('.')[0].replace('mod_','')
        search = 'use '+temp['master_file'].upper()+'_KERNELS_'+name+'\n'
        i = text.rfind(search)
        if i > -1:
          temp['mod_file'] = search.strip()
        else:
          search = 'use '+temp['master_file'].upper()+'_KERNELS_'+name[:-1]
          i = text.rfind(search)
          if i > -1:
            temp['mod_file'] = search.strip()
          else:
            print'  ERROR: no module file found!  '
      if bookleaf==1:
        file_part = src_file.split('/')
        file_part = file_part[len(file_part)-1]
        temp['master_file'] = file_part.split('.')[0]
        if temp['master_file'] in name:
          temp['mod_file'] = temp['master_file'] + '_kernels.f90'
        else:
          temp['mod_file'] = 'common_kernels.f90'

      kernels.append(temp)

########################## output source file  ############################

  fid = open(src_file.replace('.','_op.'), 'w')
#  if file_format == 90:
#    fid = open(src_file.split('.')[0]+'_op.F90', 'w')
#  elif file_format == 77:
#    fid = open(src_file.split('.')[0]+'_op.F', 'w')
  date = datetime.datetime.now()
  #fid.write('!\n! auto-generated by op2_fortran.py on '+date.strftime("%Y-%m-%d %H:%M")+'\n!\n\n')
  fid.write('!\n! auto-generated by op2_fortran.py\n!\n\n')

  loc_old = 0
  #read original file and locate header location
  if bookleaf:
    loc_header = [text.lower().find('use op2_bookleaf')]
  else:
    loc_header = [text.find('use OP2_Fortran_Reference')]

  #get locations of all op_decl_consts
  n_consts = len(const_args)
  loc_consts = [0]*n_consts
  for n in range(0,n_consts):
    loc_consts[n] = const_args[n]['loc']

  #get locations of all op_par_loops
  n_loops    = len(loop_args);
  loc_loops  = [0]*n_loops
  for n in range(0,n_loops):
    loc_loops[n] = loop_args[n]['loc']

  locs = sorted(loc_header+loc_consts+loc_loops)



#
# process header, loops and constants
#
  for loc in range(0,len(locs)):
    fid.write(text[loc_old:locs[loc]-1])
    loc_old = locs[loc]-1
    indent = ''
    ind = 0;
    while 1:
      if text[locs[loc]-ind] == '\n':
        break
      indent = indent + ' '
      ind = ind + 1

    if locs[loc] in loc_header:
      line = ''
      if hydra==0:
        for nk in range (0,len(kernels)):
          if text.find(kernels[nk]['name']) > -1:
            line = line +'\n'+'  use ' + kernels[nk]['name'].upper()+'_MODULE'
        line = line + '\n'+indent

      fid.write(line[2:len(line)]);
      if bookleaf:
        loc_old = locs[loc] # keep the original include
      else:
        loc_old = locs[loc]+25
      continue

    if locs[loc] in loc_consts:# stripping the op_decl_consts -- as there is no implementation required
      line = ''
      fid.write(line);
      endofcall = text.find('\n', locs[loc])
      loc_old = endofcall+1
      continue

    if locs[loc] in loc_loops:
       indent = indent + ' '*len('op_par_loop')
       if file_format == 77:
         indent='     '
       endofcall = arg_parse(text,locs[loc]+11)
       #endofcall = text.find('\n\n', locs[loc])
       curr_loop = loc_loops.index(locs[loc])
       name = loop_args[curr_loop]['name1']
       if file_format == 90:
         line = str(' '+name+'_host(&\n'+indent+'& "'+name+'",'+
                loop_args[curr_loop]['set']+', '+cont_end+'\n')
       elif file_format == 77:
         line = str(' '+name+'_host(\n'+indent+'& "'+name+'",'+
                 loop_args[curr_loop]['set']+', '+cont_end+'\n')

       for arguments in range(0,loop_args[curr_loop]['nargs']):
         elem = loop_args[curr_loop]['args'][arguments]
         if elem['type'] == 'op_arg_dat':
            line = line + indent + cont + elem['type'] + '(' + elem['dat'] + ','+ elem['idx'] \
            + ','+ elem['map'] + ','+ elem['dim']+ ','+ typechange(elem['typ']) +','+ elem['acc']
         elif elem['type'] == 'op_opt_arg_dat':
             line = line + indent + cont + elem['type'] + '(' +elem['opt']+','+ elem['dat'] + ','+ elem['idx'] \
             + ','+ elem['map'] + ','+ elem['dim']+ ','+ typechange(elem['typ']) +','+ elem['acc']
         elif elem['type'] == 'op_arg_gbl':
            line = line + indent + cont + elem['type'] + '(' + elem['data'] + ','+ elem['dim'] \
            +','+ typechange(elem['typ'])+','+ elem['acc']

         if arguments <> loop_args[curr_loop]['nargs'] - 1:
           line = line + '), '+cont_end+'\n'
         else:
           line = line + '))\n'

       fid.write(line)

       loc_old = endofcall+1
       continue



  fid.write(text[loc_old:])
  fid.close()
  if hydra == 1 or bookleaf==1:
    fid = open(src_file.replace('.','_op.'), 'r')
    #if file_format == 90:
    #  fid = open(src_file.split('.')[0]+'_op.F90', 'r')
    #elif file_format == 77:
    #  fid = open(src_file.split('.')[0]+'_op.F', 'r')


    text = fid.read()
    fid.close()
    if hydra:
      #replace = 'use OP2_FORTRAN_DECLARATIONS\n#ifdef OP2_ENABLE_CUDA\n       use HYDRA_CUDA_MODULE\n#endif\n'
      replace = 'use OP2_FORTRAN_DECLARATIONS\n'
      text = text.replace('use OP2_FORTRAN_DECLARATIONS\n',replace)
    if bookleaf:
      text = text.replace('USE OP2_Fortran_Reference\n','')
      text = text.replace('USE common_kernels','! USE common_kernels')
      file_part = src_file.split('/')
      file_part = file_part[len(file_part)-1]
      master_file = file_part.split('.')[0]
      text = text.replace('USE '+master_file+'_kernels','! USE USE '+master_file+'_kernels')
    for nk in range (0,len(kernels)):
      replace = kernels[nk]['mod_file']+'_MODULE'+'\n'
      text = text.replace(kernels[nk]['mod_file']+'\n', replace)

    fid = open(src_file.replace('.','_op.'), 'w')
    #if file_format == 90:
    #  fid = open(src_file.split('.')[0]+'_op.F90', 'w')
    #elif file_format == 77:
    #  fid = open(src_file.split('.')[0]+'_op.F', 'w')
    fid.write(text)
    fid.close()

  f.close()
#end of loop over input source files

########################## errors and warnings ############################

if ninit==0:
  print' '
  print'-----------------------------'
  print'  ERROR: no call to op_init  '
  print'-----------------------------'

if nexit==0:
  print' '
  print'-------------------------------'
  print'  WARNING: no call to op_exit  '
  print'-------------------------------'

if npart==0 and nhdf5>0:
  print' '
  print'---------------------------------------------------'
  print'  WARNING: hdf5 calls without call to op_partition '
  print'---------------------------------------------------'

########## finally, generate target-specific kernel files ################

#MPI+SEQ
#op2_gen_mpiseq(str(sys.argv[init_ctr]), date, consts, kernels, hydra)  # generate host stubs for MPI+SEQ
op2_gen_mpiseq3(str(sys.argv[init_ctr]), date, consts, kernels, hydra, bookleaf)  # generate host stubs for MPI+SEQ -- optimised by removing the overhead due to fortran c to f pointer setups
op2_gen_mpivec(str(sys.argv[init_ctr]), date, consts, kernels, hydra, bookleaf)  # generate host stubs for MPI+SEQ with intel vectorization optimisations

#OpenMP
op2_gen_openmp3(str(sys.argv[init_ctr]), date, consts, kernels, hydra, bookleaf)  # optimised by removing the overhead due to fortran c to f pointer setups
#op2_gen_openmp2(str(sys.argv[init_ctr]), date, consts, kernels, hydra) # version without staging
#op2_gen_openmp(str(sys.argv[init_ctr]), date, consts, kernels, hydra)  # original version - one that most op2 papers refer to

#CUDA
#op2_gen_cuda(str(sys.argv[1]), date, consts, kernels, hydra, bookleaf)
op2_gen_cuda_permute(str(sys.argv[init_ctr]), date, consts, kernels, hydra,bookleaf) # permute does a different coloring (permute execution within blocks by color)
#op2_gen_cudaINC(str(sys.argv[1]), date, consts, kernels, hydra)      # stages increment data only in shared memory
#op2_gen_cuda_old(str(sys.argv[1]), date, consts, kernels, hydra)     # Code generator targettign Fermi GPUs

#OpenACC
op2_gen_openacc(str(sys.argv[init_ctr]), date, consts, kernels, hydra, bookleaf)  # optimised by removing the overhead due to fortran c to f pointer setups

#OpenMP4 offload
op2_gen_openmp4(str(sys.argv[init_ctr]), date, consts, kernels, hydra, bookleaf)  # optimised by removing the overhead due to fortran c to f pointer setups

#if hydra:
#  op2_gen_cuda_hydra() #includes several Hydra specific features

##########################################################################
#                      ** END MAIN APPLICATION **
##########################################################################
