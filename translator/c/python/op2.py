#!/usr/bin/python
#
# OP2 source code transformation tool
#
# This tool parses the user's original source code to produce
# target-specific code to execute the user's kernel functions.
#
# This prototype is written in Python 
#
# usage: op2('file1','file2',...)
#
# This takes as input
#
# file1.cpp, file2.cpp, ...
#
# and produces as output modified versions
#
# file1_op.cpp, file2_op.cpp, ...
#
# then calls a number of target-specific code generators
# to produce individual kernel files of the form
#
# xxx_kernel.cpp  -- for OpenMP x86 execution
# xxx_kernel.cu   -- for CUDA execution
#
# plus a master kernel file of the form
#
# file1_kernels.cpp  -- for OpenMP x86 execution
# file1_kernels.cu   -- for CUDA execution
#

import sys
import re
import datetime

#import openmp code generation function
import op2_gen_openmp
from op2_gen_openmp import *

#
# declare constants
#

ninit = 0
nexit = 0
npart = 0
nhdf5 = 0

nconsts  = 0
nkernels = 0
consts = []
kernels = []

OP_ID   = 1;  OP_GBL   = 2;  OP_MAP = 3;

OP_READ = 1;  OP_WRITE = 2;  OP_RW  = 3;
OP_INC  = 4;  OP_MAX   = 5;  OP_MIN = 6;

OP_accs_labels = ['OP_READ','OP_WRITE','OP_RW','OP_INC','OP_MAX','OP_MIN' ]



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
    
    #remove comments just for this call
    text = comment_remover(text)
    
    inits = 0
    search = "op_init"
    found = text.find(search)
    while found > -1:
      found=text.find(search, found+1)
      inits = inits + 1
    
    exits = 0
    search = "op_exit"
    found = text.find(search)
    while found > -1:
      found=text.find(search, found+1)
      exits = exits + 1
      
    parts = 0
    search = "op_partition"
    found = text.find(search)
    while found > -1:
      found=text.find(search, found+1)
      parts = parts + 1
            
    hdf5s = 0
    search = "hdf5"
    found = text.find(search)
    while found > -1:
      found=text.find(search, found+1)
      hdf5 = hdf5s + 1
    
    return (inits, exits, parts, hdf5s)

##########################################################################
# parsing for op_decl_const calls
##########################################################################

def op_decl_const_parse(text):
    consts = []
    num_const = 0
    search = "op_decl_const"
    i = text.find(search)
    while i > -1:
      const_string = text[text.find('(',i)+1: text.find(')',i+13)]
      #print 'Args found at index i : '+const_string      
      
      #remove comments
      const_string = comment_remover(const_string)
        
      #check for syntax errors
      if len(const_string.split(',')) <> 3:
        print 'Error in op_decl_const : must have three arguments'
        return
      
      temp = {'loc': i,
        'dim':const_string.split(',')[0],
        'type': const_string.split(',')[1],
        'name':const_string.split(',')[2],
        'name2':const_string.split(',')[2]}
      
      consts.append(temp)
      
      i=text.find(search, i+13)
      num_const = num_const + 1
            
    return (consts)


##########################################################################
# parsing for arguments in op_par_loop to find the correct closing brace
##########################################################################
def arg_parse(text,j):
    #print text
    #text = text.strip()
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


def find_all(string, occurrence):
    found = 0
    while True:
      found = string.find(occurrence, found)
      if found != -1:
        yield found
      else:
        break
      found += 1
    

##########################################################################
# parsing for op_par_loop calls
##########################################################################

def op_par_loop_parse(text):
    loop_args = []
    
    search = "op_par_loop"
    i = text.find(search)
    while i > -1:
      arg_string = text[text.find('(',i)+1:text.find(';',i+11)]
      #print arg_string

      #parse arguments in par loop
      temp_args = []
      num_args = 0

      #try:
      #parse each op_arg_dat
      search2 = "op_arg_dat"
      search3 = "op_arg_gbl"
      j = arg_string.find(search2)
      k = arg_string.find(search3)
      
      while j > -1 or k > -1:
        print str(j) +'  '+str(k)
        if  k <= -1 : 
          loc = arg_parse(arg_string,j+1)
          dat_args_string = arg_string[arg_string.find('(',j)+1:loc]
          print dat_args_string
        
          #remove comments
          dat_args_string = comment_remover(dat_args_string)
        
          #check for syntax errors
          if len(dat_args_string.split(',')) <> 6:
            print 'Error in parsing op_arg_dat('+ dat_args_string +'): must have six arguments'
            return
        
          #split the dat_args_string into  6 and create a struct with the elements and type as op_arg_dat
          temp_dat = {'type':'op_arg_dat', 
            'dat':dat_args_string.split(',')[0].strip(),
            'idx':dat_args_string.split(',')[1].strip(),
            'map':dat_args_string.split(',')[2].strip(),
            'dim':dat_args_string.split(',')[3].strip(),
            'typ':dat_args_string.split(',')[4].strip(),
            'acc':dat_args_string.split(',')[5].strip()
          }
          #append this struct to a temporary list/array
          num_args = num_args + 1
          temp_args.append(temp_dat)
          j= arg_string.find(search2, j+11) 
          #k= arg_string.find(search3, j+11) 
        
        elif  j <= -1 :
          loc = arg_parse(arg_string,k+1)
          gbl_args_string = arg_string[arg_string.find('(',k)+1:loc]
          print gbl_args_string
        
          #remove comments
          gbl_args_string = comment_remover(gbl_args_string)
        
          #check for syntax errors
          if len(gbl_args_string.split(',')) <> 4:
            print 'Error in parsing op_arg_gbl('+ gbl_args_string +'): must have four arguments'
            return
         
          #split the gbl_args_string into  4 and create a struct with the elements and type as op_arg_gbl
          temp_gbl = {'type':'op_arg_gbl', 
            'data':gbl_args_string.split(',')[0].strip(),
            'dim':gbl_args_string.split(',')[1].strip(),
            'typ':gbl_args_string.split(',')[2].strip(),
            'acc':gbl_args_string.split(',')[3].strip()}
            
          #append this struct to a temporary list/array
          num_args = num_args + 1
          temp_args.append(temp_gbl)
          #j= arg_string.find(search2, k+11) 
          k= arg_string.find(search3, k+11) 
          
        elif j < k:
          loc = arg_parse(arg_string,j+1)
          dat_args_string = arg_string[arg_string.find('(',j)+1:loc]
          print dat_args_string
        
          #remove comments
          dat_args_string = comment_remover(dat_args_string)
        
          #check for syntax errors
          if len(dat_args_string.split(',')) <> 6:
            print 'Error in parsing op_arg_dat('+ dat_args_string +'): must have six arguments'
            return
        
          #split the dat_args_string into  6 and create a struct with the elements and type as op_arg_dat
          temp_dat = {'type':'op_arg_dat', 
            'dat':dat_args_string.split(',')[0].strip(),
            'idx':dat_args_string.split(',')[1].strip(),
            'map':dat_args_string.split(',')[2].strip(),
            'dim':dat_args_string.split(',')[3].strip(),
            'typ':dat_args_string.split(',')[4].strip(),
            'acc':dat_args_string.split(',')[5].strip()
          }
          #append this struct to a temporary list/array
          num_args = num_args + 1
          temp_args.append(temp_dat)
          j= arg_string.find(search2, j+11) 
          #k= arg_string.find(search3, j+11) 
          
        else:
          loc = arg_parse(arg_string,k+1)
          gbl_args_string = arg_string[arg_string.find('(',k)+1:loc]
          print gbl_args_string
        
          #remove comments
          gbl_args_string = comment_remover(gbl_args_string)
        
          #check for syntax errors
          if len(gbl_args_string.split(',')) <> 4:
            print 'Error in parsing op_arg_gbl('+ gbl_args_string +'): must have four arguments'
            return
         
          #split the gbl_args_string into  4 and create a struct with the elements and type as op_arg_gbl
          temp_gbl = {'type':'op_arg_gbl', 
            'data':gbl_args_string.split(',')[0].strip(),
            'dim':gbl_args_string.split(',')[1].strip(),
            'typ':gbl_args_string.split(',')[2].strip(),
            'acc':gbl_args_string.split(',')[3].strip()}
        
          #append this struct to a temporary list/array
          num_args = num_args + 1
          temp_args.append(temp_gbl)
          #j= arg_string.find(search2, k+11) 
          k= arg_string.find(search3, k+11) 
       
                
        #j= arg_string.find(search2, j+10) 
        #k= arg_string.find(search3, k+10)     
        #end of inner while loop
          
      temp = {'loc':i,
      'name1':arg_string.split(',')[0].strip(),
      'name2':arg_string.split(',')[1].strip(),
      'set':arg_string.split(',')[2].strip(),
      'args':temp_args,
      'nargs':num_args}
      
      loop_args.append(temp)
      i=text.find(search, i+10)
    #print loop_args
    print '\n\n'    
    return (loop_args)


###################END OF FUNCTIONS DECLARATIONS #########################




    
##########################################################################
#  loop over all input source files
##########################################################################
for a in range(1,len(sys.argv)):
  print 'processing file '+ str(a) + ' of ' + str(len(sys.argv)-1) + ' '+ \
  str(sys.argv[a]) 
    
  src_file = str(sys.argv[a])
  f = open(src_file,'r')
  text = f.read()
 
    
##########################################################################
# check for op_init/op_exit/op_partition/op_hdf5 calls
##########################################################################    
    
  inits, exits, parts, hdf5s = op_parse_calls(text)
  
  if inits+exits+parts+hdf5s > 0:
    print ' '
  if inits > 0:
    print'contains op_init call'
  if exits > 0:
    print'contains op_exit call'
  if parts > 0:
    print'contains op_partition call'
  if hdf5s > 0:
    print'contains op_hdf5 calls'
  
  ninit = ninit + inits;
  nexit = nexit + exits;
  npart = npart + parts;
  nhdf5 = nhdf5 + hdf5s;
    
    
##########################################################################
# parse and process constants
##########################################################################    
    
  const_args = op_decl_const_parse(text)
  #print'  number of constants declared '+str(len(const_args))
  #print'  contains constants '+str(const_args)
    
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
      print '\nglobal constant ('+ const_args[i]['name'].strip() + ') of size ' + str(const_args[i]['dim'])
      
  #store away in master list
    if repeat == 0:
      nconsts = nconsts + 1
      temp = {'dim': const_args[i]['dim'],
      'type': const_args[i]['type'].strip(),
      'name': const_args[i]['name'].strip()}
      consts.append(temp)
      

##########################################################################
# parse and process op_par_loop calls
##########################################################################
   
  loop_args = op_par_loop_parse(text)
  #print loop_args
  for i in range (0, len(loop_args)):
    name = loop_args[i]['name1']
    nargs = loop_args[i]['nargs']
    print '\nprocessing kernel '+name+' with '+str(nargs)+' arguments',
      
#
# process arguments
#
    var = ['']*nargs
    idxs = [0]*nargs
    dims = ['']*nargs
    maps = [0]*nargs
    typs = ['']*nargs
    accs = [0]*nargs
    soaflags = [0]*nargs
      
    for m in range (0,nargs):
      arg_type =  loop_args[i]['args'][m]['type']
      args =  loop_args[i]['args'][m]
       
      if arg_type.strip() == 'op_arg_dat':
        var[m] = args['dat']
        idxs[m] =  args['idx']
         
        if str(args['map']).strip() == 'OP_ID':
          maps[m] = OP_ID
          if int(idxs[m]) <> -1:
             print 'invalid index for argument'+str(m)
        else:
          maps[m] = OP_MAP
        
        dims[m] = args['dim']
        soa_loc = args['typ'].find(':soa')
        
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
       
      if arg_type.strip() == 'op_arg_gbl':
        maps[m] = OP_GBL
        var[m] = args['data']
        dims[m] = args['dim']
        typs[m] = args['typ'][1:-1]
          
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
       
      #print var[m]+' '+str(idxs[m])+' '+str(maps[m])+' '+\
      #str(dims[m])+' '+typs[m]+' '+str(accs[m])
      
    print ' '
    
#
# identify indirect datasets
#
    ninds = 0
    inds = [0]*nargs
    invinds = [0]*nargs
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
        
    #print inds
    #print invinds
    
    
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
    	    	kernels[nk]['inds'][arg] == inds[arg]
    	    
    	   for arg in range(0,ninds):
    	    	rep2 =  rep2 and kernels[nk]['inddims'][arg] == inddims[arg] and \
    	    	kernels[nk]['indaccs'][arg] == indaccs[arg] and \
    	    	kernels[nk]['indtyps'][arg] == indtyps[arg] and \
    	    	kernels[nk]['invinds'][arg] == invinds[arg]
    	   
	   if rep2:
    	    	print 'repeated kernel with compatible arguments: '+ kernels[nk]['name'],
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
	'nargs': nargs,
	'dims': dims,
	'maps': maps,
	'var': var,
	'typs': typs,
	'accs': accs,
	'idxs': idxs,
	'inds': inds,
	'soaflags': soaflags,
	
	'ninds': ninds,
	'inddims': inddims,
	'indaccs': indaccs,
	'indtyps': indtyps,
	'invinds': invinds
      }
      kernels.append(temp)

  
##########################################################################
# output new source file
########################################################################## 
  fid = open(src_file.split('.')[0]+'_op.cpp', 'w')
  date = datetime.datetime.now()
  fid.write('//\n// auto-generated by op2.py on '+date.strftime("%Y-%m-%d %H:%M")+'\n//\n\n')

  loc_old = 0

  #read original file and locate header location
  loc_header = [text.find("op_seq.h")]
  
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
        
    if locs[loc]in loc_header:
      fid.write(' "op_lib_cpp.h"\nint op2_stride = 1;\n#define OP2_STRIDE(arr, idx) arr[op2_stride*(idx)]\n\n')
      fid.write('//\n// op_par_loop declarations\n//\n')
      
      for k in range (0,nkernels):
        line = '\nvoid op_par_loop_'+kernels[k]['name']+'(char const *, op_set,\n'
        for n in range(1,kernels[k]['nargs']):
          line = line+'  op_arg,\n'
        line = line+'  op_arg );\n'
        fid.write(line);
      
      fid.write('\n');  
      loc_old = locs[loc]+11
      continue
      
    if locs[loc] in loc_loops:
       indent = indent + ' '*len('op_par_loop')
       endofcall = text.find(';', locs[loc]) 
       curr_loop = loc_loops.index(locs[loc])
       name = loop_args[curr_loop]['name1']
       line = str(' op_par_loop_'+name+'('+loop_args[curr_loop]['name2']+','+
              loop_args[curr_loop]['set']+',\n'+indent)
       
       for arguments in range(0,loop_args[curr_loop]['nargs']):
         elem = loop_args[curr_loop]['args'][arguments]
         if elem['type'] == 'op_arg_dat':
            line = line + elem['type'] + '(' + elem['dat'] + ','+ elem['idx'] \
            + ','+ elem['map'] + ','+ elem['dim'] + ','+ elem['typ'] + ',' \
            + elem['acc'] +'),\n'+indent
         elif elem['type'] == 'op_arg_gbl':
            line = line + elem['type'] + '(' + elem['data'] + ','+ elem['dim'] \
            + ','+ elem['typ'] + ','+ elem['acc'] +'),\n'+indent
       
       fid.write(line[0:-len(indent)-2]+');')
       
       loc_old = endofcall+1
       continue
       
    if locs[loc] in loc_consts:
       curr_const = loc_consts.index(locs[loc])
       endofcall = text.find(';', locs[loc]) 
       name = const_args[curr_const]['name'] 
       fid.write(indent[0:-2]+'op_decl_const2("'+name.strip()+'",'+ \
       str(const_args[curr_const]['dim'])+',' +const_args[curr_const]['type']+ \
       ','+const_args[curr_const]['name2'].strip()+');')
       loc_old = endofcall+1
       continue       
       

  fid.write(text[loc_old:]) 
  fid.close()
    
  f.close() 
#end of loop over input source files


##########################################################################
#  errors and warnings
###########################################################################

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


##########################################################################
#  finally, generate target-specific kernel files
##########################################################################

op2_gen_openmp(str(sys.argv[1]), date, consts, kernels)

