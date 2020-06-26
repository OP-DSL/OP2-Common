#!/usr/bin/env python

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
import os
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

def arg_parse2(text, j):
    """Parsing arguments in op_par_loop to find the correct closing brace"""

    depth = 0
    loc2 = j
    arglist = []
    prev_start = j
    while 1:
        if text[loc2] == '(':
            if depth == 0:
                prev_start = loc2+1
            depth = depth + 1

        elif text[loc2] == ')':
            depth = depth - 1
            if depth == 0:
                arglist.append(text[prev_start:loc2].replace('&','').strip())
                return arglist

        elif text[loc2] == ',':
            if depth == 1:
                arglist.append(text[prev_start:loc2].replace('&','').strip())
                prev_start = loc2+1
        elif text[loc2] == '{':
            depth = depth + 1
        elif text[loc2] == '}':
            depth = depth - 1
        loc2 = loc2 + 1


const_list = []

def replace_consts(text):
  global const_list
  i = text.find('use HYDRA_CONST_MODULE')
  if i > -1:
    fi2 = open("hydra_constants_list.txt","r")
    for line in fi2:
      fstr = '\\b'+line[:-1]+'\\b'
      rstr = line[:-1]+'_OP2CONSTANT'
      j = re.search(fstr,text)
      if not (j is None) and not (line[:-1] in const_list):
        const_list = const_list + [line[:-1]]
      text = re.sub(fstr,rstr,text)
  return text

def replace_npdes(text):
  #
  # substitute npdes with DNPDE
  #
  i = re.search('\\bnpdes\\b',text)
  if not (i is None):
    j = i.start()
    i = re.search('\\bnpdes\\b',text[j:])
    j = j + i.start()+5
    i = re.search('\\bnpdes\\b',text[j:])
    j = j + i.start()+5
    text = text[0:j] + re.sub('\\bnpdes\\b','NPDE',text[j:])
  return text

funlist = []
funlist2 = []

def get_stride_string(g_m,maps,stride,set_name):
  OP_ID   = 1;  OP_GBL   = 2;  OP_MAP = 3;
  if maps[g_m] == OP_ID:
    return 'direct_stride_OP2CONSTANT'
  if maps[g_m] == OP_GBL:
    return '(gridDim%x*blockDim%x)'
  else:
    idx = stride[g_m]
    return 'opDat'+str(idx+1)+'_stride_OP2CONSTANT'

def replace_soa_subroutines(funcs,idx,soaflags,maps,accs,mapnames,repl_inc,hydra,bookleaf,stride=[]):
  OP_ID   = 1;  OP_GBL   = 2;  OP_MAP = 3;
  name = funcs[idx]['function_name']
  if len(stride)==0:
    stride = [0]*len(funcs[idx]['args'])
    for g_m in range(0,len(funcs[idx]['args'])):
      if g_m >= len(maps):
          print name, maps, funcs[idx]['args']
      if maps[g_m] == OP_MAP:
        stride[g_m] = mapnames.index(mapnames[g_m])
  if funcs[idx]['soa_converted'] == 0:
    funcs[idx]['soaflags'] = soaflags
    funcs[idx]['function_text'] = replace_soa(funcs[idx]['function_text'],
                          len(funcs[idx]['args']),
                          soaflags, name, maps, accs, '', mapnames, repl_inc, hydra, bookleaf,stride)
  funcs[idx]['soa_converted'] = 1
  for funcall in funcs[idx]['calls']:
    OP_ID   = 1;  OP_GBL   = 2;  OP_MAP = 3;
    nargs = len(funcall['args'])
    call_name = funcall['function_name']
    idx_called = -1
    for i in range(0,len(funcs)):
      if funcs[i]['function_name']==call_name:
        idx_called = i
    if idx_called == -1:
      print 'ERROR, subroutine not found in replace_soa_subroutines: ' + call_name
    if funcs[idx_called]['soa_converted'] == 1:
      return funcs
    soaflags2 = [0]*nargs
    stride2 = [0]*nargs
    maps2 = [0]*nargs
    accs2 = [0]*nargs
    for i in range(0,nargs):
      arg = funcall['args'][i]
      #strip indexing
      j = arg.find('(')
      if j > 0:
        arg = arg[0:j]
      if arg in funcs[idx]['args']:
        orig_idx = funcs[idx]['args'].index(arg)
        soaflags2[i] = soaflags[orig_idx]
        maps2[i] = maps[orig_idx]
        accs2[i] = accs[orig_idx]
        stride2[i] = stride[orig_idx]
    if funcs[idx_called]['soa_converted'] == 1 and soaflags2 != funcs[idx_called]['soaflags']:
      print 'ERROR: soaflags mismatch for repeated function call: ' + funcs[idx_called]['function_name']
    funcs = replace_soa_subroutines(funcs,idx_called,soaflags2,maps2,accs2,mapnames,repl_inc,hydra,bookleaf,stride2)
  return funcs

def replace_soa(text,nargs,soaflags,name,maps,accs,set_name,mapnames,repl_inc,hydra,bookleaf,stride=[]):
  OP_ID   = 1;  OP_GBL   = 2;  OP_MAP = 3;

  OP_READ = 1;  OP_WRITE = 2;  OP_RW  = 3;
  OP_INC  = 4;  OP_MAX   = 5;  OP_MIN = 6;

  if len(stride)==0:
    stride = [0]*nargs
    for g_m in range(0,nargs):
      if maps[g_m] == OP_MAP:
        stride[g_m] = mapnames.index(mapnames[g_m])
  #
  # Apply SoA to variable accesses
  #
  j = text.find(name)
  endj = arg_parse(text,j)
  while text[j] <> '(':
      j = j + 1
  arg_list = text[j+1:endj]
  arg_list = arg_list.replace('&','')
  varlist = ['']*nargs
  leading_dim = [-1]*nargs
  follow_dim = ['-1']*nargs
  
  for g_m in range(0,nargs):
    varlist[g_m] = arg_list.split(',')[g_m].strip()
  for g_m in range(0,nargs):
    if soaflags[g_m] and (repl_inc or (not (maps[g_m]==OP_MAP and accs[g_m]==OP_INC))):
      #Start looking for the variable in the code, after the function signature
      loc1 = endj
      p = re.compile('\\b'+varlist[g_m]+'\\b')
      nmatches = len(p.findall(text[loc1:]))
      for id in range(0,nmatches):
        #Search for the next occurence
        i = p.search(text[loc1:])
        #Skip commented out ones
        j = text[:loc1+i.start()].rfind('\n')
        if j > -1 and text[j:loc1+i.start()].find('!')>-1:
          loc1 = loc1+i.end()
          continue

        #check to see if label is followed by array subscript
        idx = loc1 + i.end()
        while text[idx].isspace():
          idx = idx+1

        #normal subscript access or shape definition with varname(size)
        if text[idx] == '(':
          #opening bracket
          beginarg = idx+1
          #Find closing bracket
          endarg = arg_parse(text,loc1+i.start())
        #looking for shape definition, look backward for DIMENSION(size)
        elif leading_dim[g_m] == -1:
          j = text[:loc1+i.start()].rfind('\n')
          j2 = loc1+i.start()
          while text[j+1:j2].strip()[0] == '&':
            j2 = j-1
            j = text[:j2].rfind('\n')

          k = text[j:j2].lower().find('dimension')
          if k <> -1:
            beginarg = j+k+text[j+k:j2].find('(')+1
            endarg = arg_parse(text,beginarg-1)
          else:
            print 'Could not find shape specification for '+varlist[g_m]+' in '+name+'- assuming scalar'
            beginarg = loc1 + i.end()
            endarg=beginarg
        else:
          print 'Warning: array subscript not found for ' + varlist[g_m] + ' in '+name
          loc1 = loc1+i.start() + len(varlist[g_m])
          continue


        #If this is the first time we see the argument (i.e. its declaration)
        if leading_dim[g_m] == -1:
          if (len(text[beginarg:endarg].split(',')) > 1):
            #if it's 2D, remember leading dimension, and make it 1D
            leading_dim[g_m] = text[beginarg:endarg].split(',')[0]
            if '0:' in text[beginarg:endarg].split(',')[1]:
              follow_dim[g_m] = ''
#              follow_dim[g_m] = text[beginarg:endarg].split(',')[1].split(':')[0]
            text = text[:beginarg] + '*'+' '*(endarg-beginarg-1) + text[endarg:]
          elif beginarg==endarg:
            leading_dim[g_m] = 0
          else:
            leading_dim[g_m] = 1
          #Continue search after this instance of the variable
          loc1 = loc1+i.start()+len(varlist[g_m])
        else:
          #If we have seen this variable already, then it's in the actual code, replace it with macro
          macro = 'OP2_SOA('+text[loc1+i.start():loc1+i.end()]+','
          if leading_dim[g_m] == 1:
            macro = macro + text[beginarg:endarg]
          elif leading_dim[g_m] == 0:
            if beginarg==endarg:
              loc1 = loc1+i.start() + len(varlist[g_m])
            else:
              print 'Warning: '+varlist[g_m]+' in '+name+' was assumed scalar, but now accessed as array: '+text[beginarg:endarg]
              macro = macro + text[beginarg:endarg]
          else:
            macro = macro + text[beginarg:endarg].split(',')[0] + '+('+text[beginarg:endarg].split(',')[1]+follow_dim[g_m]+')*('+leading_dim[g_m]+')'
          macro = macro + ', ' + get_stride_string(g_m,maps,stride,set_name) + ')'
          text = text[:loc1+i.start()] + macro + text[endarg+1:]
          #Continue search after this instance of the variable
          loc1 = loc1+i.start() + len(macro)

  return text

def convert_F90(text):
    text = re.sub(r'\nc','\n!',text)
    text = re.sub(r'\n     &','&\n     &',text)
    return text

#i may point into the middle of the line...
def comment_line(text, i):
    orig_i = i
    linebegin = text[0:i].rfind('\n')
    lineend = i+text[i:].find('\n')
    line = text[linebegin:lineend]
    #comment this line, shift indices:
    text = text[0:linebegin+1]+'!'+text[linebegin+1:]
    lineend = lineend+1
    i=i+1
    if len(line.strip())>0 and line.strip()[0]=='&':
        #keep going backwards
        b_lineend = linebegin
        b_linebegin = text[0:b_lineend].rfind('\n')
        line = text[b_linebegin:b_lineend]
        text = text[0:b_linebegin+1]+'!'+text[b_linebegin+1:]
        lineend = lineend+1
        linebegin = linebegin+1
        b_lineend = b_lineend + 1
        i=i+1
        while len(line.strip())>0 and line.strip()[0]=='&':
            b_lineend = b_linebegin
            b_linebegin = text[0:b_lineend].rfind('\n')
            line = text[b_linebegin:b_lineend]
            text = text[0:b_linebegin+1]+'!'+text[b_linebegin+1:]
            lineend = lineend+1
            linebegin = linebegin+1
            b_lineend = b_lineend + 1
            i=i+1
    nextline_end = lineend+1+text[lineend+1:].find('\n')
    line = text[lineend:nextline_end]
    while len(line.strip())>0 and line.strip()[0]=='&':
      text = text[0:lineend+1]+'!'+text[lineend+1:]
      lineend = nextline_end + 1
      nextline_end = lineend+1+text[lineend+1:].find('\n')
      line = text[lineend:nextline_end]
    return text, i-orig_i

def remove_jm76(text):
  jm76_funs = ['SET_RGAS_RATIOS', 'INITGAS0', 'INITGAS1', 'INITGAS2', 'INITFUEL', 'INITVAP0', 'TOTALTP', 'TOTALP', 'TOTALT', 'STATICTP', 'QSTATICTP', 'USTATICTP', 'FLOWSPEED', 'DREALGA', 'SPECS', 'REALPHI', 'REALH', 'REALCP', 'REALRG', 'REALT', 'ISENT', 'ISENP']
  for fun in jm76_funs:
    k = re.search(r'\n\s+.*\b'+fun+r'\b',text)
    while not (k is None):
      text,comm_inserted = comment_line(text,k.start()+1)
      k = re.search(r'\n\s+.*\b'+fun+r'\b',text)
  return text

def get_kernel(text, name):
  i = re.search(r'\n\s*\bsubroutine\b\s*'+name+r'\b', text, re.IGNORECASE)
  if i:
    #attempt 1: find end subroutine
    j = re.search(r'\n\s*\bend\s+subroutine\b'+name+r'\b', text[i.start():], re.IGNORECASE)
    if j:
      return text[i.start():i.start()+j.end()]
    #attempt 2: find next subroutine
    j = re.search(r'\n\s*\bsubroutine\b', text[i.end():], re.IGNORECASE)
    if j:
      last_end = i.start()+[m.end() for m in re.finditer(r'\n\s*\bend\b',text[i.start():i.end()+j.start()], re.IGNORECASE)][-1]
      return text[i.start():last_end+text[last_end:].find('\n')]
    #attempt 3: end of file
    last_end = i.start()+[m.end() for m in re.finditer(r'\n\s*\bend\b',text[i.start():], re.IGNORECASE)][-1]
    return text[i.start():last_end+text[last_end:].find('\n')]
  else:
    return ''


def find_function_calls(text, attr, name=''):
  global funlist
  global funlist2
  text = remove_jm76(text)
  j = re.search(r'\n\s*call hyd_',text)
  while not (j is None):
    text,comm_inserted = comment_line(text,j.start()+1)
    j = re.search(r'\n\s*call hyd_',text)
  j = re.search(r'\n\s*external',text)
  while not (j is None):
    text,comm_inserted = comment_line(text,j.start()+1)
    j = re.search(r'\n\s*external',text)
  j = re.search(r'\n\s*call op_',text)
  while not (j is None):
    text,comm_inserted = comment_line(text,j.start()+1)
    j = re.search(r'\n\s*call op_',text)
  j = re.search(r'\n\s*write\b',text)
  while not (j is None):
    text,comm_inserted = comment_line(text,j.start()+1)
    j = re.search(r'\n\s*write\b',text)

  search_offset = 0
  my_subs = ''
  children_subs=''
  funlist_index = len(funlist2)
  if name == '':
      i = text.find('subroutine')
      openbracket = i+text[i:].find('(')
      name = text[i+len('subroutine'):openbracket].strip()
  else:
      i = text.find(name)
      openbracket = i+text[i:].find('(')
  funlist_entry = {'function_name' : name,
                  'function_text' : text,
                  'args' : arg_parse2(text,openbracket-1),
                  'soa_converted' : 0,
                  'calls': []}
  funlist2 = funlist2 + [funlist_entry]
  res=re.search('\\bcall\\b',text)
  while (not (res is None)):
    i = search_offset + res.start() + 4
    #Check if line is commented
    j = text[:i].rfind('\n')
    if j > -1 and text[j:i].find('!')>-1:
      search_offset = i
      res=re.search('\\bcall\\b',text[search_offset:])
      continue
    #find name: whatever is in front of opening bracket
    openbracket = i+text[i:].find('(')
    fun_name = text[i:openbracket].strip()
    if 'hyd_' in fun_name:
      print text[j:openbracket]
    if fun_name.lower() in funlist:
      funcall_entry = {'function_name': fun_name+'_gpu',
                       'args' : arg_parse2(text,openbracket-1)}
      funlist2[funlist_index]['calls'].append(funcall_entry)
      search_offset = i
      res=re.search('\\bcall\\b',text[search_offset:])
      continue

    #print fun_name

    funlist = funlist + [fun_name.lower()]
    funcall_entry = {'function_name': fun_name+'_gpu',
                     'args' : arg_parse2(text,openbracket-1)}
    funlist2[funlist_index]['calls'].append(funcall_entry)
    #find signature
    line = text[openbracket:openbracket+text[openbracket:].find('\n')].strip()
    curr_pos = openbracket+text[openbracket:].find('\n')+1
    while (line[len(line)-1] == '&'):
      line = text[curr_pos:curr_pos+text[curr_pos:].find('\n')].strip()
      curr_pos = curr_pos+text[curr_pos:].find('\n')+1
    curr_pos = curr_pos-1
    arglist = text[openbracket:curr_pos]
    #find the file containing the implementation
    subr_file =  os.popen('grep -Rilw --include "*.F90" --include "*.F" --exclude "*kernel.*" "subroutine '+fun_name+'\\b" . | head -n1').read().strip()
    if (len(subr_file) == 0) or (not os.path.exists(subr_file)):
      print 'Error, subroutine '+fun_name+' implementation not found in files, check parser!'
      exit(1)
    #read the file and find the implementation
    subr_fileh = open(subr_file,'r')
    subr_fileh_text = subr_fileh.read()
    if subr_file[len(subr_file)-1]=='F':
        subr_fileh_text = convert_F90(subr_fileh_text)
    subr_text = get_kernel(subr_fileh_text,fun_name)
    #get rid of comments and realgas calls
    subr_text = re.sub('\n*!.*\n','\n',subr_text)
    subr_text = re.sub('!.*\n','\n',subr_text)

    if attr <> '':
      subr_text = replace_consts(subr_text)
    subr_text = re.sub(r'(\n\s*)\bsubroutine\b\s+'+fun_name+r'\b', r'\1'+attr+' subroutine '+fun_name+'_gpu',subr_text,flags=re.IGNORECASE)
    my_subs = my_subs + '\n' + subr_text
    text1, text2 = find_function_calls(subr_text, attr, fun_name+'_gpu')
    children_subs = children_subs + '\n' + text1
    search_offset = i
    res=re.search('\\bcall\\b',text[search_offset:])
  funlist2[funlist_index]['function_text']=text
  return my_subs+children_subs, text
