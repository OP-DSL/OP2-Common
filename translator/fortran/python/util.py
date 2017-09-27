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

def get_stride_string(g_m,maps,mapnames,set_name):
  OP_ID   = 1;  OP_GBL   = 2;  OP_MAP = 3;
  if maps[g_m] == OP_ID:
    return 'direct_stride_OP2CONSTANT'
  if maps[g_m] == OP_GBL:
    return '(gridDim%x*blockDim%x)'
  else:
    idx = mapnames.index(mapnames[g_m])
    return 'opDat'+str(idx+1)+'_stride_OP2CONSTANT'


def replace_soa(text,nargs,soaflags,name,maps,accs,set_name,mapnames,repl_inc,hydra,bookleaf):
  OP_ID   = 1;  OP_GBL   = 2;  OP_MAP = 3;

  OP_READ = 1;  OP_WRITE = 2;  OP_RW  = 3;
  OP_INC  = 4;  OP_MAX   = 5;  OP_MIN = 6;
  #
  # Apply SoA to variable accesses
  #
  j = text.find(name+'_gpu')
  endj = arg_parse(text,j)
  while text[j] <> '(':
      j = j + 1
  arg_list = text[j+1:endj]
  arg_list = arg_list.replace('&','')
  varlist = ['']*nargs
  leading_dim = [-1]*nargs
  
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
          else:
            print 'Could not find shape specification for '+varlist[g_m]+' in '+name
          endarg = arg_parse(text,beginarg-1)
        else:
          print 'Error: array subscript not found'


        #If this is the first time we see the argument (i.e. its declaration)
        if leading_dim[g_m] == -1:
          if (len(text[beginarg:endarg].split(',')) > 1):
            #if it's 2D, remember leading dimension, and make it 1D
            leading_dim[g_m] = text[beginarg:endarg].split(',')[0]
            text = text[:beginarg] + '*'+' '*(endarg-beginarg-1) + text[endarg:]
          else:
            leading_dim[g_m] = 1
          #Continue search after this instance of the variable
          loc1 = loc1+i.start()+len(varlist[g_m])
        else:
          #If we have seen this variable already, then it's in the actual code, replace it with macro
          macro = 'OP2_SOA('+text[loc1+i.start():loc1+i.end()]+','
          if leading_dim[g_m] == 1:
            macro = macro + text[beginarg:endarg]
          else:
            macro = macro + text[beginarg:endarg].split(',')[0] + '+('+text[beginarg:endarg].split(',')[1]+'-1)*'+leading_dim[g_m]
          macro = macro + ', ' + get_stride_string(g_m,maps,mapnames,set_name) + ')'
          text = text[:loc1+i.start()] + macro + text[endarg+1:]
          #Continue search after this instance of the variable
          loc1 = loc1+i.start() + len(macro)

  return text

def find_function_calls(text, attr):
  global funlist
  search_offset = 0
  res=re.search('\\bcall\\b',text)
  my_subs = ''
  children_subs=''
  while (not (res is None)):
    i = search_offset + res.start() + 4
    #find name: whatever is in front of opening bracket
    openbracket = i+text[i:].find('(')
    fun_name = text[i:openbracket].strip()
    # #if hyd_dump, comment it out
    # if fun_name == hyd_dump:
    #   text = text[0:search_offset + res.start()]+'!'+text[search_offset + res.start():]
    #   search_offset = i
    #   res=re.search('\\bcall\\b',text[search_offset:])
    #   continue

    if fun_name.lower() in funlist:
      search_offset = i
      res=re.search('\\bcall\\b',text[search_offset:])
      continue

    #print fun_name

    funlist = funlist + [fun_name.lower()]
    #find signature
    line = text[openbracket:openbracket+text[openbracket:].find('\n')].strip()
    curr_pos = openbracket+text[openbracket:].find('\n')+1
    while (line[len(line)-1] == '&'):
      line = text[curr_pos:curr_pos+text[curr_pos:].find('\n')].strip()
      curr_pos = curr_pos+text[curr_pos:].find('\n')+1
    curr_pos = curr_pos-1
    arglist = text[openbracket:curr_pos]
    #find the file containing the implementation
    subr_file =  os.popen('grep -Rilw --include "*.F95" --exclude "*kernel.*" "subroutine '+fun_name+'" . | head -n1').read().strip()
    if (len(subr_file) == 0) or (not os.path.exists(subr_file)):
      print 'Error, subroutine '+fun_name+' implementation not found in files, check parser!'
      exit(1)
    #read the file and find the implementation
    subr_fileh = open(subr_file,'r')
    subr_fileh_text = subr_fileh.read()
    subr_fileh_text = re.sub('\n*!.*\n','\n',subr_fileh_text)
    subr_fileh_text = re.sub('!.*\n','\n',subr_fileh_text)
    subr_begin = subr_fileh_text.lower().find('subroutine '+fun_name.lower())
    #function name as spelled int he file
    fun_name = subr_fileh_text[subr_begin+11:subr_begin+11+len(fun_name)]
    subr_end = subr_fileh_text[subr_begin:].lower().find('end subroutine')
    if subr_end<0:
      print 'Error, could not find string "end subroutine" for implemenatation of '+fun_name+' in '+subr_file
      exit(-1)
    subr_end= subr_begin+subr_end
    subr_text =  subr_fileh_text[subr_begin:subr_end+14]
    if subr_text[10:len(subr_text)-20].lower().find('subroutine')>=0:
      print 'Error, could not properly parse subroutine, more than one encompassed '+fun_name+' in '+subr_file
      #print subr_text
      exit(-1)

    if attr == '':
      subr_text = subr_text.replace(')\n',')\n!$acc routine seq\n',1)
    subr_text = subr_text.replace('call hyd_','!call hyd_')
    subr_text = subr_text.replace('call op_','!call op_')
    writes = re.search('\\bwrite\\b',subr_text)
    writes_offset = 0
    while not (writes is None):
      writes_offset = writes_offset + writes.start()
      subr_text = subr_text[0:writes_offset]+'!'+subr_text[writes_offset:]
      writes_offset = writes_offset + subr_text[writes_offset:].find('\n')+1
      while (subr_text[writes_offset:].strip()[0] == '&'):
        subr_text = subr_text[0:writes_offset]+'!'+subr_text[writes_offset:]
        writes_offset = writes_offset + subr_text[writes_offset:].find('\n')+1
      writes = re.search('\\bwrite\\b',subr_text[writes_offset:])

    subr_text = replace_npdes(subr_text)
    if attr <> '':
      subr_text = replace_consts(subr_text)
    subr_text = subr_text.replace('subroutine '+fun_name, attr+' subroutine '+fun_name+'_gpu',1)
    my_subs = my_subs + '\n' + subr_text
    subr_text = re.sub('!.*\n','\n',subr_text)
    children_subs = children_subs + '\n' + find_function_calls(subr_text, attr)
    search_offset = i
    res=re.search('\\bcall\\b',text[search_offset:])
  return my_subs+children_subs
