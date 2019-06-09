##########################################################################
#
# Common code generation functions
#
# These functions are called from the target-specific code generators
#
##########################################################################


import re
import datetime
import glob
import os

def comment_remover(text):
    """Remove comments from text"""

    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return ''
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)

def remove_trailing_w_space(text):
  text = text+' '
  line_start = 0
  line = ""
  line_end = 0
  striped_test = ''
  count = 0
  while 1:
    line_end =  text.find("\n",line_start+1)
    line = text[line_start:line_end]
    line = line.rstrip()
    striped_test = striped_test + line +'\n'
    line_start = line_end + 1
    line = ""
    if line_end < 0:
      return striped_test[:-1]

def extract_includes(text):
  ## Find all '#includes ...' that are not inside of function declarations.

  includes = []

  include_pattern = r'([ \t]*#include[\s]+[\'\"<][\w\.]+[\'\">])'

  first_line_pattern = '(^)([^\n]+)'
  rest_of_lines_pattern = '([\n])([^\n]+)'
  line_pattern = re.compile(first_line_pattern + '|' + rest_of_lines_pattern)

  function_depth = 0
  for match in re.findall(line_pattern, text):
    if match[1] != "":
      line = match[1]
    else:
      line = match[3]

    ## Remove noise from the line to improve search for 
    ## entering and exiting of functions:
    line_clean = line
    # Remove escaped quotation character:
    line_clean = re.sub(r"\\\'", '', line_clean)
    line_clean = re.sub(r"\\\"", '', line_clean)
    # Remove quoted string in single line:
    line_clean = re.sub(r'"[^"]*"', '', line_clean)
    line_clean = re.sub(r"'[^']*'", '', line_clean)
    # Remove quoted string split over two lines:
    line_clean = re.sub(r'"[^"]*\\\n[^"]*"', '', line_clean)
    line_clean = re.sub(r"'[^']*\\\n[^']*'", '', line_clean)
    # Remove inline scoped logic ( {...} ):
    line_clean = re.sub(r'{[^{]*}', '', line_clean)

    function_depth += line_clean.count('{') - line_clean.count('}')
    if function_depth != 0:
      continue

    match = re.search(include_pattern, line)
    if match:
      includes.append(line)

  return includes

def para_parse(text, j, op_b, cl_b):
    """Parsing code block, i.e. text to find the correct closing brace"""

    depth = 0
    loc2 = j

    while 1:
      if text[loc2] == op_b:
            depth = depth + 1

      elif text[loc2] == cl_b:
            depth = depth - 1
            if depth == 0:
                return loc2
      loc2 = loc2 + 1

def replace_local_includes_with_file_contents(text, search_dir):
  ''' Replace occurences of '#include "<FILE>"' with <FILE> contents '''
  include_rgx = r'' + "^([\s]*)" + "#include" + "[\s]+" + '"([\w\.]+)"'

  text2 = ''
  for line in text.split('\n'):
    if not "#include" in line:
      text2 += line+'\n'
    else:
      include_item_filepath = ""
      matches = re.findall(include_rgx, line)[0]
      if len(matches) != 2:
        text2 += line+'\n'
      else:
        leading_whitespace = matches[0]
        include_item = matches[1]
        for r, d, f in os.walk(search_dir):
          for f_item in f:
            if f_item == include_item:
              include_item_filepath = os.path.join(r, f_item)
              break
          if include_item_filepath != "":
            break
        if include_item_filepath == "":
          print("Failed to locate file '{0}'".format(include_item))
          quit()
        f = open(include_item_filepath, 'r')
        include_file_text = f.read()
        f.close()
        include_file_text = comment_remover(include_file_text)
        for line in include_file_text.split('\n'):
          text2 += leading_whitespace + line+'\n'
  return text2

def get_stride_string(g_m,maps,mapnames,name):
  OP_ID   = 1;  OP_GBL   = 2;  OP_MAP = 3;
  if maps[g_m] == OP_ID:
    return 'direct_'+name+'_stride_OP2CONSTANT'
  if maps[g_m] == OP_GBL:
    return '(gridDim%x*blockDim%x)'
  else:
    idx = mapnames.index(mapnames[g_m])
    return 'opDat'+str(idx)+'_'+name+'_stride_OP2CONSTANT'

arithmetic_regex_pattern = r'^[ \(\)\+\-\*\\\.\%0-9]+$'

def op_parse_macro_defs(text):
  """Parsing for C macro definitions"""

  defs = {}
  macro_def_pattern = r'(\n|^)[ ]*(#define[ ]+)([A-Za-z0-9\_]+)[ ]+([0-9A-Za-z\_\.\+\-\*\/\(\) ]+)'
  for match in re.findall(macro_def_pattern, text):
    if len(match) < 4:
      continue
    elif len(match) > 4:
      print("Unexpected format for macro definition: " + str(match))
      continue
    key = match[2]
    value = match[3]
    defs[key] = value
    # print(key + " -> " + value)
  return defs

def self_evaluate_macro_defs(macro_defs):
  """Recursively evaluate C macro definitions that refer to other detected macros"""

  ## First, calculate the expected number of substitutions to perform:
  num_subs_expected = 0
  for k in macro_defs.keys():
    k_val = macro_defs[k]
    m = re.search(arithmetic_regex_pattern, k_val)
    if m != None:
      continue

    pattern = r'' + '([a-zA-Z0-9_]+)'
    occurences = re.findall(pattern, k_val)
    for o in occurences:
      m = re.search(arithmetic_regex_pattern, o)
      if m == None:
        if o in macro_defs.keys():
          num_subs_expected += 1

  substitutions_performed = True
  num_subs_performed = 0
  while substitutions_performed:
    substitutions_performed = False
    for k in macro_defs.keys():
      k_val = macro_defs[k]
      m = re.search(arithmetic_regex_pattern, k_val)
      if m != None:
        ## This macro definiton is numeric
        continue

      if k == k_val:
        del macro_defs[k]
        continue

      # print("Processing '{0}' -> '{1}'".format(k, k_val))

      ## If value of key 'k' depends on value of other
      ## keys, then substitute in value:
      for k2 in macro_defs.keys():
        if k == k2:
          continue

        pattern = r'' + '(^|[^a-zA-Z0-9_])' + k2 + '($|[^a-zA-Z0-9_])'
        m = re.search(pattern, k_val)

        if m != None:
          ## The macro "k" refers to macro "k2"
          k2_val = macro_defs[k2]

          m = re.search(arithmetic_regex_pattern, k2_val)
          if m == None:
            # 'k2_val' has not been resolved. Wait for this to occur before
            # substituting its value into 'k_val', as this minimises the total 
            # number of substitutions performed across all macros and so 
            # improves detection of infinite substitution loops.
            continue

          macro_defs[k] = re.sub(pattern, "\\g<1>"+k2_val+"\\g<2>", k_val)
          # print("- performing a substitution of '" + k2 + "'->'" + k2_val + "' into '" + k_val + "' to produce '" + macro_defs[k] + "'")
          k_val = macro_defs[k]
          substitutions_performed = True

          num_subs_performed += 1
          if num_subs_performed > num_subs_expected:
            print("WARNING: " + str(num_subs_performed) + " macro substitutions performed, but expected " + str(num_subs_expected) + ", probably stuck in a loop.")
            return

  ## Evaluate any mathematical expressions:
  for k in macro_defs.keys():
    val = macro_defs[k]
    m = re.search(arithmetic_regex_pattern, val)
    if m != None:
      res = ""
      try:
        res = eval(val)
      except:
        pass
      if type(res) != type(""):
        if str(res) != val:
          # print("Replacing '" + val + "' with '" + str(res) + "'")
          macro_defs[k] = str(res)

def evaluate_macro_defs_in_string(macro_defs, string):
  """Recursively evaluate C macro definitions in 'string' """

  ## First, calculate the expected number of substitutions to perform:
  num_subs_expected = 0
  m = re.search(arithmetic_regex_pattern, string)
  if m == None:
    pattern = r'' + '([a-zA-Z0-9_]+)'
    occurences = re.findall(pattern, string)
    for o in occurences:
      m = re.search(arithmetic_regex_pattern, o)
      if m == None:
        if o in macro_defs.keys():
          num_subs_expected = num_subs_expected + 1

  resolved_string = string

  substitutions_performed = True
  num_subs_performed = 0
  while substitutions_performed:
    substitutions_performed = False
    for k in macro_defs.keys():
      k_val = macro_defs[k]

      k_pattern = r'' + r'' + '(^|[^a-zA-Z0-9_])' + k + '($|[^a-zA-Z0-9_])'
      m = re.search(k_pattern, resolved_string)
      if m != None:
        ## "string" contains a reference to macro "k", so substitute in its definition:
        resolved_string_new = re.sub(k_pattern, "\\g<1>"+k_val+"\\g<2>", resolved_string)
        # print("Performing a substitution of '" + k + "'->'" + k_val + "' into '" + resolved_string + "'' to produce '" + resolved_string_new + "'")
        resolved_string = resolved_string_new
        substitutions_performed = True

        num_subs_performed = num_subs_performed + 1
        if num_subs_performed > num_subs_expected:
          print("WARNING: " + str(num_subs_performed) + " macro substitutions performed, but expected " + str(num_subs_expected) + ", probably stuck in a loop.")
          return


  if re.search(arithmetic_regex_pattern, resolved_string) != None:
    res = ""
    try:
      res = eval(resolved_string)
    except:
      return resolved_string
    else:
      if type(res) != type(""):
        resolved_string = str(res)

  return resolved_string

def create_kernel_info(kernel, inc_stage = 0):
    OP_ID   = 1;  OP_GBL   = 2;  OP_MAP = 3;

    OP_READ = 1;  OP_WRITE = 2;  OP_RW  = 3;
    OP_INC  = 4;  OP_MAX   = 5;  OP_MIN = 6;

    name  = kernel['name']
    nargs = kernel['nargs']
    dims  = kernel['dims']
    maps  = kernel['maps']
    var   = kernel['var']
    typs  = kernel['typs']
    accs  = kernel['accs']
    idxs  = kernel['idxs']
    inds  = kernel['inds']
    soaflags = kernel['soaflags']
    optflags = kernel['optflags']
    decl_filepath = kernel['decl_filepath']

    ninds   = kernel['ninds']
    inddims = kernel['inddims']
    indaccs = kernel['indaccs']
    indtyps = kernel['indtyps']
    invinds = kernel['invinds']
    mapnames = kernel['mapnames']
    invmapinds = kernel['invmapinds']
    mapinds = kernel['mapinds']

    nmaps = 0
    if ninds > 0:
      nmaps = max(mapinds)+1
    nargs_novec = nargs

    vec =  [m for m in range(0,nargs) if int(idxs[m])<0 and maps[m] == OP_MAP]
    if len(vec) > 0:
      unique_args = [1];
      vec_counter = 1;
      vectorised = []
      new_dims = []
      new_maps = []
      new_vars = []
      new_typs = []
      new_accs = []
      new_idxs = []
      new_inds = []
      new_soaflags = []
      new_optflags = []
      new_mapnames = []
      for m in range(0,nargs):
          if int(idxs[m])<0 and maps[m] == OP_MAP:
            if m > 0:
              unique_args = unique_args + [len(new_dims)+1]
            temp = [0]*(-1*int(idxs[m]))
            for i in range(0,-1*int(idxs[m])):
              temp[i] = var[m]
            new_vars = new_vars+temp
            for i in range(0,-1*int(idxs[m])):
              temp[i] = typs[m]
            new_typs = new_typs+temp
            for i in range(0,-1*int(idxs[m])):
              temp[i] = dims[m]
            new_dims = new_dims+temp
            new_maps = new_maps+[maps[m]]*int(-1*int(idxs[m]))
            new_mapnames = new_mapnames+[mapnames[m]]*int(-1*int(idxs[m]))
            new_soaflags = new_soaflags+[soaflags[m]]*int(-1*int(idxs[m]))
            new_optflags = new_optflags+[optflags[m]]*int(-1*int(idxs[m]))
            new_accs = new_accs+[accs[m]]*int(-1*int(idxs[m]))
            for i in range(0,-1*int(idxs[m])):
              new_idxs = new_idxs+[i]
            new_inds = new_inds+[inds[m]]*int(-1*int(idxs[m]))
            vectorised = vectorised + [vec_counter]*int(-1*int(idxs[m]))
            vec_counter = vec_counter + 1;
          else:
            if m > 0:
              unique_args = unique_args + [len(new_dims)+1]
            new_dims = new_dims+[dims[m]]
            new_maps = new_maps+[maps[m]]
            new_mapnames = new_mapnames+[mapnames[m]]
            new_accs = new_accs+[int(accs[m])]
            new_soaflags = new_soaflags+[soaflags[m]]
            new_optflags = new_optflags+[optflags[m]]
            new_idxs = new_idxs+[int(idxs[m])]
            new_inds = new_inds+[inds[m]]
            new_vars = new_vars+[var[m]]
            new_typs = new_typs+[typs[m]]
            vectorised = vectorised+[0]
      dims = new_dims
      maps = new_maps
      mapnames = new_mapnames
      accs = new_accs
      idxs = new_idxs
      inds = new_inds
      var = new_vars
      typs = new_typs
      soaflags = new_soaflags;
      optflags = new_optflags;
      nargs = len(vectorised);
      mapinds = [0]*nargs
      for i in range(0,nargs):
        mapinds[i] = i
        for j in range(0,i):
          if (maps[i] == OP_MAP) and (mapnames[i] == mapnames[j]) and (idxs[i] == idxs[j]):
            mapinds[i] = mapinds[j]

      for i in range(1,ninds+1):
        for index in range(0,len(inds)+1):
          if inds[index] == i:
            invinds[i-1] = index
            break
      invmapinds = invinds[:]
      for i in range(0,ninds):
        for j in range(0,i):
          if (mapnames[invinds[i]] == mapnames[invinds[j]]):
            invmapinds[i] = invmapinds[j]
    else:
      vectorised = [0]*nargs
      unique_args = range(1,nargs+1)

    cumulative_indirect_index = [-1]*nargs;
    j = 0;
    for i in range (0,nargs):
      if maps[i] == OP_MAP and ((not inc_stage) or accs[i] == OP_INC):
        cumulative_indirect_index[i] = j
        j = j + 1

    return name, nargs, dims, maps, var, typs, accs, idxs, inds, soaflags, optflags, decl_filepath, \
          ninds, inddims, indaccs, indtyps, invinds, mapnames, invmapinds, mapinds, nmaps, nargs_novec, \
          unique_args, vectorised, cumulative_indirect_index
