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
          num_subs_expected = num_subs_expected + 1

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
          # print("Performing a substitution of '" + k2 + "'->'" + k2_val + "' into '" + k_val + "' to produce '" + macro_defs[k] + "'")
          substitutions_performed = True

          num_subs_performed = num_subs_performed + 1
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
